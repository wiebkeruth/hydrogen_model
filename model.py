

""""
Function: Builds the optimisation model from all component modules and solves it.
Zielfunktion ist die Deckungsbeitragsmaximierung nach Gleichung 5.1 bei
exogen vorgegebenem Wasserstoffpreis - siehe set_objective.

Created on Fri Jun 12

@author: Wiebke G
"""

import time
import pyomo.environ as pyo

from Configuration import ElyConfig, BessConfig, PPAConfig, RfnboConfig, MarketConfig, EconomicsConfig
from ely import compute_ely_params, add_ely_variables, add_ely_constraints
from bess import compute_bess_params, add_bess_variables, add_bess_constraints
from markets import add_market_variables, apply_market_bounds, add_market_constraints


# ==================================
# MODELLING PARAMETERS
# ==================================

SOLVER_NAME    = "gurobi_direct"
SOLVER_THREADS = 16
MIP_REL_GAP    = 0.01    # relative optimality gap
EXTRA_SOLVER_OPTIONS = {}  # optional override hook for scripted sweeps, see solve_period


# ==================================
# MODEL ASSEMBLY
# ==================================

def build_model(df,
                ely: ElyConfig,
                bess: BessConfig,
                ppa: PPAConfig,
                rfnbo: RfnboConfig,
                market: MarketConfig,
                economics: EconomicsConfig) -> pyo.ConcreteModel:
    """
    Assemble the full Pyomo model for one period.

    Parameters
    ----------
    df        : pd.DataFrame     – Time series for the period (already sliced).
    ely       : ElyConfig        – Electrolyser configuration.
    bess      : BessConfig       – Battery configuration.
    ppa       : PPAConfig        – PPA prices and P_nom_wind/pv upper bounds.
    rfnbo     : RfnboConfig      – RFNBO config (mode, correlation).
    market    : MarketConfig     – Market toggle switches.
    economics : EconomicsConfig  – Financing assumptions (WACC inputs for compute_ely_params).

    Returns
    -------
    pyo.ConcreteModel – Model with all variables and constraints, no objective.
    """
    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=list(df.index))

    # Pre-calculations
    ely_params  = compute_ely_params(ely, economics)
    bess_params = compute_bess_params(bess, economics)

    # Variables (all modules first)
    add_ely_variables(model, ely)
    add_bess_variables(model, bess)
    add_market_variables(model, ppa)

    # Market toggles & RFNBO bounds
    apply_market_bounds(model, market)

    # Constraints (after all variables exist)
    add_ely_constraints(model, ely, ely_params, df)
    add_bess_constraints(model, bess, bess_params, df)
    add_market_constraints(model, rfnbo, market, df)

    print(f"  Variables:   {model.nvariables()}")
    print(f"  Constraints: {model.nconstraints()}")
    return model


# ==================================
# OBJECTIVE (Deckungsbeitragsmaximierung, Gleichung 5.1)
# ==================================

def set_objective(model, df,
                  ppa: PPAConfig,
                  economics: EconomicsConfig) -> None:
    """
    (Re)define the objective: Deckungsbeitragsmaximierung nach Gleichung 5.1.

        max  E_H2 + E_DA + R_13k - C_PPA - C_DA

    Bewusst OHNE CAPEX und OPEX von Elektrolyseur, BESS und Netzanschluss:
    deren Dimensionierung ist exogen vorgegeben (siehe Kapitel 1.2.2), die
    zugehoerigen Kapital- und Betriebskosten sind im Optimierungsproblem
    also Konstanten und beeinflussen die optimale Loesung nicht. Sie gehen
    erst nachgelagert in die LCOH-Berechnung ein (Lcoh.py). Endogen ist
    allein die PPA-Dimensionierung - deren Kosten stecken in C_PPA, weil der
    PPA-Vertrag als pay-as-produced auf die Erzeugung bezahlt wird.

    Alle Groessen in k€ (price_h2 [€/kg] * H2 [t]: die Umrechnungen kg->t
    und €->k€ kuerzen sich gegenseitig weg).

    Parameters
    ----------
    model     : pyo.ConcreteModel  – Model to attach the objective to.
    df        : pd.DataFrame       – Period time series.
    ppa       : PPAConfig          – PPA prices.
    economics : EconomicsConfig    – Assumed hydrogen sales price.
    """
    if hasattr(model, "objective"):
        model.del_component(model.objective)

    def _objective(model):
        revenue_h2 = economics.price_h2 * sum(model.H2[t] for t in model.T)

        revenue_da = sum(
            (model.ppa_to_da[t] + model.bess_to_da[t]) * df["da_price"][t]
            for t in model.T
        )

        reimbursement_rd = sum(
            model.rd_to_ely[t] * df["rd_reimbursement"][t]
            for t in model.T
        )

        # C_PPA (Gleichung 5.5): faellt auf die gesamte vertraglich gebundene
        # Erzeugung an - unabhaengig davon, ob der Strom genutzt, vermarktet
        # oder abgeregelt wird (pay-as-produced).
        cost_ppa = sum(
            df["cf_pv"][t]   * model.P_nom_pv   * ppa.price_pv
            + df["cf_wind"][t] * model.P_nom_wind * ppa.price_wind
            for t in model.T
        )

        # C_DA (Gleichung 5.6): Strombezug fuer Elektrolyseur und Speicher
        # zum jeweiligen Day-Ahead-Preis.
        cost_da = sum(
            (model.da_to_ely[t] + model.da_to_bess[t]) * df["da_price"][t]
            for t in model.T
        )

        return revenue_h2 + revenue_da + reimbursement_rd - cost_ppa - cost_da

    model.objective = pyo.Objective(rule=_objective, sense=pyo.maximize)


# ==================================
# SOLVE ONE PERIOD
# ==================================

def _warmstart_faehig(solver) -> bool:
    """
    Kann dieser Solver eine Startloesung verwerten?

    ACHTUNG: Die Pyomo-Methode heisst warm_start_capable() MIT Unterstrich.
    Ein Tippfehler hier faellt nicht auf, sondern schaltet den Warmstart
    stillschweigend ab - deshalb wird das Ergebnis in solve_period auch
    ausdruecklich protokolliert.
    """
    pruefung = getattr(solver, "warm_start_capable", None)
    if pruefung is None:
        return False
    try:
        return bool(pruefung())
    except Exception:
        return False

def solve_period(df,
                 ely: ElyConfig,
                 bess: BessConfig,
                 ppa: PPAConfig,
                 rfnbo: RfnboConfig,
                 market: MarketConfig,
                 economics: EconomicsConfig,
                 warmstart_z: dict | None = None):
    """
    Loest das Modell fuer das gesamte Kalenderjahr in einem Zug.

    Parameters
    ----------
    df          : pd.DataFrame            – Zeitreihe des Betrachtungszeitraums.
    ely         : ElyConfig               – Electrolyser configuration.
    bess        : BessConfig              – Battery configuration.
    ppa         : PPAConfig               – PPA prices.
    rfnbo       : RfnboConfig             – RFNBO config.
    market      : MarketConfig            – Market toggles.
    economics   : EconomicsConfig         – Assumed hydrogen sales price.
    warmstart_z : dict | None             – Optionale Startloesung: {t: 0/1} fuer
        die Binaervariablen z_ely, ueblicherweise die Loesung der vorherigen
        Sweep-Stufe (siehe main.run_bess_sweep). Uebergeben werden bewusst NUR
        die Binaervariablen - der Solver ergaenzt die stetigen Groessen selbst.
        Damit bleibt die Startloesung auch dann gueltig, wenn sie fuer die neue
        BESS-Groesse nicht mehr exakt passt (z. B. weil die alte
        Speicherfuellstandskurve die kleinere Kapazitaet ueberschreiten wuerde).

    Returns
    -------
    tuple (model, deckungsbeitrag, solve_time, converged, solver_status)
    """
    model = build_model(df, ely, bess, ppa, rfnbo, market, economics)
    set_objective(model, df, ppa, economics)

    solver = pyo.SolverFactory(SOLVER_NAME)
    if SOLVER_NAME in ("gurobi", "gurobi_direct"):
        solver.options["Threads"]   = SOLVER_THREADS
        solver.options["MIPGap"]    = MIP_REL_GAP
        # Optional override hook for scripted sweeps (e.g. a wall-clock time
        # limit for a coarse parameter sweep) - empty by default, so normal
        # single runs via main.py are completely unaffected. Set from
        # outside, e.g. `import model; model.EXTRA_SOLVER_OPTIONS = {"TimeLimit": 900}`.
        for _k, _v in EXTRA_SOLVER_OPTIONS.items():
            solver.options[_k] = _v
    else:
        # Fallback fuer HiGHS (appsi_highs/highs), falls kein Gurobi verfuegbar ist.
        solver.highs_options["threads"]     = SOLVER_THREADS
        solver.highs_options["presolve"]    = "on"
        solver.highs_options["mip_rel_gap"] = MIP_REL_GAP
        for _k, _v in EXTRA_SOLVER_OPTIONS.items():
            solver.highs_options[_k] = _v

    # --- Warmstart: Startloesung fuer die Binaervariablen uebergeben -------
    solve_kwargs = {"tee": True}
    if warmstart_z:
        n = 0
        for t in model.T:
            if t in warmstart_z:
                model.z_ely[t].value = warmstart_z[t]
                n += 1
        if _warmstart_faehig(solver):
            solve_kwargs["warmstart"] = True
            print(f"  Warmstart: Startloesung fuer {n} Binaervariablen uebergeben")
        else:
            print(f"  Warmstart: {SOLVER_NAME} unterstuetzt keine Startloesung "
                  f"- wird ignoriert (nur Gurobi nutzt sie)")

    start = time.time()
    result = solver.solve(model, **solve_kwargs)
    solve_time = time.time() - start

    tc = result.solver.termination_condition
    converged = tc in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible)
    print(f"Solver finished ({solve_time:.1f}s) – {tc}")

    # Zielfunktionswert = Deckungsbeitrag nach Gleichung 5.1 (ohne CAPEX/OPEX
    # von Ely, BESS und Netzanschluss - siehe set_objective). Die LCOH werden
    # nachgelagert aus der optimalen Loesung berechnet (Lcoh.py, aufgerufen
    # aus main.py), nicht hier.
    deckungsbeitrag = pyo.value(model.objective)
    print(f"  Deckungsbeitrag = {deckungsbeitrag:.2f} k€")

    return model, deckungsbeitrag, solve_time, converged, str(tc)