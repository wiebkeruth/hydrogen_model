

""""
Function: Builds the optimisation model from all component modules and solves it
for maximum profit at a fixed assumed hydrogen sales price.

Created on Fri Jun 12

@author: Wiebke G
"""

import math
import time
import pyomo.environ as pyo

from Configuration import ElyConfig, BessConfig, PPAConfig, RfnboConfig, MarketConfig, EconomicsConfig, Redispatch13kConfig
from ely import compute_ely_params, add_ely_variables, add_ely_constraints
from bess import compute_bess_params, add_bess_variables, add_bess_constraints
from markets import add_market_variables, apply_market_bounds, add_market_constraints


# ==================================
# MODELLING PARAMETERS
# ==================================

SOLVER_NAME    = "appsi_highs"
SOLVER_THREADS = 16
MIP_REL_GAP    = 0.02    # relative optimality gap
GAP_ABSOLUTE   = 100.0   # [k€]


# ==================================
# MODEL ASSEMBLY
# ==================================

def build_model(df,
                ely: ElyConfig,
                bess: BessConfig,
                ppa: PPAConfig,
                rfnbo: RfnboConfig,
                market: MarketConfig,
                economics: EconomicsConfig,
                time_share: float) -> pyo.ConcreteModel:
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
    ely_params = compute_ely_params(ely, economics)

    # Variables (all modules first)
    add_ely_variables(model, ely)
    add_bess_variables(model, bess)
    #model.P_nom_bess.fix(0)
    add_market_variables(model, ppa)

    # Market toggles & RFNBO bounds
    apply_market_bounds(model, market)

    # Constraints (after all variables exist)
    add_ely_constraints(model, ely, ely_params, time_share)
    add_bess_constraints(model, bess, df, time_share)
    add_market_constraints(model, rfnbo, market, df)

    print(f"  Variables:   {model.nvariables()}")
    print(f"  Constraints: {model.nconstraints()}")
    return model


# ==================================
# OBJECTIVE (profit maximisation)
# ==================================

def set_objective(model, df,
                  ppa: PPAConfig,
                  ely: ElyConfig,
                  bess: BessConfig,
                  ely_params,
                  bess_params,
                  economics: EconomicsConfig,
                  redispatch: Redispatch13kConfig,
                  time_share: float) -> None:
    """
    (Re)define the profit objective for a fixed assumed hydrogen price.

    Maximises:  revenue_h2 - total_cost

    Parameters
    ----------
    model       : pyo.ConcreteModel      – Model to attach the objective to.
    df          : pd.DataFrame           – Period time series.
    ppa         : PPAConfig              – PPA prices.
    ely         : ElyConfig              – Electrolyser configuration (p_nom, for grid connection CAPEX).
    bess        : BessConfig             – Battery configuration.
    ely_params  : ElyParams              – Pre-calculated electrolyser parameters.
    bess_params : BessParams             – Pre-calculated battery parameters.
    economics   : EconomicsConfig        – Assumed hydrogen sales price.
    redispatch  : Redispatch13kConfig    – Grid connection CAPEX.
    time_share  : float                  – Share of the year (month/8784 or 1.0).
    """
    if hasattr(model, "objective"):
        model.del_component(model.objective)

    def _objective(model):
        cost_ppa = sum(
            df["cf_pv"][t]   * model.P_nom_pv   * ppa.price_pv
            + df["cf_wind"][t] * model.P_nom_wind * ppa.price_wind
            for t in model.T
        )
        cost_rd = sum(
            -model.rd_to_ely[t] * df["rd_reimbursement"][t]
            for t in model.T
        )
        cost_da_buy = sum(
            (model.da_to_ely[t] + model.da_to_bess[t]) * df["da_price"][t] *1.05
            for t in model.T
        )
        revenue_da_sell = sum(
            (model.ppa_to_da[t] + model.bess_to_da[t]) * df["da_price"][t]*0.95
            for t in model.T
        )
        # Ely annual cost. P_nom_ely is a Pyomo Var - fixed to ely.p_nom when
        # ely.p_nom_variable is False, so these expressions are exact in
        # both modes (mirrors the BESS terms below, and replaces the
        # precomputed ely_params.capex_annual/opex_annual, which are only
        # valid at a single fixed ely.p_nom).
        capex_annual_ely = ely.capex * model.P_nom_ely * ely_params.annuity_factor
        opex_annual_ely  = ely.opex  * ely.capex * model.P_nom_ely

        # BESS annual cost (P_nom_bess is a variable)
        capex_annual_bess = bess.capex * model.P_nom_bess * bess_params.annuity_factor
        opex_annual_bess  = bess.opex  * model.P_nom_bess

        # Grid connection / transformer CAPEX, sized off the combined ely +
        # BESS connected capacity (see Redispatch13kConfig.gridconnection_capex/
        # EconomicsConfig.trafo_capex) and annuitized with the same WACC-based
        # annuity factor as the rest of the ely CAPEX. P_nom_ely/P_nom_bess
        # are decision variables, so this term is linear in them (constant
        # rate * capacity).
        capex_annual_grid = ((redispatch.gridconnection_capex + economics.trafo_capex)
                             * (model.P_nom_ely + model.P_nom_bess) * ely_params.annuity_factor)

        total_h2 = sum(model.H2[t] for t in model.T)
        # price_h2 [€/kg] * H2 [t]: the kg->t (x1000) and €->k€ (/1000)
        # conversions cancel, so this is already in k€ like every other term.
        revenue_h2 = economics.price_h2 * total_h2

        total_cost = (cost_ppa + cost_rd + cost_da_buy - revenue_da_sell
                      + time_share * (capex_annual_bess + opex_annual_bess
                                      + capex_annual_ely + opex_annual_ely
                                      + capex_annual_grid))

        return revenue_h2 - total_cost

    model.objective = pyo.Objective(rule=_objective, sense=pyo.maximize)


# ==================================
# SOLVE ONE PERIOD
# ==================================

def solve_period(df,
                 ely: ElyConfig,
                 bess: BessConfig,
                 ppa: PPAConfig,
                 rfnbo: RfnboConfig,
                 market: MarketConfig,
                 economics: EconomicsConfig,
                 redispatch: Redispatch13kConfig,
                 time_share: float):
    """
    Solve one period (month or full year) for maximum profit.

    Parameters
    ----------
    df         : pd.DataFrame            – Period time series (already sliced).
    ely        : ElyConfig               – Electrolyser configuration.
    bess       : BessConfig              – Battery configuration.
    ppa        : PPAConfig               – PPA prices.
    rfnbo      : RfnboConfig             – RFNBO config.
    market     : MarketConfig            – Market toggles.
    economics  : EconomicsConfig         – Assumed hydrogen sales price.
    redispatch : Redispatch13kConfig     – Grid connection CAPEX.
    time_share : float                   – Share of the year this period represents.

    Returns
    -------
    tuple (model, profit, lcoh, solve_time, converged, solver_status)
    """
    ely_params  = compute_ely_params(ely, economics)
    bess_params = compute_bess_params(bess, economics)

    model = build_model(df, ely, bess, ppa, rfnbo, market, economics, time_share)
    set_objective(model, df, ppa, ely, bess, ely_params, bess_params,
                  economics, redispatch, time_share)

    solver = pyo.SolverFactory(SOLVER_NAME)
    solver.highs_options["threads"]  = SOLVER_THREADS
    solver.highs_options["presolve"] = "on"

    start = time.time()
    result = solver.solve(model, tee=True)
    solve_time = time.time() - start

    tc = result.solver.termination_condition
    converged = tc in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible)
    print(f"Solver finished ({solve_time:.1f}s) - {tc}")

    # Achtung: profit enthaelt den fiktiven Erloesverlust der Stillstands-
    # stunden (H2 < 0, siehe ely.py). Die bereinigte Groesse steht in
    # extract_results als fiktiver_Erloesverlust_kEUR zur Verfuegung.
    profit = pyo.value(model.objective)

    lcoh = _compute_lcoh(model, df, ppa, ely, bess, ely_params, bess_params, economics, redispatch, time_share)
    print(f"  Profit = {profit:.2f} k EUR"
          + (f", realised LCOH = {lcoh:.4f} EUR/kg" if math.isfinite(lcoh) else " (no H2 produced)"))

    return model, profit, lcoh, solve_time, converged, str(tc)


# ==================================
# LCOH (post-solve evaluation)
# ==================================

def _compute_lcoh(model, df, ppa, ely, bess, ely_params, bess_params, economics, redispatch, time_share) -> float:
    """Compute the realised LCOH from the optimised solution."""
    p_nom_wind = max(0.0, pyo.value(model.P_nom_wind))
    p_nom_pv   = max(0.0, pyo.value(model.P_nom_pv))
    p_nom_bess = max(0.0, pyo.value(model.P_nom_bess))
    p_nom_ely  = max(0.0, pyo.value(model.P_nom_ely))

    cost_ppa = sum(df["cf_pv"][t] * p_nom_pv * ppa.price_pv
                   + df["cf_wind"][t] * p_nom_wind * ppa.price_wind
                   for t in model.T)
    cost_da_buy = sum((pyo.value(model.da_to_ely[t]) + pyo.value(model.da_to_bess[t]))
                      * df["da_price"][t] for t in model.T)
    revenue_da_sell = sum((pyo.value(model.ppa_to_da[t]) + pyo.value(model.bess_to_da[t]))
                          * df["da_price"][t] for t in model.T)
    cost_rd = sum(-pyo.value(model.rd_to_ely[t]) * df["rd_reimbursement"][t]
                  for t in model.T)
    total_cost = cost_ppa + cost_da_buy - revenue_da_sell + cost_rd

    capex_annual_ely  = ely.capex * p_nom_ely * ely_params.annuity_factor
    opex_annual_ely   = ely.opex  * ely.capex * p_nom_ely

    capex_annual_bess = bess.capex * p_nom_bess * bess_params.annuity_factor
    opex_annual_bess  = bess.opex  * p_nom_bess

    # H2 kann im Modus "absolute" negativ werden (fiktiver Erloesverlust in
    # Stillstandsstunden). Fuer die LCOH zaehlt nur die tatsaechlich erzeugte
    # Menge, deshalb wird hier auf null geklemmt - analog zu H2_bereinigt in
    # extract_results.
    h2_total = sum(max(0.0, pyo.value(model.H2[t])) for t in model.T)

    # Same grid connection / transformer annuity as in set_objective above.
    capex_annual_grid = ((redispatch.gridconnection_capex + economics.trafo_capex)
                         * (p_nom_ely + p_nom_bess) * ely_params.annuity_factor)

    fixed_cost = time_share * (capex_annual_ely + opex_annual_ely
                               + capex_annual_bess + opex_annual_bess
                               + capex_annual_grid)

    if h2_total < 1e-9:
        # No production – ratio is undefined.
        return math.inf

    return (fixed_cost + total_cost) / h2_total
