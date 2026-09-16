""""
Function: operating behaviour and limitations BESS

Created on Fri Jun 12

@author: Wiebke G
"""


from dataclasses import dataclass
import pyomo.environ as pyo
from Configuration import BessConfig, EconomicsConfig


# ==================================
# PRE-CALCULATIONS
# ==================================

@dataclass
class BessParams:
    """Derived financial and ageing parameters computed from BessConfig."""
    annuity_factor: float
    soh_avg:        float   # Ueber die Lebensdauer gemittelte Restkapazitaet [-], siehe compute_bess_params


def compute_bess_params(bess: BessConfig, economics: EconomicsConfig) -> BessParams:
    """
    Compute annuity factor for BESS.

    Note: Annual CAPEX and OPEX depend on P_nom_BESS, which is a Pyomo
    decision variable — they are therefore computed inside the objective
    function, not here.

    Parameters
    ----------
    bess      : BessConfig       – Battery configuration.
    economics : EconomicsConfig  – Financing assumptions (WACC inputs).

    Returns
    -------
    BessParams – Annuity factor for use in the objective function.
    """
    annuity_factor = compute_annuity(wacc=economics.wacc, lifetime=bess.lifetime)

    soh_avg = (1 + bess.capacity_eol_share) / 2

    return BessParams(annuity_factor=annuity_factor, soh_avg=soh_avg)


def compute_annuity(wacc: float, lifetime: float) -> float:
    return (wacc * (1 + wacc) ** lifetime) / ((1 + wacc) ** lifetime - 1)


# ==================================
# PYOMO VARIABLES
# ==================================

def add_bess_variables(model: pyo.ConcreteModel, bess: BessConfig) -> None:
    """Add all BESS-related Pyomo variables to the model."""
    # Die BESS-Leistung ist exogen vorgegeben (Kapitel 1.5: der Einfluss des
    # BESS wird ueber einen Sweep verschiedener fester Groessen untersucht,
    # nicht endogen mitoptimiert). Sie bleibt als fixierte Pyomo-Variable im
    # Modell, damit die Auswertung sie wie jede andere Groesse auslesen kann.
    model.P_nom_bess   = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, bess.p_nom))
    model.P_nom_bess.fix(bess.p_nom)

    model.soc          = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.bess_to_ely  = pyo.Var(model.T, domain=pyo.NonNegativeReals)

# ==================================
# PYOMO CONSTRAINTS
# ==================================

def add_bess_constraints(model: pyo.ConcreteModel, bess: BessConfig, params: "BessParams", df) -> None:
    """
    Add all BESS constraints to the model.

    Parameters
    ----------
    model      : pyo.ConcreteModel – Pyomo model with T and BESS variables defined.
    bess       : BessConfig        – Battery configuration.
    params     : BessParams        – Pre-calculated battery parameters (annuity_factor, soh_avg).
    df         : pd.DataFrame      – Period time series (needs 'timestamp'), used to group hours by calendar day for the daily cycle budget.
    """
    # Nutzbare Energiekapazitaet nach Alterung: die Nennkapazitaet aus der
    # C-Rate (P_nom_bess / c_rate) wird mit der ueber die Lebensdauer
    # gemittelten Restkapazitaet SOH_avg skaliert (siehe compute_bess_params).
    # Leistungsgrenzen (Lade-/Entladelimits weiter unten) bleiben bewusst
    # UNVERAENDERT an P_nom_bess gekoppelt - nur die Energie-Seite altert.
    def _capacity(model):
        return model.P_nom_bess / bess.c_rate * params.soh_avg

    def _charge_limit(model, t):
        return model.ppa_to_bess[t] + model.da_to_bess[t] <= model.P_nom_bess
    model.con_bess_charge_limit = pyo.Constraint(model.T, rule=_charge_limit)

    def _discharge_limit(model, t):
        return model.bess_to_ely[t] + model.bess_to_da[t] <= model.P_nom_bess
    model.con_bess_discharge_limit = pyo.Constraint(model.T, rule=_discharge_limit)

    def _soc(model, t):
        charge   = model.ppa_to_bess[t] * bess.eta_charge \
                 + model.da_to_bess[t]  * bess.eta_charge
        discharge = model.bess_to_ely[t] / bess.eta_discharge \
                  + model.bess_to_da[t]  / bess.eta_discharge
        if t == 0:
            return model.soc[t] == (
                bess.soc_initial_share * _capacity(model)
                + charge - discharge
            )
        return model.soc[t] == model.soc[t - 1] + charge - discharge
    model.con_bess_soc = pyo.Constraint(model.T, rule=_soc)

    def _soc_max(model, t):
        return model.soc[t] <= _capacity(model)
    model.con_bess_soc_max = pyo.Constraint(model.T, rule=_soc_max)

    # Entladetiefe: der Speicher darf nur bis auf soc_min_share der
    # (gealterten) Kapazitaet entladen werden - siehe BessConfig.soc_min_share.
    def _soc_min(model, t):
        return model.soc[t] >= bess.soc_min_share * _capacity(model)
    model.con_bess_soc_min = pyo.Constraint(model.T, rule=_soc_min)

    def _soc_end(model):
        last_t = max(model.T)
        return (model.soc[last_t]
                >= bess.soc_initial_share * _capacity(model))
    model.con_bess_soc_end = pyo.Constraint(rule=_soc_end)

    def _max_cycles(model):
        total_discharge = sum(
            (model.bess_to_ely[t] + model.bess_to_da[t]) / bess.eta_discharge
            for t in model.T
        )
        total_charge = sum(
            (model.ppa_to_bess[t] + model.da_to_bess[t]) * bess.eta_charge
            for t in model.T
        )
        capacity = _capacity(model)
        return (total_discharge + total_charge) <= bess.max_cycles_per_year * capacity*2
    model.con_bess_max_cycles = pyo.Constraint(rule=_max_cycles)

    # --- Daily cycle budget (independent of the annual budget above) ---
    dates = df["timestamp"].dt.date
    days  = dates.unique().tolist()
    T_by_day = {d: [t for t in model.T if dates[t] == d] for d in days}

    def _max_cycles_daily(model, d):
        total_discharge_day = sum(
            (model.bess_to_ely[t] + model.bess_to_da[t]) / bess.eta_discharge
            for t in T_by_day[d]
        )
        total_charge_day = sum(
            (model.ppa_to_bess[t] + model.da_to_bess[t]) * bess.eta_charge
            for t in T_by_day[d]
        )
        capacity = _capacity(model)
        return (total_discharge_day + total_charge_day) <= (bess.max_cycles_per_day * capacity*2)

    model.bess_day_set = pyo.Set(initialize=days)
    model.con_bess_max_cycles_daily = pyo.Constraint(model.bess_day_set, rule=_max_cycles_daily)

    def _combined_limit(model, t):
        return (model.ppa_to_bess[t] + model.da_to_bess[t]
            + model.bess_to_ely[t] + model.bess_to_da[t]) <= model.P_nom_bess
    model.con_bess_combined_limit = pyo.Constraint(model.T, rule=_combined_limit)