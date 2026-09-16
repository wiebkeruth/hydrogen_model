""""
Function: Defines all electricity market flows and their constraints

Created on Fri Jun 12

@author: Wiebke G
"""

import pandas as pd
import pyomo.environ as pyo
from Configuration import MarketConfig, PPAConfig, RfnboConfig


# ==================================
# PYOMO VARIABLES
# ==================================

def add_market_variables(model: pyo.ConcreteModel, ppa: PPAConfig) -> None:
    """
    Add all market-side flow variables to the model.

    These connect the generation sources (PPA, DA, RD) to the consumers
    (electrolyser, battery) and to the DA market for selling.

    Parameters
    ----------
    ppa : PPAConfig – PPA prices and P_nom_wind/pv upper bounds.
    """
    model.P_ppa       = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.ppa_to_ely  = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.ppa_to_da   = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.ppa_to_bess = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.da_to_ely   = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.da_to_bess  = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.rd_to_ely   = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.bess_to_da  = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    # Portion of rd_to_ely attributable to DA-sourced power (rd_to_ely itself
    # is channel-neutral - see con_rd_limit_sourced). Bounded above by both
    # rd_to_ely and da_to_ely (con_rd_da_ely_*) so it never exceeds what's
    # physically plausible; used to exempt RD-covered DA purchases from the
    # correlation constraints below, independent of the price threshold.
    model.rd_da_ely   = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    # Unused PPA power. Part of con_ppa_balance below - kept explicit (rather
    # than implicit slack) so it's visible in the results. Bounded to 0 by
    # default via MarketConfig.use_curtailment (see apply_market_bounds).
    model.ppa_curtailed = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # PPA capacity decision variables
    model.P_nom_wind  = pyo.Var(domain=pyo.NonNegativeReals, bounds=(10, ppa.p_nom_wind_max))
    model.P_nom_pv    = pyo.Var(domain=pyo.NonNegativeReals, bounds=(10, ppa.p_nom_pv_max))

    # Fixing these (rather than leaving them free) lets presolve eliminate
    # P_ppa[t] for every hour too (see PPAConfig.p_nom_wind_fixed/pv_fixed) -
    # useful for dispatch-only runs at a known plant size.
    if ppa.p_nom_wind_fixed is not None:
        model.P_nom_wind.fix(ppa.p_nom_wind_fixed)
    if ppa.p_nom_pv_fixed is not None:
        model.P_nom_pv.fix(ppa.p_nom_pv_fixed)


# ==================================
# MARKET TOGGLES & RFNBO BOUNDS
# ==================================

def apply_market_bounds(model: pyo.ConcreteModel,
                        market: MarketConfig) -> None:
    """
    Disable markets by fixing variable upper bounds to zero. Keeps the model
    linear. RFNBO correlation (price-dependent or not) is enforced entirely
    via the constraints in add_market_constraints, not here.

    Parameters
    ----------
    model  : pyo.ConcreteModel – Model with market variables already added.
    market : MarketConfig      – Market toggle switches.
    """

    # --- Market toggles ---
    if not market.use_ppa:
        model.P_nom_wind.setub(0)
        model.P_nom_pv.setub(0)
        print("Market toggle: PPA disabled")

    if not market.use_da_buy:
        for t in model.T:
            model.da_to_bess[t].setub(0)
        # da_to_ely is deliberately NOT capped to 0 here: it stays available
        # up to whatever rd_to_ely justifies (see con_da_for_rd_only in
        # add_market_constraints), so RD-covered DA purchases still work
        # even with general DA buying switched off. If use_rd is also off,
        # rd_to_ely is 0 anyway (see below), so da_to_ely ends up fully
        # blocked regardless - same net effect as before in that case.
        print("Market toggle: DA purchasing disabled (except to cover an RD need, if use_rd is on)")

    if not market.use_ppa_da_sell:
        for t in model.T:
            model.ppa_to_da[t].setub(0)
        print("Market toggle: PPA selling to DA disabled")

    if not market.use_bess_da_sell:
        for t in model.T:
            model.bess_to_da[t].setub(0)
        print("Market toggle: BESS selling to DA disabled")

    if not market.use_curtailment:
        for t in model.T:
            model.ppa_curtailed[t].setub(0)
        print("Market toggle: PPA curtailment disabled (all PPA power must be used)")

    if not market.use_rd:
        for t in model.T:
            model.rd_to_ely[t].setub(0)
        print("Market toggle: Redispatch disabled")

    # No price-based hard bound here anymore. Both correlation_mode="monthly"
    # (no price restriction at all - only the monthly PPA/DA balance matters)
    # and "hourly" (con_correlation_hourly requires ppa_to_da[t] >=
    # da_to_ely[t]+da_to_bess[t] specifically in hours with da_price >=
    # threshold) rely entirely on the constraints in add_market_constraints.
    # A setub(0) here on those same hours would zero out da_to_ely/da_to_bess
    # before con_correlation_hourly ever runs, making it vacuous - which was
    # the actual bug making "hourly" behave just like "monthly".


def compute_rfnbo_threshold(df: pd.DataFrame, rfnbo: RfnboConfig) -> pd.Series:
    """
    Effective RFNBO price threshold per hour: DA purchase counts as
    renewable-backed whenever the DA price is below EITHER the flat
    threshold OR a factor of the EUA price - i.e. below the higher of the
    two (an OR of two "price <= X" conditions is a "price <= max(...)").
    """
    return (rfnbo.eua_price_factor * df["eua_price"]).clip(lower=rfnbo.rfnbo_threshold)


# ==================================
# PYOMO CONSTRAINTS
# ==================================

def add_market_constraints(model: pyo.ConcreteModel,
                           rfnbo: RfnboConfig,
                           market: MarketConfig,
                           df: pd.DataFrame) -> None:
    """
    Add market balance and connection constraints.

    Requires model.P_ely (electrolyser) and model.bess_to_ely (battery)
    to already exist.

    Parameters
    ----------
    model  : pyo.ConcreteModel – Model with all variables defined.
    rfnbo  : RfnboConfig       – RFNBO config (correlation mode).
    market : MarketConfig      – Market toggle switches (use_da_buy, for con_da_for_rd_only).
    df     : pd.DataFrame      – Time series (needs cf_wind, cf_pv, rd_available).
    """

    def _ppa_generation(model, t):
        return model.P_ppa[t] == (df["cf_wind"][t] * model.P_nom_wind
                                  + df["cf_pv"][t] * model.P_nom_pv)
    model.con_ppa_generation = pyo.Constraint(model.T, rule=_ppa_generation)


    def _ppa_balance(model, t):
        return (model.ppa_to_ely[t] + model.ppa_to_da[t] + model.ppa_to_bess[t]
                + model.ppa_curtailed[t] == model.P_ppa[t])
    model.con_ppa_balance = pyo.Constraint(model.T, rule=_ppa_balance)

    # rd_to_ely is a reimbursement-accounting variable, not a physical supply
    # channel: redispatch electricity is still physically sourced via PPA/DA
    # (PPA preferentially, since it carries no marginal cost), it merely
    # qualifies for a rebate (see set_objective / extract_results).
    def _rd_limit_available(model, t):
        return model.rd_to_ely[t] <= df["rd_available"][t]
    model.con_rd_limit_available = pyo.Constraint(model.T, rule=_rd_limit_available)

    def _rd_limit_sourced(model, t):
        return model.rd_to_ely[t] <= model.ppa_to_ely[t] + model.da_to_ely[t]
    model.con_rd_limit_sourced = pyo.Constraint(model.T, rule=_rd_limit_sourced)

    # With general DA purchasing switched off, da_to_ely may still be used,
    # but only up to what rd_to_ely justifies that hour - i.e. DA buying is
    # allowed exclusively to realize an available redispatch rebate, not for
    # general procurement. See apply_market_bounds for the matching relaxed
    # bound (da_to_ely is no longer setub(0) there when use_da_buy is off).
    if not market.use_da_buy:
        def _da_for_rd_only(model, t):
            return model.da_to_ely[t] <= model.rd_to_ely[t]
        model.con_da_for_rd_only = pyo.Constraint(model.T, rule=_da_for_rd_only)

    # rd_da_ely caps how much of rd_to_ely may be treated as DA-sourced for
    # the correlation exemption below - never more than was actually
    # redispatch-compensated, and never more than was actually drawn from DA.
    def _rd_da_ely_le_rd(model, t):
        return model.rd_da_ely[t] <= model.rd_to_ely[t]
    model.con_rd_da_ely_le_rd = pyo.Constraint(model.T, rule=_rd_da_ely_le_rd)

    def _rd_da_ely_le_da(model, t):
        return model.rd_da_ely[t] <= model.da_to_ely[t]
    model.con_rd_da_ely_le_da = pyo.Constraint(model.T, rule=_rd_da_ely_le_da)

    # Connection equation: electrolyser power = sum of all physical inflows
    def _p_ely_definition(model, t):
        return model.P_ely[t] == (model.ppa_to_ely[t] + model.da_to_ely[t]
                                  + model.bess_to_ely[t])
    model.con_p_ely_definition = pyo.Constraint(model.T, rule=_p_ely_definition)

    # Correlation: PPA generation must cover grid consumption.
    #   monthly – aggregated over each calendar month (temporary RFNBO
    #             derogation).
    #   hourly  – matched every single hour, except hours where the DA price
    #             is below the RFNBO threshold (see compute_rfnbo_threshold),
    #             where DA purchase may break the hourly balance as long as
    #             the monthly aggregate (backstop) still holds.
    # In both cases, rd_da_ely is deducted from the DA-side demand: DA power
    # drawn to cover a redispatch need is exempt from needing PPA backing,
    # independent of the price threshold (con_rd_da_ely_* above caps this to
    # what's actually redispatch-compensated and DA-sourced).
    #
    # rfnbo.correlation_ely_only controls whether da_to_bess counts toward
    # that demand at all - when True, only the Ely's DA draw needs PPA
    # backing, so the BESS can buy/sell DA freely.
    def _da_demand(model, t):
        if rfnbo.correlation_ely_only:
            return model.da_to_ely[t] - model.rd_da_ely[t]
        return model.da_to_ely[t] + model.da_to_bess[t] - model.rd_da_ely[t]

    def _add_monthly_correlation(con_name: str):
        months = df["month"].unique().tolist()
        T_by_month = {m: [t for t in model.T if df["month"][t] == m] for m in months}

        def _rule(model, m):
            ppa_da_sum = sum(model.ppa_to_da[t] for t in T_by_month[m])
            demand_sum = sum(_da_demand(model, t) for t in T_by_month[m])
            return ppa_da_sum >= demand_sum

        model.add_component(f"{con_name}_month_set", pyo.Set(initialize=months))
        model.add_component(con_name, pyo.Constraint(getattr(model, f"{con_name}_month_set"), rule=_rule))

    if not rfnbo.enforce_correlation:
        print("Market toggle: RFNBO correlation constraint disabled")
    else:
        if rfnbo.correlation_ely_only:
            print("Market toggle: RFNBO correlation restricted to the Ely - BESS trades DA freely")

        if rfnbo.correlation_mode == "monthly":
            _add_monthly_correlation("con_correlation")

        elif rfnbo.correlation_mode == "hourly":
            threshold = compute_rfnbo_threshold(df, rfnbo)

            def _correlation_hourly(model, t):
                if df["da_price"][t] < threshold[t]:
                    return pyo.Constraint.Skip  # relaxed – covered by monthly backstop below
                return model.ppa_to_da[t] >= _da_demand(model, t)
            model.con_correlation_hourly = pyo.Constraint(model.T, rule=_correlation_hourly)

            _add_monthly_correlation("con_correlation_monthly_backstop")
        else:
            raise ValueError(f"Unknown correlation_mode: {rfnbo.correlation_mode}")

