""""
Function: Runs the optimisation for a given year, either month by month or for the
full year at once, extracts the key results per period, writes the
per-period time series to Excel and a summary table across all periods.

Created on Fri Jun 12

@author: Wiebke G
"""

import pandas as pd
import pyomo.environ as pyo
from datetime import datetime
from dataclasses import asdict, replace

from Configuration import (
    ElyConfig, BessConfig, PPAConfig,
    Redispatch13kConfig, RfnboConfig, MarketConfig, EconomicsConfig
)
from data  import load_data, slice_month, load_gridconnection_capex
from model import solve_period
from markets import compute_rfnbo_threshold
from cashflow import compute_cashflow, export_cashflow_excel
from Lcoh import compute_lcoh, jahreskosten, crf, _LABELS as LCOH_COLUMN_LABELS


# ==================================
# CONFIGURATION
# ==================================

YEAR = 2025 

MODE   = "annual"              # "monthly" → solve month by month | "annual"  → solve full year at once
MONTHS =  list(range(1, 13))   # which months to solve (monthly mode)

ely        = ElyConfig()
bess       = BessConfig()
ppa        = PPAConfig()
redispatch = Redispatch13kConfig()
redispatch.gridconnection_capex = load_gridconnection_capex(redispatch.rd_region)  # [k€/MW], looked up by rd_region
rfnbo      = RfnboConfig()
market     = MarketConfig()
economics  = EconomicsConfig()


# ==================================
# RESULT EXTRACTION
# ==================================

def extract_results(model, df, ely, rfnbo, index_offset =0) -> dict:
    """
    Pull the optimised values into the DataFrame and compute summary metrics.

    Parameters
    ----------
    model : pyo.ConcreteModel  – Solved model.
    df    : pd.DataFrame       – Period time series (will be extended in place).
    ely   : ElyConfig          – Electrolyser config (for FLH reference).
    rfnbo : RfnboConfig        – RFNBO config (for correlation validation).

    Returns
    -------
    dict – Summary metrics for this period.
    """
    T = [t + index_offset for t in range(len(df))]

    df["P_ppa_opt"]      = [pyo.value(model.P_ppa[t])       for t in T]
    df["ppa_to_ely_opt"] = [pyo.value(model.ppa_to_ely[t])  for t in T]
    df["ppa_to_da_opt"]  = [pyo.value(model.ppa_to_da[t])   for t in T]
    df["ppa_to_bess_opt"]= [pyo.value(model.ppa_to_bess[t]) for t in T]
    df["ppa_curtailed_opt"] = [pyo.value(model.ppa_curtailed[t]) for t in T]
    df["da_to_ely_opt"]  = [pyo.value(model.da_to_ely[t])   for t in T]
    df["da_to_bess_opt"] = [pyo.value(model.da_to_bess[t])  for t in T]
    df["rd_to_ely_opt"]  = [pyo.value(model.rd_to_ely[t])   for t in T]
    df["rd_da_ely_opt"]  = [pyo.value(model.rd_da_ely[t])   for t in T]
    df["bess_to_ely_opt"]= [pyo.value(model.bess_to_ely[t]) for t in T]
    df["bess_to_da_opt"] = [pyo.value(model.bess_to_da[t])  for t in T]
    df["soc_opt"]        = [pyo.value(model.soc[t])         for t in T]
    df["H2_opt"]         = [pyo.value(model.H2[t])          for t in T]
    df["P_ely_opt"]      = (df["ppa_to_ely_opt"] + df["da_to_ely_opt"]
                            + df["bess_to_ely_opt"])
    # price_h2 [€/kg] * H2 [t]: kg->t (x1000) and €->k€ (/1000) cancel, so
    # this is already in k€ like every other cost/revenue figure.
    # Im Modus "absolute" darf H2 negativ werden (fiktiver Erloesverlust in
    # Stillstandsstunden, siehe Modulkopf von ely.py). H2_opt bleibt roh,
    # damit es zur Zielfunktion passt; fuer jede Mengen- und Kostenauswertung
    # wird H2_bereinigt verwendet.
    df["H2_bereinigt"]   = df["H2_opt"].clip(lower=0.0)
    df["H2_fiktiv"]      = df["H2_opt"].clip(upper=0.0)          # <= 0
    df["H2_revenue_opt"] = economics.price_h2 * df["H2_bereinigt"]

    # Lastanteil der Anlage [-] - beim Monolithen einfach P_ely / p_nom
    df["load_opt"] = df["P_ely_opt"] / ely.p_nom if ely.p_nom > 0 else 0.0

    p_nom_wind = max(0.0, pyo.value(model.P_nom_wind))
    p_nom_pv   = max(0.0, pyo.value(model.P_nom_pv))
    p_nom_bess = max(0.0, pyo.value(model.P_nom_bess))
    p_nom_ely  = max(0.0, pyo.value(model.P_nom_ely))

    p_ely_total = df["P_ely_opt"].sum()
    h2_total    = df["H2_bereinigt"].sum()
    flh_ely     = p_ely_total / p_nom_ely if p_nom_ely > 1e-9 else 0.0

    cost_ppa = (df["cf_pv"] * p_nom_pv * ppa.price_pv
                + df["cf_wind"] * p_nom_wind * ppa.price_wind).sum()
    cost_da  = ((df["da_to_ely_opt"] + df["da_to_bess_opt"]) * df["da_price"]).sum()
    revenue_da = ((df["ppa_to_da_opt"] + df["bess_to_da_opt"]) * df["da_price"]).sum()
    cost_rd  = -(df["rd_to_ely_opt"] * df["rd_reimbursement"]).sum()
    total_electricity_cost = cost_da + cost_ppa - revenue_da + cost_rd

    revenue_h2    = df["H2_revenue_opt"].sum()
    revenue_total = revenue_h2 + revenue_da
    da_revenue_share_pct = revenue_da / revenue_total * 100 if revenue_total > 0 else 0.0

    # Mindestlastdiagnose. Die Mindestlast wird nicht als Nebenbedingung
    # erzwungen, sondern folgt aus der Effizienzkurve (siehe ely.py). Dieser
    # Test muss deshalb 0 ergeben - ein Wert > 0 hiesse, dass die Kurve den
    # Bereich doch zulaesst, und waere ein Befund fuer die Arbeit.
    sockel  = ely.min_load_share * ely.p_nom
    running = df["P_ely_opt"] > 1e-6
    below   = running & (df["P_ely_opt"] < sockel - 1e-6)
    n_below_min_load = int(below.sum())

    h_running    = int(running.sum())
    h_stillstand = int((~running).sum())
    load_mean    = float(df.loc[running, "load_opt"].mean()) if h_running else 0.0

    # --- POST-HOC ECONOMIC ADJUSTMENTS (reporting only - see main() for how
    # these feed into compute_cashflow/compute_lcoh; they do NOT change the
    # dispatch model or its solution, only how it is evaluated afterwards) ---

    # 1. H2 produced in hours running below minimum load doesn't count as
    # produced at all, unconditionally - zeroed at the quantity level (not
    # just its revenue), so it also drags down the realised efficiency
    # (h2_annual_t is used as the denominator of p_el_annual/h2_annual_t in
    # derive_capex_opex - see Lcoh.py/cashflow.py), which then correctly
    # propagates into every downstream year's projected production/revenue.
    df["H2_opt_adjusted"]  = df["H2_bereinigt"].where(~below, 0.0)
    h2_total_adjusted       = df["H2_opt_adjusted"].sum()
    revenue_h2_adjusted     = economics.price_h2 * h2_total_adjusted

    # 1b. Same treatment for FLH (Volllaststunden): hours running below
    # minimum load don't count as productive operation either, so
    # flh_ely_adjusted only credits power drawn above P_min - unlike
    # p_ely_total/P_ely_total (real electricity consumption/cost, kept
    # unadjusted, see main()), this is purely a reporting metric.
    p_ely_total_adjusted = df["P_ely_opt"].where(~below, 0.0).sum()
    flh_ely_adjusted = p_ely_total_adjusted / p_nom_ely if p_nom_ely > 1e-9 else 0.0

    # 2. Optional: credit curtailed PPA power as if it had been sold on the
    # DA market whenever da_price > 0 (economics.sell_curtailment_ex_post).
    if economics.sell_curtailment_ex_post:
        sellable_hours = df["da_price"] > 0
        curtailment_da_revenue = (df["ppa_curtailed_opt"] * df["da_price"] * sellable_hours).sum()
        curtailment_da_volume  = (df["ppa_curtailed_opt"] * sellable_hours).sum()
    else:
        curtailment_da_revenue = 0.0
        curtailment_da_volume  = 0.0
    revenue_da_adjusted = revenue_da + curtailment_da_revenue

    # POST-CHECK: SIMULATENOUS CHARGE/DISCHARGE

    charge    = df["ppa_to_bess_opt"] + df["da_to_bess_opt"]
    discharge = df["bess_to_ely_opt"] + df["bess_to_da_opt"]
    both_active = (charge > 0.001) & (discharge > 0.001)
    n_simultaneous = int(both_active.sum())

    if n_simultaneous == 0:
        print(f"  Post-check OK: no simultaneous charge/discharge")
    elif n_simultaneous / len(df) < 0.01:
        print(f"  Post-check: negligible simultaneous charge/discharge "
               f"in {n_simultaneous} h")
    else:
        print(f"  WARNING: simultaneous charge/discharge in {n_simultaneous} h")

    # POST-CHECK: RFNBO-KORRELATION (PPA-Erzeugung >= Netzbezug)
    # rd_da_ely is deducted here exactly as in the actual con_correlation*
    # constraints (markets.py): DA power drawn to cover a redispatch need is
    # exempt from needing PPA backing, so it must be added back on the
    # "supply" side here or this check reports false violations whenever RD
    # is used (see con_rd_da_ely_* / _add_monthly_correlation).
    # rfnbo.correlation_ely_only: when the enforced constraint only covers
    # the Ely's DA draw (BESS trades DA freely, see markets.py), mirror that
    # scope here too - otherwise this check would flag "violations" the
    # actual constraint was never designed to prevent.
    ppa_supply  = (df["ppa_to_ely_opt"] + df["ppa_to_bess_opt"] + df["ppa_to_da_opt"]
                  + df["rd_da_ely_opt"])
    grid_demand = df["ppa_to_ely_opt"] + df["da_to_ely_opt"]
    if not rfnbo.correlation_ely_only:
        grid_demand = grid_demand + df["ppa_to_bess_opt"] + df["da_to_bess_opt"]
    correlation_surplus = ppa_supply.sum() - grid_demand.sum()
    correlation_ok = correlation_surplus >= -1e-3

    n_hourly_correlation_violations = 0
    if rfnbo.correlation_mode == "hourly":
        threshold = compute_rfnbo_threshold(df, rfnbo)
        strict_hours = df["da_price"] >= threshold
        hourly_gap = (grid_demand - ppa_supply) * strict_hours
        n_hourly_correlation_violations = int((hourly_gap > 1e-4).sum())
        if n_hourly_correlation_violations == 0:
            print(f"  Post-check OK: stündliche RFNBO-Korrelation eingehalten")
        else:
            print(f"  WARNING: stündliche Korrelation verletzt in "
                  f"{n_hourly_correlation_violations} Stunden")

    if correlation_ok:
        print(f"  Post-check OK: RFNBO-Korrelation eingehalten "
              f"(Überschuss {correlation_surplus:.1f} MWh)")
    else:
        print(f"  WARNING: RFNBO-Korrelation verletzt "
              f"(Defizit {-correlation_surplus:.1f} MWh)")

    # --- AVERAGE PRICE METRICS [k€/MWh, same raw-da_price convention as
    # cost_da/revenue_da above - no 1.05/0.95 DA spread applied here, so
    # these stay consistent with the other cost/revenue figures reported ---

    # 1. Average purchase price, incl./excl. PPA. "Incl." blends PPA (charged
    # on generation, see cost_ppa) and DA buying into one volume-weighted
    # price; "excl." is the DA-only average buy price.
    da_bought_total = (df["da_to_ely_opt"] + df["da_to_bess_opt"]).sum()
    p_ppa_total     = df["P_ppa_opt"].sum()
    avg_buy_price_incl_ppa = ((cost_ppa + cost_da) / (p_ppa_total + da_bought_total)
                              if (p_ppa_total + da_bought_total) > 1e-9 else 0.0)
    avg_buy_price_excl_ppa = cost_da / da_bought_total if da_bought_total > 1e-9 else 0.0

    # 2. Average DA sell price (volume-weighted over everything sold to DA).
    # Includes the optional ex-post curtailment sale (economics.
    # sell_curtailment_ex_post, see above) - both its extra revenue AND its
    # extra volume, so it's a genuine weighted average, not just a revenue
    # bump against the unadjusted (smaller) real dispatch volume.
    da_sold_total = (df["ppa_to_da_opt"] + df["bess_to_da_opt"]).sum() + curtailment_da_volume
    avg_sell_price_da = revenue_da_adjusted / da_sold_total if da_sold_total > 1e-9 else 0.0

    # 3. Average electricity procurement price attributable to the Ely
    # specifically: sum_t[P_ely(t) * Strombezugskosten(t)] / sum_t P_ely(t).
    # PPA/DA/RD legs are priced with their actual per-hour rate; BESS-sourced
    # power (bess_to_ely) has no well-defined per-hour acquisition cost (it
    # was charged at a possibly different hour/price), so it's priced at the
    # PERIOD-AVERAGE BESS charging cost per MWh discharged (total charging
    # cost, PPA+DA, spread over total discharge to Ely+DA - this naturally
    # reflects the round-trip efficiency loss).
    ppa_rate = (df["cf_pv"] * p_nom_pv * ppa.price_pv
               + df["cf_wind"] * p_nom_wind * ppa.price_wind) / df["P_ppa_opt"].where(df["P_ppa_opt"] > 1e-9)
    ppa_rate = ppa_rate.fillna(0.0)

    cost_ely_ppa  = (df["ppa_to_ely_opt"] * ppa_rate).sum()
    cost_ely_da   = (df["da_to_ely_opt"] * df["da_price"]).sum()
    credit_ely_rd = (df["rd_to_ely_opt"] * df["rd_reimbursement"]).sum()

    # PPA is paid on generation, not usage (see cost_ppa above) - curtailed
    # PPA power (ppa_curtailed) is still fully paid for even though nobody
    # draws it. It's attributed entirely to the Ely here (not split off to
    # the BESS), since the Ely is the plant's primary offtaker and the PPA
    # is sized/contracted around serving it - i.e. curtailment is treated as
    # a cost of securing the Ely's supply, not a cost-free non-event.
    cost_ely_curtailment = (df["ppa_curtailed_opt"] * ppa_rate).sum()

    bess_charge_cost_hourly = (df["ppa_to_bess_opt"] * ppa_rate
                               + df["da_to_bess_opt"] * df["da_price"])
    total_bess_discharge = (df["bess_to_ely_opt"] + df["bess_to_da_opt"]).sum()
    avg_bess_cost_per_mwh_out = (bess_charge_cost_hourly.sum() / total_bess_discharge
                                 if total_bess_discharge > 1e-9 else 0.0)
    cost_ely_bess = df["bess_to_ely_opt"].sum() * avg_bess_cost_per_mwh_out

    cost_ely_total = (cost_ely_ppa + cost_ely_da - credit_ely_rd
                      + cost_ely_bess + cost_ely_curtailment)
    avg_price_ely  = cost_ely_total / p_ely_total if p_ely_total > 1e-9 else 0.0

    # 4. Further volume-weighted average prices, each restricted to one
    # specific flow (all [k€/MWh], same raw-da_price convention as above).
    avg_ppa_cost_incl_curtailment = cost_ppa / p_ppa_total if p_ppa_total > 1e-9 else 0.0

    ppa_to_da_total = df["ppa_to_da_opt"].sum()
    avg_da_price_ppa_to_da = ((df["ppa_to_da_opt"] * df["da_price"]).sum() / ppa_to_da_total
                              if ppa_to_da_total > 1e-9 else 0.0)

    avg_da_price_buy = avg_buy_price_excl_ppa    # same figure, see 1. above

    bess_to_da_total = df["bess_to_da_opt"].sum()
    avg_da_price_bess_sell = ((df["bess_to_da_opt"] * df["da_price"]).sum() / bess_to_da_total
                              if bess_to_da_total > 1e-9 else 0.0)

    rd_to_ely_total = df["rd_to_ely_opt"].sum()
    avg_rd_reimbursement = (-cost_rd / rd_to_ely_total
                            if rd_to_ely_total > 1e-9 else 0.0)

    # 5. Average GROSS cost of RD-eligible consumption, split by source -
    # i.e. what the underlying PPA/DA electricity itself cost, before the
    # RD rebate (avg_rd_reimbursement above). rd_da_ely is the model's own
    # "DA-sourced share of rd_to_ely" variable (con_rd_da_ely_* in
    # markets.py); the PPA-sourced share is simply the rest of rd_to_ely.
    rd_da_ely_total  = df["rd_da_ely_opt"].sum()
    rd_ppa_ely_opt   = df["rd_to_ely_opt"] - df["rd_da_ely_opt"]
    rd_ppa_ely_total = rd_ppa_ely_opt.sum()

    avg_rd_cost_ppa = ((rd_ppa_ely_opt * ppa_rate).sum() / rd_ppa_ely_total
                       if rd_ppa_ely_total > 1e-9 else 0.0)
    avg_rd_cost_da  = ((df["rd_da_ely_opt"] * df["da_price"]).sum() / rd_da_ely_total
                       if rd_da_ely_total > 1e-9 else 0.0)

    # General figure: blended gross cost across both sources together.
    avg_rd_cost_total = ((rd_ppa_ely_opt * ppa_rate + df["rd_da_ely_opt"] * df["da_price"]).sum()
                         / rd_to_ely_total if rd_to_ely_total > 1e-9 else 0.0)

    return {
        "FLH_ely":          flh_ely,
        "FLH_ely_adjusted": flh_ely_adjusted,
        "H2_amount":        h2_total,
        "H2_amount_adjusted": h2_total_adjusted,
        "P_ely_total":      p_ely_total,
        "P_nom_wind":       p_nom_wind,
        "P_nom_pv":         p_nom_pv,
        "P_nom_bess":       p_nom_bess,
        "P_nom_ely":        p_nom_ely,
        "cost_ppa":         cost_ppa,
        "cost_da":          cost_da,
        "cost_rd":          cost_rd,
        "revenue_da":       revenue_da,
        "revenue_h2":       revenue_h2,
        "revenue_total":    revenue_total,
        "revenue_h2_adjusted": revenue_h2_adjusted,
        "revenue_da_adjusted": revenue_da_adjusted,
        "curtailment_da_revenue_expost": curtailment_da_revenue,
        "da_revenue_share_pct": da_revenue_share_pct,
        "total_elec_cost":  total_electricity_cost,
        "rd_share_pct":     df["rd_to_ely_opt"].sum() / p_ely_total * 100
                            if p_ely_total > 0 else 0.0,
        "avg_electricity_costs":    (total_electricity_cost + revenue_da) / p_ely_total
                                    if p_ely_total > 0 else 0.0,  # k€/MWh, gross (excl. DA sales revenue)
        "avg_electricity_costs_revenues":    total_electricity_cost / p_ely_total
                                    if p_ely_total > 0 else 0.0,  # k€/MWh, net (incl. DA sales revenue)
        "avg_buy_price_incl_ppa":  avg_buy_price_incl_ppa,   # k€/MWh
        "avg_buy_price_excl_ppa":  avg_buy_price_excl_ppa,   # k€/MWh
        "avg_sell_price_da":       avg_sell_price_da,        # k€/MWh
        "avg_price_ely":           avg_price_ely,            # k€/MWh
        "avg_ppa_cost_incl_curtailment": avg_ppa_cost_incl_curtailment,  # k€/MWh
        "avg_da_price_ppa_to_da":        avg_da_price_ppa_to_da,        # k€/MWh
        "avg_da_price_buy":              avg_da_price_buy,              # k€/MWh
        "avg_da_price_bess_sell":        avg_da_price_bess_sell,        # k€/MWh
        "avg_rd_reimbursement":          avg_rd_reimbursement,          # k€/MWh
        "avg_rd_cost_ppa":               avg_rd_cost_ppa,               # k€/MWh, gross, PPA-sourced RD share
        "avg_rd_cost_da":                avg_rd_cost_da,                # k€/MWh, gross, DA-sourced RD share
        "avg_rd_cost_total":             avg_rd_cost_total,             # k€/MWh, gross, blended PPA+DA
        "h_below_min_load":          n_below_min_load,
        "h_stillstand":              h_stillstand,
        "h_running":                 h_running,
        "load_mean":                 load_mean,
        "H2_fiktiv_negativ_t":       float(df["H2_fiktiv"].sum()),
        "fiktiver_Erloesverlust_kEUR": float(-economics.price_h2 * df["H2_fiktiv"].sum()),
        "correlation_ok":            correlation_ok,
        "correlation_surplus_MWh":   correlation_surplus,
        "n_hourly_correlation_violations": n_hourly_correlation_violations,
    }

def export_timeseries(df, label) -> None:
    """Save the period time series to Excel."""
    filename = f"timeseries_{label}.xlsx"
    df.to_excel(filename, index=False)
    print(f"Time series saved: {filename}")

# ==================================
# MAIN
# ==================================
def build_parameter_table() -> pd.DataFrame:
    """Collects all config fields into a table (config, parameter, value)."""
    configs = {
        "ely":        ely,
        "bess":       bess,
        "ppa":        ppa,
        "redispatch": redispatch,
        "rfnbo":      rfnbo,
        "market":     market,
        "economics":  economics,
    }
    rows = []
    for config_name, config_obj in configs.items():
        for field, value in asdict(config_obj).items():
            rows.append({
                "Config":    config_name,
                "Parameter": field,
                "Wert":      value,
            })
    # Zusätzlich die Lauf-Einstellungen aus main.py
    for name, value in {"YEAR": YEAR, "MODE": MODE, "MONTHS": str(MONTHS)}.items():
        rows.append({"Config": "run", "Parameter": name, "Wert": value})
    return pd.DataFrame(rows)

def main():
    df_full = load_data(year=YEAR, market=redispatch)
    total_hours = len(df_full)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    results = {}
    lcoh_result = None  # populated in "annual" mode, merged into results_summary below

    if MODE == "annual":
        print("\n" + "=" * 60)
        print(f"SOLVING FULL YEAR {YEAR}")
        print("=" * 60)

        time_share = 1.0  # this branch always solves the full year at once
        model, profit, lcoh, solve_time, converged, solver_status = solve_period(
            df_full, ely, bess, ppa, rfnbo, market, economics, redispatch, time_share=time_share
        )
        summary = extract_results(model, df_full, ely, rfnbo)
        summary["Profit"]         = profit
        summary["LCOH"]           = lcoh
        summary["converged"]      = converged
        summary["solver_status"]  = solver_status
        export_timeseries(df_full, label=f"{YEAR}_full_{timestamp}")
        results["Year"] = summary

        # --- 20-year cashflow / economics report, built from the Year summary ---
        # Electricity cost and DA market revenue are passed separately (not
        # netted): total_elec_cost already has revenue_da subtracted, so
        # adding it back gives the gross cost excl. market revenue.
        # c_el_annual deliberately uses the RAW (unadjusted) revenue_da - it's
        # the actual electricity procurement cost realised by the dispatch,
        # independent of the ex-post curtailment-sale/min-load adjustments
        # below (see extract_results).
        p_el_annual       = summary["P_ely_total"] * 1000                            # MWh -> kWh
        c_el_annual       = (summary["total_elec_cost"] + summary["revenue_da"]) * 1000  # k€ -> €
        # revenue_da_annual uses the ADJUSTED total (incl. the optional ex-post
        # curtailment sale, see economics.sell_curtailment_ex_post) - this is
        # the DA credit line that feeds NPV/IRR and the LCOH DA credit.
        revenue_da_annual = summary["revenue_da_adjusted"] * 1000                    # k€ -> €
        # h2_annual_t uses the ADJUSTED quantity (H2 produced while running
        # below minimum load doesn't count as produced - see extract_results).
        # This is what derive_capex_opex divides p_el_annual by to get
        # efficiency_year1, so the shortfall also worsens the realised
        # efficiency and correctly propagates into every projected year's
        # production/revenue - no separate price override needed.
        h2_annual_t       = summary["H2_amount_adjusted"]
        p_nom_bess        = summary["P_nom_bess"]
        # ely.p_nom is only the config's fixed/reference value - when
        # ely.p_nom_variable is True the solver picks its own size (see
        # extract_results' P_nom_ely). All downstream reporting (cashflow,
        # LCOH) must use that resolved size, so build a copy of ely with
        # p_nom overwritten accordingly and use it (not the original ely)
        # for every call below that reads ely.p_nom.
        p_nom_ely   = summary["P_nom_ely"]
        ely_resolved = replace(ely, p_nom=p_nom_ely)

        cf_result = compute_cashflow(
            p_el_annual, c_el_annual, revenue_da_annual, h2_annual_t,
            ely_resolved, bess, p_nom_bess, economics, redispatch,
        )
        export_cashflow_excel(
            cf_result, p_el_annual, c_el_annual, revenue_da_annual, ely_resolved, bess, p_nom_bess, economics, redispatch,
            out_path=f"cashflow_{YEAR}_{timestamp}.xlsx",
        )

        # --- Project LCOH (Lcoh.py) - same reference-year totals as the
        # cashflow report above, defensively projected to a full year via
        # time_share (a no-op here since time_share == 1.0, but this branch
        # would still be correct if that ever changed). ---
        p_el_annual_year       = p_el_annual / time_share
        c_el_annual_year       = c_el_annual / time_share
        revenue_da_annual_year = revenue_da_annual / time_share
        h2_annual_t_year       = h2_annual_t / time_share
        # cost_rd is already negative (a rebate) - see extract_results. It's
        # already netted into c_el_annual above; kept separately here only
        # so it can be broken back out as its own line in the LCOH cost
        # breakdown below, without changing what's passed to compute_lcoh.
        rd_credit_annual_eur  = summary["cost_rd"] * 1000 / time_share

        if p_nom_bess > 0:
            bess_discharge = sum(
                (pyo.value(model.bess_to_ely[t]) + pyo.value(model.bess_to_da[t])) / bess.eta_discharge
                for t in model.T
            )
            bess_capacity = p_nom_bess / bess.c_rate
            bess_cycles_per_year = bess_discharge / bess_capacity / time_share
        else:
            bess_cycles_per_year = None

        lcoh_result = compute_lcoh(
            p_el_annual_year, c_el_annual_year, revenue_da_annual_year, h2_annual_t_year,
            ely_resolved, bess, economics, redispatch, p_nom_bess,
            bess_cycles_per_year=bess_cycles_per_year,
        )

        # --- Sanity checks ---
        # LCOH is mathematically undefined when no H2 is produced at all
        # (e.g. P_nom_ely solved to 0 in a free-sizing run, or a BESS-only
        # scenario) - lcoh_result.lcoh is then NaN by design (see
        # compute_lcoh in Lcoh.py) and the Jahreskosten cross-check has
        # nothing to verify, so it's skipped rather than asserted.
        h2_produced = h2_annual_t_year > 1e-6
        if h2_produced:
            lcoh_jahreskosten = jahreskosten(lcoh_result)
            lcoh_check = lcoh_jahreskosten.loc[
                lcoh_jahreskosten["Position"] == "Summe / LCOH", "Anteil LCOH [EUR/kg]"
            ].iloc[0]
            assert abs(lcoh_check - lcoh_result.lcoh) < 1e-9, (
                f"Jahreskosten-Gegenprobe weicht von result.lcoh ab: "
                f"{lcoh_check} vs. {lcoh_result.lcoh} - eine Kostenposition ist falsch verdrahtet."
            )
            print(f"  LCOH-Gegenprobe (Jahreskosten) OK: {lcoh_check:.6f} == {lcoh_result.lcoh:.6f} EUR/kg")
            print(f"  LCOH (Lcoh.py, mit DA-Gutschrift):  {lcoh_result.lcoh:.4f} EUR/kg")
            print(f"  LCOH (Lcoh.py, ohne DA-Gutschrift): {lcoh_result.lcoh_ohne_gutschrift:.4f} EUR/kg")
        else:
            print("  Kein H2 produziert (P_nom_ely = 0 oder Volllast unter Mindestlast) - "
                  "LCOH nicht definiert, Gegenprobe übersprungen. NPV/IRR bleiben gültig "
                  "und sind die richtige Kennzahl für den Vergleich mit einem Ely-Case (siehe unten).")
        assert economics.i_debt <= lcoh_result.wacc <= economics.i_equity, (
            f"WACC {lcoh_result.wacc:.4f} liegt nicht zwischen i_debt ({economics.i_debt}) "
            f"und i_equity ({economics.i_equity})."
        )
        print(f"  BESS-Lebensdauer (Lcoh.py):         {lcoh_result.bess.lifetime} Jahre"
              + (f" ({bess_cycles_per_year:.1f} Vollzyklen/a)" if bess_cycles_per_year else " (kein BESS)"))

        # --- Scale-independent profitability metrics (NPV/IRR), reported
        # regardless of whether the Ely is used at all - unlike LCOH (cost
        # per kg, undefined without H2 output) or absolute profit (not
        # comparable across very different project sizes/mixes), NPV and
        # IRR stay well-defined for a BESS-only solution (P_nom_ely = 0)
        # and let you directly compare that against an Ely-included run on
        # equal footing. summary is the same dict object already stored in
        # results["Year"] above, so mutating it here still reaches the
        # exported results_summary Excel.
        summary["NPV_equity_EUR"] = cf_result.npv_equity
        summary["IRR_equity"]     = cf_result.irr_equity
        summary["IRR_project"]    = cf_result.irr_project
        summary["LCOH_final"]                = lcoh_result.lcoh
        summary["LCOH_final_ohne_Gutschrift"] = lcoh_result.lcoh_ohne_gutschrift
        print(f"  NPV (Equity): {cf_result.npv_equity:,.1f} EUR   |  "
              f"IRR (Equity): {cf_result.irr_equity:.1%}  |  IRR (Projekt): {cf_result.irr_project:.1%}")

    elif MODE == "monthly":
        monthly_dfs = []
        for month in MONTHS:
            print("\n" + "=" * 60)
            print(f"SOLVING MONTH {month}")
            print("=" * 60)

            df_month = slice_month(df_full, month)
            time_share = len(df_month) / total_hours

            model, profit, lcoh, solve_time, converged, solver_status = solve_period(
                df_month, ely, bess, ppa, rfnbo, market, economics, redispatch, time_share=time_share
            )
            label = f"Month_{month:02d}"
            summary = extract_results(model, df_month, ely, rfnbo)
            summary["Profit"]         = profit
            summary["LCOH"]           = lcoh
            summary["converged"]      = converged
            summary["solver_status"]  = solver_status
            monthly_dfs.append(df_month)
            results[label] = summary

        # One combined time series file across all solved months, instead of
        # one file per month (same convention as the "annual" mode's single
        # full-year export above).
        df_all_months = pd.concat(monthly_dfs, ignore_index=True)
        export_timeseries(df_all_months, label=f"{YEAR}_monthly_{timestamp}")

    else:
        raise ValueError(f"Unknown optimization mode: {MODE}")

    # --- Save summary table ---
    df_results = pd.DataFrame(results)
    filename = f"results_summary_{timestamp}.xlsx"

    with pd.ExcelWriter(filename) as writer:
        df_results.to_excel(writer, sheet_name="Results")
        build_parameter_table().to_excel(writer, sheet_name="Parameters", index=False)

        # --- LCOH (Lcoh.py) sheets, merged into the same workbook instead of
        # a standalone file (see export_lcoh_excel in Lcoh.py for the
        # original four-sheet layout this mirrors). Prefixed with "LCOH_" to
        # avoid colliding with "Results"/"Parameters" above. ---
        if lcoh_result is not None:
            lcoh_co = lcoh_result.capex
            lcoh_jk = jahreskosten(lcoh_result)
            lcoh_ann_check = lcoh_jk.loc[
                lcoh_jk["Position"] == "Summe / LCOH", "Anteil LCOH [EUR/kg]"
            ].iloc[0]

            # Grid connection / transformer CAPEX is sized off (p_nom_ely +
            # p_nom_bess) - see cashflow._derive_capex_opex / Lcoh.derive_capex_opex.
            # Split back into its two rate components for the "extra block" below.
            capex_netzanschluss = redispatch.gridconnection_capex * (p_nom_ely + p_nom_bess) * 1000
            capex_trafo         = economics.trafo_capex * (p_nom_ely + p_nom_bess) * 1000

            lcoh_inputs = pd.DataFrame([
                ("CAPEX Elektrolyseur", lcoh_co.capex_ely, "EUR"),
                ("CAPEX BESS", lcoh_co.capex_bess, "EUR"),
                ("CAPEX Netzanschluss", capex_netzanschluss, "EUR"),
                ("CAPEX Trafo", capex_trafo, "EUR"),
                ("CAPEX gesamt", lcoh_co.capex_total, "EUR"),
                ("Stack-Ersatzkosten (je Wechsel)", lcoh_co.capex_stack, "EUR"),
                ("BESS-Ersatzkosten (je Wechsel)", lcoh_co.capex_bess_new, "EUR"),
                ("OPEX Elektrolyseur", lcoh_co.opex_ely, "EUR/a"),
                ("OPEX BESS", lcoh_co.opex_bess, "EUR/a"),
                ("Effizienz Jahr 1", lcoh_co.efficiency_year1, "kWh/kg"),
                ("Degradationsrate", economics.degradation_rate, "-"),
                ("EK-Quote", economics.share_equity, "-"),
                ("EK-Zinssatz (real)", economics.i_equity, "-"),
                ("FK-Zinssatz (real)", economics.i_debt, "-"),
                ("WACC (real, vor Steuern)", lcoh_result.wacc, "-"),
                ("H2-Preis", economics.price_h2, "EUR/kg"),
                ("Stack-Lebensdauer", lcoh_result.stack.lifetime, "Jahre"),
                ("BESS-Lebensdauer", lcoh_result.bess.lifetime, "Jahre"),
                ("Projektlaufzeit", int(lcoh_result.yearly["Jahr"].max()), "Jahre"),
            ], columns=["Parameter", "Wert", "Einheit"])

            lcoh_ergebnis = pd.DataFrame([
                ("LCOH (Projekt, mit DA-Gutschrift)", lcoh_result.lcoh, "EUR/kg"),
                ("LCOH (Projekt, ohne DA-Gutschrift)", lcoh_result.lcoh_ohne_gutschrift, "EUR/kg"),
                ("LCOH Gegenprobe Annuitaetendarstellung", lcoh_ann_check, "EUR/kg"),
                ("WACC (real, vor Steuern)", lcoh_result.wacc, "-"),
            ], columns=["Kennzahl", "Wert", "Einheit"])

            # --- Ely/BESS breakdown of the two lumped jahreskosten() rows
            # ("Kapitalkosten Erstinvestition" and "OPEX") - built from the
            # same public building blocks jahreskosten() itself uses
            # (crf, Diskontfaktor, capex.*), without touching Lcoh.py.
            n_years = int(lcoh_result.yearly["Jahr"].max())
            annuity_factor = crf(lcoh_result.wacc, n_years)
            diskont = lcoh_result.yearly["Diskontfaktor"]
            menge_a = (lcoh_result.yearly["Produktionsmenge"] * diskont).sum() * annuity_factor

            # Initial CAPEX is a single t=0 outlay (Diskontfaktor==1 there),
            # so annuitizing each component separately is exact.
            ann_capex_ely           = lcoh_co.capex_ely  * annuity_factor
            ann_capex_bess          = lcoh_co.capex_bess * annuity_factor
            ann_capex_netzanschluss = capex_netzanschluss * annuity_factor
            ann_capex_trafo         = capex_trafo * annuity_factor
            ann_capex_grid          = ann_capex_netzanschluss + ann_capex_trafo

            # OPEX escalates uniformly (same opex_real_escalation for the
            # combined opex_fix), so the Ely/BESS split carries through
            # unchanged to every year - a fixed proportional split of the
            # annuitized total is exact.
            opex_fix_year1 = lcoh_co.opex_ely + lcoh_co.opex_bess
            share_ely  = lcoh_co.opex_ely  / opex_fix_year1 if opex_fix_year1 > 0 else 0.0
            share_bess = lcoh_co.opex_bess / opex_fix_year1 if opex_fix_year1 > 0 else 0.0
            ann_opex_total = (lcoh_result.yearly["OPEX"] * diskont).sum() * annuity_factor
            ann_opex_ely  = ann_opex_total * share_ely
            ann_opex_bess = ann_opex_total * share_bess

            # Sanity check: components must sum back to jahreskosten()'s
            # lumped rows exactly - if not, the split logic above is wrong.
            jk_capex = lcoh_jk.loc[lcoh_jk["Position"] == "Kapitalkosten Erstinvestition", "Jahreskosten [EUR/a]"].iloc[0]
            jk_opex  = lcoh_jk.loc[lcoh_jk["Position"] == "OPEX", "Jahreskosten [EUR/a]"].iloc[0]
            assert abs((ann_capex_ely + ann_capex_bess + ann_capex_grid) - jk_capex) < 1e-6, \
                "CAPEX-Aufschlüsselung Ely/BESS/Netz summiert sich nicht auf jahreskosten()'s CAPEX-Zeile."
            assert abs((ann_opex_ely + ann_opex_bess) - jk_opex) < 1e-6, \
                "OPEX-Aufschlüsselung Ely/BESS summiert sich nicht auf jahreskosten()'s OPEX-Zeile."

            # RD (Redispatch) savings: c_el_annual_year already nets the RD
            # rebate into the electricity cost (see extract_results), so
            # jahreskosten()'s "Stromkosten" row is already net-of-RD. Split
            # it back into a gross electricity cost and the RD credit,
            # proportionally to how c_el_annual_year relates to the
            # annuitized "Stromkosten" row - exact by construction, no new
            # assumption beyond what's already in that row.
            jk_stromkosten = lcoh_jk.loc[lcoh_jk["Position"] == "Stromkosten", "Jahreskosten [EUR/a]"].iloc[0]
            stromkosten_scale = jk_stromkosten / c_el_annual_year if c_el_annual_year != 0 else 0.0
            ann_rd_ersparnis     = rd_credit_annual_eur * stromkosten_scale
            ann_stromkosten_brutto = jk_stromkosten - ann_rd_ersparnis

            # menge_a is 0 when no H2 is produced (P_nom_ely = 0, e.g. a
            # BESS-only solution) - divide defensively to NaN instead of inf
            # (openpyxl refuses to write +/-inf to a cell and would crash
            # this export).
            def _share(val, _menge_a=menge_a):
                return val / _menge_a if _menge_a > 0 else float("nan")

            lcoh_component_breakdown = pd.DataFrame([
                {"Position": "CAPEX Elektrolyseur", "Jahreskosten [EUR/a]": ann_capex_ely,
                 "Anteil LCOH [EUR/kg]": _share(ann_capex_ely)},
                {"Position": "CAPEX BESS", "Jahreskosten [EUR/a]": ann_capex_bess,
                 "Anteil LCOH [EUR/kg]": _share(ann_capex_bess)},
                # --- Netzanschluss/Trafo: eigener Block, da an (Ely + BESS)-
                # Kapazität gekoppelt und damit weder rein Ely noch rein BESS. ---
                {"Position": "CAPEX Netzanschluss", "Jahreskosten [EUR/a]": ann_capex_netzanschluss,
                 "Anteil LCOH [EUR/kg]": _share(ann_capex_netzanschluss)},
                {"Position": "CAPEX Trafo", "Jahreskosten [EUR/a]": ann_capex_trafo,
                 "Anteil LCOH [EUR/kg]": _share(ann_capex_trafo)},
                {"Position": "CAPEX Netzanschluss + Trafo (gesamt)", "Jahreskosten [EUR/a]": ann_capex_grid,
                 "Anteil LCOH [EUR/kg]": _share(ann_capex_grid)},
                {"Position": "OPEX Elektrolyseur", "Jahreskosten [EUR/a]": ann_opex_ely,
                 "Anteil LCOH [EUR/kg]": _share(ann_opex_ely)},
                {"Position": "OPEX BESS", "Jahreskosten [EUR/a]": ann_opex_bess,
                 "Anteil LCOH [EUR/kg]": _share(ann_opex_bess)},
                # --- Stromkosten, aufgeteilt in brutto + RD-Gutschrift.
                # Beide Zeilen summieren sich exakt auf jahreskosten()'s
                # "Stromkosten"-Zeile (dort bereits netto RD). ---
                {"Position": "Stromkosten (brutto, ohne RD-Gutschrift)", "Jahreskosten [EUR/a]": ann_stromkosten_brutto,
                 "Anteil LCOH [EUR/kg]": _share(ann_stromkosten_brutto)},
                {"Position": "RD-Ersparnis (Gutschrift)", "Jahreskosten [EUR/a]": ann_rd_ersparnis,
                 "Anteil LCOH [EUR/kg]": _share(ann_rd_ersparnis)},
            ])

            lcoh_result.yearly.rename(columns=LCOH_COLUMN_LABELS).to_excel(
                writer, sheet_name="LCOH_Jahresuebersicht", index=False)
            lcoh_jk.to_excel(writer, sheet_name="LCOH_Jahreskosten", index=False)
            lcoh_component_breakdown.to_excel(writer, sheet_name="LCOH_Kosten_Ely_BESS", index=False)
            lcoh_inputs.to_excel(writer, sheet_name="LCOH_Inputs", index=False)
            lcoh_ergebnis.to_excel(writer, sheet_name="LCOH_Ergebnis", index=False)

    print(f"\nDone. Summary saved to {filename}")
    return df_results


if __name__ == "__main__":
    df_results = main()
