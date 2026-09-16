""""
Function: Runs the optimisation for a full calendar year in a single solve
(Kapitel X.1.3/X.2.1: das Modell optimiert immer den gesamten
Betrachtungszeitraum in einem Zug).

Standardmaessig (RUN_BESS_SWEEP = True) IST der BESS-Groessen-Sweep der
Hauptlauf: main() rechnet alle Groessen aus BESS_SWEEP_STEPS_MW x
BESS_SWEEP_C_RATES und schreibt sie als Blatt "BESS_Sweep" in die
Ergebnisdatei. Setzt man RUN_BESS_SWEEP auf False, wird
stattdessen NUR eine einzelne, feste BESS-Konfiguration (BESS_SIZE_FIXED_MW/
BESS_C_RATE_FIXED) gerechnet - nuetzlich, um gezielt eine Konfiguration zu
untersuchen, ohne den (unter Umstaenden lange laufenden) Sweep abzuwarten.

Created on Fri Jun 12

@author: Wiebke G
"""

import os
import pandas as pd
import pyomo.environ as pyo
from datetime import datetime
from dataclasses import asdict, replace

from Configuration import (
    ElyConfig, BessConfig, PPAConfig,
    Redispatch13kConfig, RfnboConfig, MarketConfig, EconomicsConfig, ScenarioConfig
)
from data  import load_data, load_gridconnection_capex
import model as model_mod
from model import solve_period
from markets import compute_rfnbo_threshold
from Lcoh import compute_lcoh


# ==================================
# CONFIGURATION
# ==================================

YEAR = 2025

# --- Betriebsmodus --------------------------------------------------------
# True  (Standard): main() rechnet den BESS-Sweep (siehe unten) als
#        Hauptlauf - ueber BESS_SWEEP_STEPS_MW x BESS_SWEEP_C_RATES.
# False: main() rechnet stattdessen NUR die feste Einzelkonfiguration
#        BESS_SIZE_FIXED_MW/BESS_C_RATE_FIXED - kein Sweep.
RUN_BESS_SWEEP      = True

# --- BESS: feste Groesse fuer die Einzelkonfiguration (RUN_BESS_SWEEP=False) ---
BESS_SIZE_FIXED_MW = 0.0
BESS_C_RATE_FIXED  = 0.5

# --- Solver-Tuning fuer schwere Faelle (v.a. BESS = 0 MW) -----------------
# Ohne Speicherpuffer haengt z_ely[t] direkt an der stuendlichen PPA-/DA-Lage:
# viele Stunden sind fast gleichwertig "an" oder "aus", was den Beweis der
# Optimalitaet sehr teuer macht (die Incumbent-Loesung steht meist frueh
# fest, der grosse Zeitaufwand steckt im Schliessen der oberen Schranke).
# Alles hier sind reine Gurobi-Suchparameter - sie aendern NICHT das
# Ergebnis, nur wie schnell der volle Optimalitaetsbeweis (bis MIP_REL_GAP
# in model.py) erreicht wird. KEIN TimeLimit, damit der Lauf garantiert bis
# zur (gap-)optimalen Loesung durchlaeuft. Auf {} setzen, um zum
# Solver-Standardverhalten zurueckzukehren.
#model_mod.EXTRA_SOLVER_OPTIONS = {
 #   "MIPFocus": 3,   # Suchreihenfolge auf gute Loesungen ausgerichtet
#}

# --- BESS-Sweep-Einstellungen (nur bei RUN_BESS_SWEEP=True) --------------
# 1-10 MW in 1-MW-Schritten.
BESS_SWEEP_STEPS_MW = list(range(1, 11, 1))
BESS_SWEEP_C_RATES  = [0.5, 0.25]
# Warmstart: Jede Sweep-Stufe startet mit der Loesung der vorherigen Stufe.
# Benachbarte BESS-Groessen unterscheiden sich nur in rund 1 % der
# Binaervariablen, der Solver muss den Suchbaum also nicht neu aufbauen.
# Nur Gurobi wertet die Startloesung aus, HiGHS ignoriert sie. Zum Messen
# des Effekts einfach auf False setzen und den Sweep erneut laufen lassen.
USE_WARMSTART       = False

ely        = ElyConfig()
bess       = replace(BessConfig(), c_rate=BESS_C_RATE_FIXED, p_nom=BESS_SIZE_FIXED_MW)
ppa        = PPAConfig()
redispatch = Redispatch13kConfig()
redispatch.gridconnection_capex = load_gridconnection_capex(redispatch.rd_region)  # [k€/MW], looked up by rd_region
rfnbo      = RfnboConfig()
market     = MarketConfig()
economics  = EconomicsConfig()
scenario   = ScenarioConfig()


# ==================================
# RESULT EXTRACTION
# ==================================

def extract_results(model, df, ely, rfnbo, index_offset=0) -> dict:
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
    df["z_ely_opt"]      = [pyo.value(model.z_ely[t])       for t in T]
    df["P_ely_opt"]      = (df["ppa_to_ely_opt"] + df["da_to_ely_opt"]
                            + df["bess_to_ely_opt"])
    # price_h2 [€/kg] * H2 [t]: kg->t (x1000) and €->k€ (/1000) cancel, so
    # this is already in k€ like every other cost/revenue figure.
    df["H2_revenue_opt"] = economics.price_h2 * df["H2_opt"]

    p_nom_wind = max(0.0, pyo.value(model.P_nom_wind))
    p_nom_pv   = max(0.0, pyo.value(model.P_nom_pv))
    p_nom_bess = max(0.0, pyo.value(model.P_nom_bess))
    p_nom_ely  = max(0.0, pyo.value(model.P_nom_ely))

    p_ely_total = df["P_ely_opt"].sum()
    h2_total    = df["H2_opt"].sum()
    flh_ely     = p_ely_total / p_nom_ely if p_nom_ely > 1e-9 else 0.0

    cost_ppa = (df["cf_pv"] * p_nom_pv * ppa.price_pv
                + df["cf_wind"] * p_nom_wind * ppa.price_wind).sum()
    cost_da  = ((df["da_to_ely_opt"] + df["da_to_bess_opt"]) * df["da_price"]).sum()
    revenue_da = ((df["ppa_to_da_opt"] + df["bess_to_da_opt"]) * df["da_price"]).sum()
    cost_rd  = -(df["rd_to_ely_opt"] * df["rd_reimbursement"]).sum()
    total_electricity_cost = cost_da + cost_ppa - revenue_da + cost_rd

    # --- EX-POST-VERMARKTUNG DER ABGEREGELTEN PPA-ENERGIE ------------------

    if economics.sell_curtailment_ex_post:
        sellable_hours = df["da_price"] > 0
        curtailment_da_revenue = (df["ppa_curtailed_opt"] * df["da_price"] * sellable_hours).sum()
        curtailment_da_volume  = (df["ppa_curtailed_opt"] * sellable_hours).sum()
    else:
        curtailment_da_revenue = 0.0
        curtailment_da_volume  = 0.0
    revenue_da_adjusted = revenue_da + curtailment_da_revenue

    revenue_h2    = df["H2_revenue_opt"].sum()
    revenue_total = revenue_h2 + revenue_da_adjusted
    da_revenue_share_pct = revenue_da_adjusted / revenue_total * 100 if revenue_total > 0 else 0.0

    # POST-CHECK: SIMULTANEOUS CHARGE/DISCHARGE

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

    # POST-CHECK: ELY START-STOPP-ZYKLEN
    # Ein Start = Uebergang von z_ely=0 auf z_ely=1 (Gleichung 5.16, siehe
    # ely.py con_start). Vor Periodenbeginn gilt die Anlage als aus, ein
    # Betrieb in der ersten Stunde zaehlt also ebenfalls als Start.
    z_ely_series = df["z_ely_opt"].round().astype(int)
    z_ely_prev   = z_ely_series.shift(1, fill_value=0)
    n_ely_cycles = int(((z_ely_series == 1) & (z_ely_prev == 0)).sum())

    if ely.max_start_stop_cycles is None:
        print(f"  Post-check: Ely-Start-Stopp-Zyklen: {n_ely_cycles} (kein Limit gesetzt)")
    elif n_ely_cycles <= ely.max_start_stop_cycles + 1e-6:
        print(f"  Post-check OK: Ely-Start-Stopp-Zyklen: {n_ely_cycles} "
              f"(Limit {ely.max_start_stop_cycles:.0f})")
    else:
        print(f"  WARNING: Ely-Start-Stopp-Zyklen: {n_ely_cycles} "
              f"überschreiten Limit {ely.max_start_stop_cycles:.0f}")

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

    da_bought_total = (df["da_to_ely_opt"] + df["da_to_bess_opt"]).sum()
    p_ppa_total     = df["P_ppa_opt"].sum()
    avg_buy_price_incl_ppa = ((cost_ppa + cost_da) / (p_ppa_total + da_bought_total)
                              if (p_ppa_total + da_bought_total) > 1e-9 else 0.0)
    avg_buy_price_excl_ppa = cost_da / da_bought_total if da_bought_total > 1e-9 else 0.0

    # Volumengewichteter Verkaufspreis - inkl. der ex post bewerteten
    # Abregelungsmenge (Erloes UND Menge), damit es ein echter Mittelwert
    # bleibt und nicht nur der Zaehler waechst.
    da_sold_total = ((df["ppa_to_da_opt"] + df["bess_to_da_opt"]).sum()
                     + curtailment_da_volume)
    avg_sell_price_da = revenue_da_adjusted / da_sold_total if da_sold_total > 1e-9 else 0.0

    rd_to_ely_total = df["rd_to_ely_opt"].sum()
    avg_rd_reimbursement = (-cost_rd / rd_to_ely_total
                            if rd_to_ely_total > 1e-9 else 0.0)

    bess_discharge_total = df["bess_to_ely_opt"].sum() + df["bess_to_da_opt"].sum()
    bess_discharge_to_ely_share_pct = (df["bess_to_ely_opt"].sum() / bess_discharge_total * 100
                                       if bess_discharge_total > 1e-9 else 0.0)

    return {
        "FLH_ely":          flh_ely,
        "H2_amount":        h2_total,
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
        "P_ppa_total":                  p_ppa_total,
        "curtailment_MWh":              df["ppa_curtailed_opt"].sum(),
        "curtailment_share_ppa_pct":    df["ppa_curtailed_opt"].sum() / p_ppa_total * 100
                                        if p_ppa_total > 1e-9 else 0.0,
        "curtailment_da_volume_expost": curtailment_da_volume,   # ex post vermarktbarer Anteil (nur Stunden mit da_price > 0)
        "curtailment_da_revenue_expost": curtailment_da_revenue,  # k€
        "revenue_da_adjusted":          revenue_da_adjusted,      # k€, inkl. Ex-post-Erloes
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
        "avg_rd_reimbursement":    avg_rd_reimbursement,     # k€/MWh
        "bess_discharge_to_ely_share_pct": bess_discharge_to_ely_share_pct,
        "n_ely_cycles":              n_ely_cycles,
        "correlation_ok":            correlation_ok,
        "correlation_surplus_MWh":   correlation_surplus,
        "n_hourly_correlation_violations": n_hourly_correlation_violations,
    }


def export_timeseries(df, label, out_dir: str = ".") -> None:
    """Save the period time series to Excel, optionally into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"timeseries_{label}.xlsx")
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
        "scenario":   scenario,
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
    rows.append({"Config": "run", "Parameter": "YEAR", "Wert": YEAR})
    return pd.DataFrame(rows)


def _run_lcoh(model, summary, ely_cfg, bess_cfg, p_nom_bess) -> "compute_lcoh":
    """Bequemlichkeitswrapper: baut die LCOH-Rechnung aus einer bereits
    berechneten extract_results()-Summary."""
    p_nom_ely = summary["P_nom_ely"]
    ely_resolved = replace(ely_cfg, p_nom=p_nom_ely)
    return compute_lcoh(
        p_nom_ely=p_nom_ely, p_nom_bess=p_nom_bess,
        cost_ppa=summary["cost_ppa"], cost_da=summary["cost_da"],
        revenue_da=summary["revenue_da"], cost_rd=summary["cost_rd"],
        # Ex-post-Erloes aus der Vermarktung abgeregelter PPA-Energie - 0,
        # wenn economics.sell_curtailment_ex_post aus ist. Bewusst als eigenes
        # Argument (und eigene LCOH-Zeile), nicht in revenue_da eingerechnet:
        # so bleibt sichtbar, welcher Teil der DA-Erloese aus dem tatsaechlich
        # optimierten Dispatch stammt und welcher aus der Nachbewertung.
        revenue_da_curtailment=summary["curtailment_da_revenue_expost"],
        h2_annual_t=summary["H2_amount"],
        ely=ely_resolved, bess=bess_cfg, economics=economics, redispatch=redispatch,
    )


def main():
    """
    Einstiegspunkt. RUN_BESS_SWEEP entscheidet, was der Hauptlauf ist:
    True (Standard) -> BESS-Sweep (_run_sweep_main), False -> eine einzelne
    feste Konfiguration (_run_single_config).
    """
    df_full = load_data(year=YEAR, market=redispatch, cf_scenario=scenario.cf_scenario)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if RUN_BESS_SWEEP:
        return _run_sweep_main(df_full, timestamp)
    return _run_single_config(df_full, timestamp)


def _run_single_config(df_full: pd.DataFrame, timestamp: str) -> pd.DataFrame:
    """
    Rechnet GENAU EINE feste BESS-Konfiguration (BESS_SIZE_FIXED_MW/
    BESS_C_RATE_FIXED) - zum gezielten Untersuchen einer einzelnen
    Konfiguration ohne den (unter Umstaenden lange laufenden) Sweep.
    Aktiviert durch RUN_BESS_SWEEP = False.
    """
    results = {}

    print("\n" + "=" * 60)
    print(f"SOLVING FULL YEAR {YEAR}  (BESS fest: {bess.p_nom} MW, c-rate {bess.c_rate})")
    print("=" * 60)

    model, deckungsbeitrag, solve_time, converged, solver_status = solve_period(
        df_full, ely, bess, ppa, rfnbo, market, economics
    )
    summary = extract_results(model, df_full, ely, rfnbo)
    summary["Deckungsbeitrag"] = deckungsbeitrag
    summary["converged"]      = converged
    summary["solver_status"]  = solver_status
    summary["solve_time_s"]   = solve_time
    export_timeseries(df_full, label=f"{YEAR}_full_{timestamp}")
    results["Year"] = summary

    # --- Wasserstoffgestehungskosten (Lcoh.py) - einjaehrig, undiskontiert,
    # aus der optimalen Loesung des Referenzjahres. ---
    lcoh_result = _run_lcoh(model, summary, ely, bess, summary["P_nom_bess"])

    summary["LCOH_final"] = lcoh_result.lcoh

    print(f"  LCOH: {lcoh_result.lcoh:.4f} EUR/kg")

    # --- Save summary table ---
    df_results = pd.DataFrame(results)
    filename = f"results_summary_{timestamp}.xlsx"

    with pd.ExcelWriter(filename) as writer:
        df_results.to_excel(writer, sheet_name="Results")
        build_parameter_table().to_excel(writer, sheet_name="Parameters", index=False)
        lcoh_result.as_dataframe().to_excel(writer, sheet_name="LCOH", index=False)

    print(f"\nDone. Summary saved to {filename}")
    return df_results


def _run_sweep_main(df_full: pd.DataFrame, timestamp: str) -> pd.DataFrame:
    """
    BESS-Sweep als Hauptlauf (Standard, RUN_BESS_SWEEP = True): rechnet alle
    Groessen aus BESS_SWEEP_STEPS_MW x BESS_SWEEP_C_RATES und schreibt sie
    als Blatt "BESS_Sweep" in die Ergebnisdatei.
    """
    print("\n" + "=" * 60)
    print("BESS-SWEEP (Hauptlauf)")
    print("=" * 60)

    timeseries_dir = f"timeseries_sweep_{timestamp}"
    df_sweep = run_bess_sweep(df_full, timeseries_dir)
    # Zwischenstand zusaetzlich als CSV, damit die Sweep-Ergebnisse auch dann
    # vorliegen, wenn beim Schreiben der Excel-Datei etwas schiefgeht.
    # sep=";" statt "," - deutsches Excel splittet CSVs beim Doppelklick nur
    # bei Semikolon korrekt in Spalten (Komma landet sonst als Text-Blob in
    # einer einzigen Zelle je Zeile).
    df_sweep.to_csv(f"bess_sweep_{timestamp}.csv", index=False, sep=";")

    filename = f"results_summary_{timestamp}.xlsx"
    with pd.ExcelWriter(filename) as writer:
        df_sweep.to_excel(writer, sheet_name="BESS_Sweep", index=False)
        build_parameter_table().to_excel(writer, sheet_name="Parameters", index=False)

    print(f"\nDone. Sweep saved to {filename}")
    return df_sweep


# ==================================
# BESS-SWEEP (Kapitel 1.5)
# ==================================

def _solve_sweep_step(df_full, bess_mw: float, c_rate: float, warmstart_z: dict | None,
                       timeseries_dir: str | None = None) -> tuple[dict, dict | None]:
    """
    Loest eine einzelne Sweep-Stufe (eine BESS-Groesse x C-Rate) und baut die
    zugehoerige Ergebniszeile. Gibt (row, z_ely_dieser_stufe) zurueck -
    z_ely ist None, wenn der Solve fehlgeschlagen ist.

    Ist timeseries_dir gesetzt, wird pro Stufe eine Excel-Datei mit der
    stuendlichen Zeitreihe (Blatt "Zeitreihe"), der vollstaendigen Summary
    dieses Cases (Blatt "Summary", alle Kennzahlen aus extract_results plus
    Deckungsbeitrag/LCOH/Solver-Status) und der LCOH-Kostenaufschluesselung
    (Blatt "LCOH", siehe Lcoh.LcohResult.as_dataframe) dorthin exportiert.
    """
    print(f"\n  --- Sweep: BESS = {bess_mw} MW, c-rate = {c_rate} ---")
    bess_step = replace(BessConfig(), c_rate=c_rate, p_nom=float(bess_mw))
    ppa_step  = PPAConfig()   # PPA-Groesse bleibt je Lauf frei optimierbar

    row = {"BESS_MW": bess_mw, "c_rate": c_rate, "error": "",
           "warmstart": bool(USE_WARMSTART and warmstart_z is not None)}
    try:
        model_s, deckungsbeitrag_s, solve_time_s, converged_s, status_s = solve_period(
            df_full, ely, bess_step, ppa_step, rfnbo, market, economics,
            warmstart_z=warmstart_z if USE_WARMSTART else None,
        )
        z_dieser_stufe = {t: round(pyo.value(model_s.z_ely[t])) for t in model_s.T}
        df_s = df_full.copy()
        summary_s = extract_results(model_s, df_s, ely, rfnbo)
        lcoh_s = _run_lcoh(model_s, summary_s, ely, bess_step, summary_s["P_nom_bess"])

        if timeseries_dir is not None:
            c_rate_label = str(c_rate).replace(".", "p")
            case_label = f"bess{bess_mw}MW_c{c_rate_label}"
            full_summary = {
                **summary_s,
                "Deckungsbeitrag_keur": deckungsbeitrag_s,
                "LCOH_EUR_kg": lcoh_s.lcoh,
                "converged": converged_s,
                "solver_status": status_s,
                "solve_time_s": solve_time_s,
            }
            os.makedirs(timeseries_dir, exist_ok=True)
            filename = os.path.join(timeseries_dir, f"case_{case_label}.xlsx")
            with pd.ExcelWriter(filename) as writer:
                df_s.to_excel(writer, sheet_name="Zeitreihe", index=False)
                pd.DataFrame([full_summary]).to_excel(writer, sheet_name="Summary", index=False)
                lcoh_s.as_dataframe().to_excel(writer, sheet_name="LCOH", index=False)
            print(f"  Case gespeichert: {filename}")

        row.update({
            "P_nom_wind":  summary_s["P_nom_wind"],
            "P_nom_pv":    summary_s["P_nom_pv"],
            "P_nom_bess":  summary_s["P_nom_bess"],
            "P_nom_ely":   summary_s["P_nom_ely"],
            "H2_amount_t": summary_s["H2_amount"],
            "FLH_ely":     summary_s["FLH_ely"],
            "n_ely_cycles": summary_s["n_ely_cycles"],
            "Curtailment_Anteil_PPA_pct": summary_s["curtailment_share_ppa_pct"],
            "BESS_Entladung_Anteil_Ely_pct": summary_s["bess_discharge_to_ely_share_pct"],
            "Deckungsbeitrag_keur": deckungsbeitrag_s,
            "LCOH_EUR_kg": lcoh_s.lcoh,
            "correlation_ok": summary_s["correlation_ok"],
            "converged":   converged_s,
            "solver_status": status_s,
            "solve_time_s": solve_time_s,
        })
        return row, z_dieser_stufe
    except Exception as exc:
        row["error"] = str(exc)
        print(f"  FEHLER bei BESS={bess_mw} MW, c-rate={c_rate}: {exc}")
        return row, None


def run_bess_sweep(df_full: pd.DataFrame, timeseries_dir: str | None = None) -> pd.DataFrame:
    """
    Wiederholt die Optimierung fuer mehrere feste BESS-Groessen
    (BESS_SWEEP_STEPS_MW x BESS_SWEEP_C_RATES), bei sonst unveraenderter
    (fixer) Ely-Groesse und frei optimierbarer PPA-Groesse - genau wie
    Kapitel 1.5 es fuer die Szenarien beschreibt (BESS-Dimensionierung
    exogen vorgegeben, 1-MW-Schritte, mehrere C-Raten). Gibt eine
    Zusammenfassungstabelle zurueck (eine Zeile pro Groesse x C-Rate), die
    main() als eigenes Excel-Blatt "BESS_Sweep" in dieselbe Ergebnisdatei
    schreibt.

    Ist timeseries_dir gesetzt, wird zusaetzlich pro Stufe die stuendliche
    Zeitreihe dorthin exportiert (siehe _solve_sweep_step).
    """
    rows = []
    for c_rate in BESS_SWEEP_C_RATES:
        # Startloesung je C-Raten-Reihe zuruecksetzen: der Sprung von der
        # letzten Stufe der vorherigen Reihe (grosses BESS) auf die erste
        # Stufe der neuen waere ein schlechter Startpunkt.
        z_vorstufe = None

        for bess_mw in BESS_SWEEP_STEPS_MW:
            row, z_vorstufe = _solve_sweep_step(df_full, bess_mw, c_rate, z_vorstufe, timeseries_dir)
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_results = main()
