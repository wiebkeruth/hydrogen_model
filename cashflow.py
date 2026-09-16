""""
Function: 20-year cashflow / economics model built from a single reference-year
dispatch result (electrolyser electricity consumption, electricity cost, DA
market sales revenue - kept separate, not netted) plus financing and
stack-degradation assumptions. Computes NPV/IRR (levered, equity view) and
LCOH (unlevered, project view) and writes a formatted Excel report.
 
All monetary inputs are real (today's prices, no inflation) - use real
interest rates accordingly.
 
Refactored version: same public interface (compute_cashflow, CashflowResult,
export_cashflow_excel) as the original, but split into small, independently
checkable steps, plus two bug fixes:
 
  1. Stack replacement cycles now repeat correctly. A replacement is only
     funded (reserve contributions) if it actually falls before project end;
     e.g. with stack_lifetime=10 and project_lifetime=20, exactly ONE
     replacement (at year 10) is funded - years 11-20 correctly carry no
     reserve, since that stack lasts through project end. With
     project_lifetime=30, a SECOND cycle (years 11-20) would now also be
     funded, which the original code silently dropped.
  2. Efficiency degradation now resets when a fresh stack goes into service,
     instead of compounding continuously for the whole project lifetime
     regardless of stack replacements.
 
@author: Wiebke G
"""
 
from dataclasses import dataclass
 
import pandas as pd
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
 
from Configuration import EconomicsConfig, ElyConfig, BessConfig, Redispatch13kConfig
 
 
@dataclass
class CashflowResult:
    """Full year-by-year table plus the headline KPIs."""
    yearly:       pd.DataFrame
    npv_equity:   float          # [€]
    irr_equity:   float          # [-] (e.g. 0.12 = 12%), levered / equity view
    lcoh:         float          # [€/kg]
    irr_project:  float = None   # [-] unlevered / project view - sanity check for irr_equity
    lcoh_mit_finanzierung: float = None  # [€/kg] - see Projekt_Kosten_mit_Finanzierung below
 
 
# ==================================
# FINANCIAL HELPERS
# ==================================
 
def _sinking_fund_annuity(target_amount: float, rate: float, n_years: int) -> float:
    """Constant annual deposit that compounds to target_amount after n_years at rate."""
    if n_years <= 0:
        return 0.0
    if rate == 0:
        return target_amount / n_years
    return target_amount * rate / ((1 + rate) ** n_years - 1)
 
 
def _npv(rate: float, cashflows: list[float]) -> float:
    """NPV of cashflows[0..n]; cashflows[0] is at t=0 (undiscounted)."""
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))
 
 
def _npv_derivative(rate: float, cashflows: list[float]) -> float:
    return sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cashflows))
 
 
def _irr(cashflows: list[float], guess: float = 0.1,
         tol: float = 1e-8, max_iter: int = 200) -> float:
    """
    IRR via Newton-Raphson; falls back to bisection over [-0.99, 10.0] if
    Newton doesn't converge (e.g. flat derivative, bad starting point).
    """
    r = guess
    for _ in range(max_iter):
        # Bail out before Newton-Raphson has a chance to diverge into a rate
        # where (1+r)**t overflows a float (e.g. after a step through a
        # near-zero derivative) - the bisection fallback below is bounded
        # and numerically safe, so let it handle these cashflows instead.
        if r <= -1.0 or abs(r) > 1e6:
            break
        f  = _npv(r, cashflows)
        fp = _npv_derivative(r, cashflows)
        if abs(fp) < 1e-12:
            break
        r_new = r - f / fp
        if abs(r_new - r) < tol:
            return r_new
        r = r_new
 
    lo, hi = -0.99, 10.0
    f_lo, f_hi = _npv(lo, cashflows), _npv(hi, cashflows)
    if f_lo * f_hi > 0:
        # No sign change anywhere in [-99%, 1000%]: every cashflow has the
        # same sign (e.g. the project never recoups its investment) - IRR is
        # mathematically undefined here, not a numerical glitch. Report NaN
        # instead of crashing the whole cashflow report.
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = _npv(mid, cashflows)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2
 
 
# ==================================
# STEP 1: CAPEX / OPEX / BASE EFFICIENCY
# ==================================
 
@dataclass
class _CapexOpex:
    capex_total:             float  # [€] ely + BESS + grid connection + transformer
    capex_stack_replacement: float  # [€] cost of one stack swap
    opex_fix:                float  # [€/a]
    efficiency_year1:        float  # [kWh/kg] realised reference-year efficiency


def _derive_capex_opex(p_el_annual: float, h2_annual_t: float,
                        ely: ElyConfig, bess: BessConfig, p_nom_bess: float,
                        econ: EconomicsConfig, redispatch: Redispatch13kConfig) -> _CapexOpex:
    """Pull CAPEX/OPEX/efficiency straight from ElyConfig/BessConfig (single
    source of truth, same as the dispatch model) instead of duplicating them
    as separate cashflow-only inputs. Grid connection CAPEX has no natural
    home in Ely/BessConfig, so it's sized off the BESS connected capacity
    and taken from Redispatch13kConfig instead."""
    capex_grid = redispatch.gridconnection_capex * p_nom_bess * 1000  # k€ -> €
    capex_total = (ely.capex * ely.p_nom + bess.capex * (p_nom_bess / bess.c_rate)) * 1000 + capex_grid
    capex_stack_replacement = ely.stack_replacment_cost * ely.p_nom * 1000         # k€ -> €
    opex_fix = (ely.opex * ely.capex * ely.p_nom + bess.opex * p_nom_bess) * 1000  # k€/a -> €/a
    # MWh/t and kWh/kg are numerically identical (both factors of 1000 cancel).
    efficiency_year1 = (p_el_annual / 1000 / h2_annual_t) if h2_annual_t > 0 else ely.power_consumption_100
    return _CapexOpex(capex_total, capex_stack_replacement, opex_fix, efficiency_year1)
 
 
# ==================================
# STEP 2: DEBT FINANCING SCHEDULE
# ==================================
 
def _build_financing_schedule(capex_debt: float, i_debt: float,
                               grace_years: int, amort_years: int,
                               project_lifetime: int) -> pd.DataFrame:
    """
    Straight-line ("linear") amortization: equal principal repayments once
    the grace period ends, interest charged on the opening balance of each
    year. Returns one row per year (0..project_lifetime).
 
    Restschuld_FK = debt balance AT THE START of year t (i.e. balance that
    year t's interest is charged on). At t=0 this is the initial draw-down.
    """
    amort_per_year = capex_debt / amort_years if amort_years > 0 else 0.0
 
    rows = []
    balance = capex_debt
    for t in range(0, project_lifetime + 1):
        if t == 0:
            rows.append(dict(Jahr=0, Restschuld_FK=balance, Zinsen_FK=0.0, Tilgung_FK=0.0))
            continue
 
        interest = balance * i_debt
        in_amort_window = grace_years < t <= grace_years + amort_years
        amortization = amort_per_year if in_amort_window else 0.0
 
        rows.append(dict(Jahr=t, Restschuld_FK=balance, Zinsen_FK=interest, Tilgung_FK=amortization))
        balance = max(0.0, balance - amortization)
 
    return pd.DataFrame(rows)
 
 
# ==================================
# STEP 3: STACK REPLACEMENT CYCLES (reserve + efficiency reset)
# ==================================
 
def _stack_cycle_position(t: int, stack_lifetime: int) -> int:
    """
    Position of year t within the CURRENT stack's life, counting 0 at the
    year a fresh stack starts service. E.g. stack_lifetime=10:
    t=1 -> 0 (brand new stack, year 1 of its life)
    t=10 -> 9 (last year of that stack's life)
    t=11 -> 0 (replacement stack, fresh again)
    """
    return (t - 1) % stack_lifetime
 
 
def _replacement_needed_this_cycle(t: int, stack_lifetime: int, project_lifetime: int) -> bool:
    """
    Whether the stack currently in service (as of year t) will actually be
    replaced before the project ends - i.e. whether it's worth funding a
    reserve for it. If the stack in service happens to last exactly through
    project end (e.g. stack_lifetime=10, project_lifetime=20: the stack
    installed at year 10 runs to year 20), no replacement - and no reserve -
    is needed for that cycle.
    """
    cycle_end_year = ((t - 1) // stack_lifetime + 1) * stack_lifetime
    return cycle_end_year < project_lifetime
 
 
def _build_stack_schedule(efficiency_year1: float, degradation_rate: float,
                           reserve_annuity: float, stack_lifetime: int,
                           project_lifetime: int) -> pd.DataFrame:
    """
    Per-year efficiency (resets to efficiency_year1 whenever a fresh stack
    goes into service) and reserve contribution (only while a replacement is
    actually still ahead within the project lifetime - see
    _replacement_needed_this_cycle).
    """
    rows = [dict(Jahr=0, Effizienz=None, Stack_Ruecklage=0.0)]
    for t in range(1, project_lifetime + 1):
        cycle_pos = _stack_cycle_position(t, stack_lifetime)
        efficiency = efficiency_year1 * (1 + degradation_rate) ** cycle_pos
        reserve = reserve_annuity if _replacement_needed_this_cycle(t, stack_lifetime, project_lifetime) else 0.0
        rows.append(dict(Jahr=t, Effizienz=efficiency, Stack_Ruecklage=reserve))
    return pd.DataFrame(rows)
 
 
# ==================================
# STEP 4: PRODUCTION & REVENUE
# ==================================
 
def _build_production_schedule(p_el_annual: float, price_h2: float,
                                stack_df: pd.DataFrame) -> pd.DataFrame:
    """Production/revenue follow directly from the efficiency schedule.
    p_el_annual (electricity consumption) is held constant across all years
    - only efficiency changes, per the stack schedule above."""
    df = stack_df.copy()
    df["Produktionsmenge"] = df.apply(
        lambda r: p_el_annual / r["Effizienz"] if r["Jahr"] > 0 else 0.0, axis=1)
    df["Erloes"] = df["Produktionsmenge"] * price_h2
    return df[["Jahr", "Produktionsmenge", "Erloes"]]
 
 
# ==================================
# MAIN ENTRY POINT: COMBINE STEPS -> CASHFLOW -> KPIs
# ==================================
 
def compute_cashflow(
    p_el_annual: float,          # electrolyser electricity consumption, ref. year [kWh/a]
    c_el_annual: float,          # electricity cost, ref. year, EXCL. DA market revenue [€/a]
    revenue_da_annual: float,    # DA market sales revenue, ref. year (PPA surplus + BESS) [€/a]
    h2_annual_t: float,          # reference-year H2 production [t] (for realised efficiency)
    ely: ElyConfig,
    bess: BessConfig,
    p_nom_bess: float,           # solved BESS power rating [MW]
    econ: EconomicsConfig,
    redispatch: Redispatch13kConfig,
    grace_years: int = 4,
    amort_years: int = 15,
) -> CashflowResult:
    """
    Build the year-by-year (t=0..ely.lifetime) cashflow table and derive
    NPV/IRR (equity, levered), IRR (project, unlevered - sanity check) and
    LCOH (project, unlevered).
 
    Electricity cost and DA market revenue are kept as two separate line
    items (not netted) so the report shows what's actually being paid for
    electricity vs. what's earned selling back to the market.
    """
    project_lifetime = int(ely.lifetime)
    stack_lifetime = int(ely.stack_lifetime)
 
    # --- Step 1: capex/opex/base efficiency ---
    co = _derive_capex_opex(p_el_annual, h2_annual_t, ely, bess, p_nom_bess, econ, redispatch)
 
    share_debt   = 1 - econ.share_equity
    capex_debt   = co.capex_total * share_debt
    capex_equity = co.capex_total * econ.share_equity
    wacc = econ.share_equity * econ.i_equity + share_debt * econ.i_debt
 
    # --- Step 2: debt schedule ---
    debt_df = _build_financing_schedule(capex_debt, econ.i_debt, grace_years, amort_years, project_lifetime)
 
    # --- Step 3: stack reserve + efficiency (with cycle reset) ---
    reserve_annuity = _sinking_fund_annuity(co.capex_stack_replacement, econ.i_reserve, stack_lifetime)
    stack_df = _build_stack_schedule(co.efficiency_year1, econ.degradation_rate,
                                      reserve_annuity, stack_lifetime, project_lifetime)
 
    # --- Step 4: production & revenue ---
    prod_df = _build_production_schedule(p_el_annual, econ.price_h2, stack_df)
 
    # --- Combine into the final year-by-year table ---
    df = debt_df.merge(stack_df, on="Jahr").merge(prod_df, on="Jahr")
    df["Stromkosten"] = df["Jahr"].apply(lambda t: c_el_annual if t > 0 else 0.0)
    df["Erloese_Strommarkt"] = df["Jahr"].apply(lambda t: revenue_da_annual if t > 0 else 0.0)
    # OPEX escalates with inflation_rate (year 1 = base opex_fix, no inflation
    # yet - consistent with how the stack-efficiency degradation above is
    # anchored). Deliberate exception to the "all inputs real" convention
    # (see module docstring): electricity cost/H2 price/CAPEX stay flat.
    df["OPEX"] = df["Jahr"].apply(
        lambda t: co.opex_fix * (1 + econ.inflation_rate) ** (t - 1) if t > 0 else 0.0)
 
    df["Equity_Cashflow"] = df.apply(
        lambda r: (r["Erloes"] + r["Erloese_Strommarkt"] - r["Stromkosten"] - r["OPEX"]
                   - r["Stack_Ruecklage"] - r["Zinsen_FK"] - r["Tilgung_FK"])
        if r["Jahr"] > 0 else -capex_equity,
        axis=1,
    )
    df["Projekt_Cashflow"] = df.apply(  # unlevered: no interest / amortization
        lambda r: (r["Erloes"] + r["Erloese_Strommarkt"] - r["Stromkosten"] - r["OPEX"]
                   - r["Stack_Ruecklage"])
        if r["Jahr"] > 0 else -co.capex_total,
        axis=1,
    )
    df["Projekt_Kosten"] = df.apply(
        lambda r: r["OPEX"] + r["Stromkosten"] - r["Erloese_Strommarkt"] + r["Stack_Ruecklage"]
        if r["Jahr"] > 0 else co.capex_total,
        axis=1,
    )

    # --- Additional (undiscounted) LCOH variant: financing costs included as
    # explicit cost items on top of Projekt_Kosten above, instead of relying
    # on the WACC discount factor to represent the cost of capital. EK-
    # Verzinsung is charged flat on the original equity stake every year
    # (equity is not amortized like debt, unlike Zinsen_FK which already
    # declines with the debt schedule's Restschuld_FK).
    ek_verzinsung = econ.i_equity * capex_equity
    df["EK_Verzinsung"] = df["Jahr"].apply(lambda t: ek_verzinsung if t > 0 else 0.0)
    df["Projekt_Kosten_mit_Finanzierung"] = (
        df["Projekt_Kosten"] + df["Zinsen_FK"] + df["EK_Verzinsung"]
    )

    df["Equity_Cashflow_kum"] = df["Equity_Cashflow"].cumsum()
    df["Equity_CF_diskontiert"] = df["Equity_Cashflow"] / (1 + econ.i_equity) ** df["Jahr"]
    df["Projekt_Kosten_diskontiert"] = df["Projekt_Kosten"] / (1 + wacc) ** df["Jahr"]
    df["Menge_diskontiert"] = df.apply(
        lambda r: r["Produktionsmenge"] / (1 + wacc) ** r["Jahr"] if r["Jahr"] > 0 else 0.0,
        axis=1,
    )

    npv_equity  = df["Equity_CF_diskontiert"].sum()
    irr_equity  = _irr(df["Equity_Cashflow"].tolist())
    irr_project = _irr(df["Projekt_Cashflow"].tolist())
    lcoh = df["Projekt_Kosten_diskontiert"].sum() / df["Menge_diskontiert"].sum()
    lcoh_mit_finanzierung = (df["Projekt_Kosten_mit_Finanzierung"].sum()
                             / df["Produktionsmenge"].sum())
 
    if irr_equity != irr_equity:  # NaN check without importing math for this one use
        print("  WARNING: Equity-IRR undefined - equity cashflows never turn net positive "
              "(check p_el_annual / c_el_annual / revenue_da_annual / price_h2 vs. capex_total).")
 
    # Sanity check: equity IRR should exceed project IRR (leverage effect)
    # roughly in proportion to how cheap debt is vs. the project's own
    # return, and by a plausible margin given the debt share. A huge gap
    # (e.g. project IRR 8% but equity IRR 30%+) is a signal to double-check
    # share_equity / i_debt / amort_years / grace_years rather than assume
    # the number is simply "good".
    if irr_project == irr_project and irr_equity == irr_equity:
        print(f"  Projekt-IRR (unlevered): {irr_project:.1%}  |  Equity-IRR (levered): {irr_equity:.1%}"
              f"  (EK-Quote: {econ.share_equity:.0%}, FK-Zins: {econ.i_debt:.1%})")
 
    # drop helper column not meant for the report
    df = df.drop(columns=["Projekt_Cashflow"])
 
    return CashflowResult(yearly=df, npv_equity=npv_equity, irr_equity=irr_equity,
                           lcoh=lcoh, irr_project=irr_project,
                           lcoh_mit_finanzierung=lcoh_mit_finanzierung)
 
 
# ==================================
# EXCEL EXPORT
# ==================================
 
_COLUMN_LABELS = {
    "Jahr":                        "Jahr",
    "Effizienz":                   "Effizienz [kWh/kg]",
    "Produktionsmenge":            "Produktionsmenge m_t [kg]",
    "Erloes":                      "Erlös H2 [€]",
    "Stromkosten":                 "Stromkosten [€]",
    "Erloese_Strommarkt":          "Erlöse Strommarkt [€]",
    "OPEX":                        "OPEX [€]",
    "Stack_Ruecklage":             "Stack-Rücklage [€]",
    "Restschuld_FK":               "Restschuld FK [€]",
    "Zinsen_FK":                   "Zinsen FK [€]",
    "Tilgung_FK":                  "Tilgung FK [€]",
    "Equity_Cashflow":             "Equity-Cashflow [€]",
    "Equity_Cashflow_kum":         "Kum. Equity-Cashflow [€]",
    "Projekt_Kosten":              "Projekt-Kosten unlevered [€]",
    "EK_Verzinsung":               "EK-Verzinsung [€]",
    "Projekt_Kosten_mit_Finanzierung": "Projekt-Kosten inkl. Finanzierung [€]",
    "Equity_CF_diskontiert":       "Equity-CF diskontiert [€]",
    "Projekt_Kosten_diskontiert":  "Projekt-Kosten diskontiert [€]",
    "Menge_diskontiert":           "Menge diskontiert [kg]",
}
 
_INPUT_LABELS = {
    "p_el_annual":              ("Stromverbrauch Elektrolyseur, Referenzjahr", "kWh/a"),
    "c_el_annual":              ("Stromkosten (exkl. Markterlöse), Referenzjahr", "€/a"),
    "revenue_da_annual":        ("Erlöse Strommarkt, Referenzjahr", "€/a"),
    "capex_total":              ("CAPEX gesamt", "€"),
    "share_equity":             ("EK-Quote", "-"),
    "i_equity":                 ("EK-Zinssatz / geforderte Rendite", "-"),
    "i_debt":                   ("FK-Zinssatz", "-"),
    "capex_stack_replacement":  ("Stack-Ersatzkosten (pro Wechsel)", "€"),
    "i_reserve":                ("Verzinsung Stack-Rücklage", "-"),
    "opex_fix":                 ("Fixe jährliche OPEX (Basisjahr)", "€/a"),
    "degradation_rate":         ("Jährliche Effizienz-Degradation", "-"),
    "inflation_rate":           ("Jährliche OPEX-Kostensteigerung", "-"),
    "price_h2":                 ("H2-Verkaufspreis", "€/kg"),
    "gridconnection_capex":     ("Netzanschlusskosten", "k€/MW"),
    "project_lifetime":         ("Projektlaufzeit", "Jahre"),
    "stack_lifetime":           ("Stack-Lebensdauer", "Jahre"),
}
 
 
def _bold_header(ws) -> None:
    for cell in ws[1]:
        cell.font = Font(bold=True)
 
 
def _autosize_columns(ws, min_width: int = 10, max_width: int = 38) -> None:
    for col_cells in ws.columns:
        length = max((len(str(c.value)) for c in col_cells if c.value is not None), default=0)
        letter = get_column_letter(col_cells[0].column)
        ws.column_dimensions[letter].width = max(min_width, min(max_width, length + 2))
 
 
def export_cashflow_excel(
    result: CashflowResult,
    p_el_annual: float,
    c_el_annual: float,
    revenue_da_annual: float,
    ely: ElyConfig,
    bess: BessConfig,
    p_nom_bess: float,
    econ: EconomicsConfig,
    redispatch: Redispatch13kConfig,
    out_path: str,
) -> None:
    """Write the yearly table, inputs, and KPI results to a standalone Excel file."""
    yearly = result.yearly.rename(columns=_COLUMN_LABELS)

    capex_grid = redispatch.gridconnection_capex * p_nom_bess * 1000
    capex_total = (ely.capex * ely.p_nom + bess.capex * (p_nom_bess / bess.c_rate)) * 1000 + capex_grid
    capex_stack_replacement = ely.stack_replacment_cost * ely.p_nom * 1000
    opex_fix = (ely.opex * ely.capex * ely.p_nom + bess.opex * p_nom_bess) * 1000

    input_values = dict(
        p_el_annual=p_el_annual, c_el_annual=c_el_annual, revenue_da_annual=revenue_da_annual,
        capex_total=capex_total, capex_stack_replacement=capex_stack_replacement,
        opex_fix=opex_fix,
        project_lifetime=int(ely.lifetime), stack_lifetime=int(ely.stack_lifetime),
        **vars(econ), **vars(redispatch),
    )
    inputs_df = pd.DataFrame([
        {"Parameter": label, "Wert": input_values[key], "Einheit": unit}
        for key, (label, unit) in _INPUT_LABELS.items()
    ])
 
    ergebnis_rows = [
        {"Kennzahl": "NPV (Equity)",              "Wert": result.npv_equity, "Einheit": "€"},
        {"Kennzahl": "IRR (Equity, levered)",      "Wert": result.irr_equity, "Einheit": "%"},
        {"Kennzahl": "LCOH",                       "Wert": result.lcoh,       "Einheit": "€/kg"},
    ]
    if result.irr_project is not None:
        ergebnis_rows.insert(2, {"Kennzahl": "IRR (Projekt, unlevered)",
                                  "Wert": result.irr_project, "Einheit": "%"})
    if result.lcoh_mit_finanzierung is not None:
        ergebnis_rows.append({"Kennzahl": "LCOH inkl. Finanzierungskosten (undiskontiert)",
                              "Wert": result.lcoh_mit_finanzierung, "Einheit": "€/kg"})
    ergebnis_df = pd.DataFrame(ergebnis_rows)
 
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        yearly.to_excel(writer, sheet_name="Jahresübersicht", index=False)
        inputs_df.to_excel(writer, sheet_name="Inputs", index=False)
        ergebnis_df.to_excel(writer, sheet_name="Ergebnis", index=False)
 
        wb = writer.book
 
        ws = writer.sheets["Jahresübersicht"]
        _bold_header(ws)
        eur_cols = {"Erlös H2 [€]", "Stromkosten [€]", "Erlöse Strommarkt [€]", "OPEX [€]",
                    "Stack-Rücklage [€]", "Restschuld FK [€]", "Zinsen FK [€]", "Tilgung FK [€]",
                    "Equity-Cashflow [€]", "Kum. Equity-Cashflow [€]", "Projekt-Kosten unlevered [€]",
                    "EK-Verzinsung [€]", "Projekt-Kosten inkl. Finanzierung [€]",
                    "Equity-CF diskontiert [€]", "Projekt-Kosten diskontiert [€]"}
        header = [c.value for c in ws[1]]
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                col_name = header[cell.column - 1]
                if col_name in eur_cols:
                    cell.number_format = "#,##0"
                elif col_name in ("Produktionsmenge m_t [kg]", "Menge diskontiert [kg]"):
                    cell.number_format = "#,##0"
                elif col_name == "Effizienz [kWh/kg]":
                    cell.number_format = "0.00"
        _autosize_columns(ws)
 
        ws = writer.sheets["Inputs"]
        _bold_header(ws)
        for row in ws.iter_rows(min_row=2, min_col=2, max_col=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "#,##0.0000" if abs(cell.value) < 1 else "#,##0"
        _autosize_columns(ws)
 
        ws = writer.sheets["Ergebnis"]
        _bold_header(ws)
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            unit_cell = row[2]
            value_cell = row[1]
            if unit_cell.value == "%":
                value_cell.number_format = "0.00%"
            elif unit_cell.value == "€/kg":
                value_cell.number_format = "0.00"
            else:
                value_cell.number_format = "#,##0"
        _autosize_columns(ws)
 
    print(f"Cashflow report saved: {out_path}")
 