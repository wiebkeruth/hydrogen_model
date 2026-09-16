
""""
Function: Data loading and preparation

Created on Fri Jun 12 

@author: Wiebke G 
"""

import pandas as pd
import calendar
from Configuration import Redispatch13kConfig

# --- File paths (multi-year workbooks, one sheet per calendar year) ---
PATH_CF_PV     = "PV_CF_2021-2025.xlsx"
PATH_CF_WIND   = "Wind_CF_2021-2025.xlsx"
PATH_DA_PRICE  = "DA_Prices_2021-2025.xlsx"

# --- Kapazitaetsfaktor-Szenarien (nur Jahr 2025, ein Blatt je Szenario) ---
# "Base Case" wird NICHT aus diesen Dateien gelesen, sondern wie bisher aus
# PATH_CF_PV/PATH_CF_WIND (Blatt = Jahr) - siehe _load_cf_series.
PATH_CF_PV_SCENARIOS   = "PV_CF_Szenarien.xlsx"
PATH_CF_WIND_SCENARIOS = "Wind_CF_Szenarien.xlsx"
CF_SCENARIOS = {"Base Case", "Hohe VLH", "Niedrige VLH"}
CF_SCENARIO_YEAR = 2025  # Jahr, fuer das die Szenariodateien Daten enthalten

# RD volume only exists as a single (non-leap) reference year. Leap years
# reuse it and splice in the extra Feb-29 day from a dedicated sheet.
PATH_RD_VOLUME       = "RD_Volume_2025.xlsx"
RD_BASE_SHEET        = "2025"  # 8760h, non-leap reference pattern
RD_LEAP_DAY_SHEET     = "2024"  # 24h Feb-29 supplement

# EUA (CO2 allowance) auction prices: only traded on auction days, sparse.
PATH_EUA_PRICE = "EUA_Price_2021-2025.xlsx"

# H1 und T6 werden als eine gemeinsame Entlastungszone behandelt: waehlt man
# eine der beiden, wird die verfuegbare Redispatch-Menge als Summe beider
# Zonen angesetzt (siehe _rd_available_series).
RD_COMBINED_REGIONS = {"H1", "T6"}

# Grid connection cost per RD region, one row per region, values in €/MW.
PATH_GRID_CONNECTION_COSTS = "Grid_Connection_Costs.xlsx"

# Noise floors: values below these are numerically but not economically
# meaningful and cause excessively small MILP coefficients (down to ~1e-5 in
# the objective/matrix, driving HiGHS scaling warnings and slow convergence).
CF_MIN_THRESHOLD    = 1e-3   # [-] capacity factor, ~0.1% of nameplate
PRICE_MIN_THRESHOLD = 1e-4   # [k€/MWh] = 0.1 €/MWh, applied to |price|


def _load_rd_volume(year: int) -> pd.DataFrame:
    """Load RD volume for the given year, splicing in Feb 29 for leap years."""
    df_rd = pd.read_excel(PATH_RD_VOLUME, sheet_name=RD_BASE_SHEET)
    if _is_leap_year(year):
        df_leap_day = pd.read_excel(PATH_RD_VOLUME, sheet_name=RD_LEAP_DAY_SHEET)
        insert_at = (31 + 28) * 24  # hours before March 1st (Jan + Feb)
        df_rd = pd.concat([
            df_rd.iloc[:insert_at],
            df_leap_day,
            df_rd.iloc[insert_at:],
        ], ignore_index=True)
    return df_rd


def _load_eua_price_series(year: int) -> pd.Series:
    """Raw auction-date -> price series for one calendar year, sorted by date."""
    df = pd.read_excel(PATH_EUA_PRICE, sheet_name=str(year))
    return df.set_index(df.columns[0])[df.columns[1]].sort_index()


def _last_eua_price_of_previous_year(year: int) -> float | None:
    """Last known EUA auction price of year-1, or None if that sheet doesn't exist."""
    try:
        prices_prev = _load_eua_price_series(year - 1)
    except (ValueError, FileNotFoundError):
        return None
    if prices_prev.empty:
        return None
    return float(prices_prev.iloc[-1])


def _load_eua_price(year: int) -> pd.Series:
    """
    Load EUA auction prices for the given year and expand to one price per
    calendar day: gaps between auction dates are forward-filled with the
    last known price. Days before the first auction of the year take the
    last known price of the previous year (Jahreswechsel-Fortschreibung),
    falling back to the first known price of the current year if no
    previous-year sheet exists (e.g. for the first year in the workbook).
    """
    prices = _load_eua_price_series(year)

    full_days = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
    daily = prices.reindex(full_days).ffill()

    if daily.isna().any():
        start_value = _last_eua_price_of_previous_year(year)
        if start_value is not None:
            daily = daily.fillna(start_value)
        else:
            daily = daily.bfill()

    daily.index.name = "date"
    daily.name = "eua_price"
    return daily


def _rd_available_series(df_rd: pd.DataFrame, rd_region: str) -> pd.Series:
    """
    Verfuegbare Redispatch-Menge fuer die gewaehlte Entlastungszone.

    H1 und T6 werden als eine gemeinsame Zone behandelt: bei rd_region="H1"
    oder rd_region="T6" wird die Summe beider Spalten zurueckgegeben, sonst
    die Spalte der gewaehlten Zone unveraendert.
    """
    if rd_region in RD_COMBINED_REGIONS:
        missing = RD_COMBINED_REGIONS - set(df_rd.columns)
        if missing:
            raise ValueError(
                f"RD_COMBINED_REGIONS {sorted(RD_COMBINED_REGIONS)} erfordert beide "
                f"Spalten in den RD-Volumendaten, es fehlen: {sorted(missing)}."
            )
        return df_rd[sorted(RD_COMBINED_REGIONS)].sum(axis=1)
    return df_rd[rd_region]


def _load_cf_series(path_multi_year: str, path_scenarios: str, column: str,
                     year: int, cf_scenario: str) -> pd.Series:
    """
    Load one capacity-factor column, either from the historical multi-year
    workbook (Base Case, one sheet per year) or from the dedicated scenario
    workbook (Hohe VLH / Niedrige VLH, one sheet per scenario, year 2025 only).
    """
    if cf_scenario not in CF_SCENARIOS:
        raise ValueError(
            f"Unknown cf_scenario: '{cf_scenario}'. "
            f"Available scenarios: {sorted(CF_SCENARIOS)}."
        )
    if cf_scenario == "Base Case":
        return pd.read_excel(path_multi_year, sheet_name=str(year))[column]
    if year != CF_SCENARIO_YEAR:
        raise ValueError(
            f"Szenario '{cf_scenario}' liegt nur fuer {CF_SCENARIO_YEAR} vor "
            f"(gewaehltes Jahr: {year})."
        )
    return pd.read_excel(path_scenarios, sheet_name=cf_scenario)[column]


def load_gridconnection_capex(rd_region: str) -> float:
    """
    Look up the grid connection cost for the given RD region.

    Returns
    -------
    float – Grid connection cost [k€/MW] (source file is in €/MW).
    """
    df = pd.read_excel(PATH_GRID_CONNECTION_COSTS)
    df.columns = df.columns.str.strip()
    row = df.loc[df["RD Region"] == rd_region]
    if row.empty:
        raise ValueError(
            f"Unknown RD region: '{rd_region}'. "
            f"Available regions: {df['RD Region'].tolist()}."
        )
    return float(row["Grid connection costs"].iloc[0]) / 1000  # €/MW -> k€/MW


def load_data(year: int, market: Redispatch13kConfig, cf_scenario: str = "Base Case") -> pd.DataFrame:
    """
    Load all input time series for a given year and return a single DataFrame.

    Parameters
    ----------
    year        : int                 – Calendar year (e.g. 2021-2025).
    market      : redispatch13kConfig – 13k-Market parameters (RD region, price cap, etc.).
    cf_scenario : str                 – Kapazitaetsfaktor-Szenario fuer PV/Wind:
        "Base Case" (Standard, wie bisher aus PATH_CF_PV/PATH_CF_WIND, Blatt =
        Jahr) oder "Hohe VLH"/"Niedrige VLH" (aus den Szenariodateien, Blatt =
        Szenarioname, nur fuer year=2025 verfuegbar). Siehe ScenarioConfig.

    Returns
    -------
    pd.DataFrame with columns:
        cf_pv, cf_wind, da_price, rd_available,
        rd_reference_price, rd_reimbursement, eua_price,
        timestamp, month
    """

    # --- Read raw files (one sheet per year, RD volume is year-independent) ---
    cf_pv_series   = _load_cf_series(PATH_CF_PV, PATH_CF_PV_SCENARIOS, "KF_PV",
                                      year, cf_scenario)
    cf_wind_series = _load_cf_series(PATH_CF_WIND, PATH_CF_WIND_SCENARIOS, "KF_Wind",
                                      year, cf_scenario)
    df_da_price = pd.read_excel(PATH_DA_PRICE, sheet_name=str(year))
    df_rd       = _load_rd_volume(year)

    # --- Validate RD region ---
    available_regions = df_rd.columns.tolist()
    if market.rd_region not in available_regions:
        raise ValueError(
            f"Unknown RD region: '{market.rd_region}'. "
            f"Available regions: {available_regions}."
        )

    # --- Build combined DataFrame ---
    df = pd.DataFrame({
        "cf_pv":        cf_pv_series,
        "cf_wind":      cf_wind_series,
        "da_price":     df_da_price["DA_Preis"],
        "rd_available": _rd_available_series(df_rd, market.rd_region),
    })

    # --- Convert € → k€ (consistent with config units) ---
    df["da_price"] = df["da_price"] / 1000

    # --- Clip numerical noise (see CF_MIN_THRESHOLD / PRICE_MIN_THRESHOLD) ---
    df["cf_pv"]   = df["cf_pv"].where(df["cf_pv"]   >= CF_MIN_THRESHOLD, 0.0)
    df["cf_wind"] = df["cf_wind"].where(df["cf_wind"] >= CF_MIN_THRESHOLD, 0.0)
    df["da_price"] = df["da_price"].where(df["da_price"].abs() >= PRICE_MIN_THRESHOLD, 0.0)

    # --- Validate row count ---
    expected_hours = 8784 if _is_leap_year(year) else 8760
    if len(df) != expected_hours:
        raise ValueError(
            f"Expected {expected_hours} rows for year {year}, "
            f"got {len(df)}."
        )

    # --- Timestamp & month ---
    df["timestamp"] = pd.date_range(
        start=f"{year}-01-01 00:00",
        periods=len(df),
        freq="h",
    )
    df["month"] = df["timestamp"].dt.month

    # --- EUA (CO2) price: daily series broadcast to every hour of the day ---
    df["eua_price"] = df["timestamp"].dt.normalize().map(_load_eua_price(year))
    df["eua_price"] = df["eua_price"] / 1000  # € → k€ (consistent with config units)

    # --- Redispatch cost columns ---
    df["rd_reference_price"] = df["da_price"].clip(upper=market.price_cap_13k)
    df["rd_reimbursement"]   = (df["rd_reference_price"] - market.price_13k).clip(lower=0)
    # da_price sits arbitrarily close to price_13k for some hours regardless of
    # the da_price noise floor above (a different threshold) - clip separately.
    df["rd_reimbursement"] = df["rd_reimbursement"].where(
        df["rd_reimbursement"] >= PRICE_MIN_THRESHOLD, 0.0
    )

    print(f"Data loaded: {len(df)} hours for year {year} (CF-Szenario: {cf_scenario})")
    print (f"RD reimbursement mean: {df['rd_reimbursement'].mean():.4f} k€/MWh")
    print (f"(DA price mean: {df['da_price'].mean():.4f} k€/MWh)")
    return df


def _is_leap_year(year: int) -> bool:
    return calendar.isleap(year)