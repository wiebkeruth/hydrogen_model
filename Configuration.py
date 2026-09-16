#-*- coding: utf-8-*-
""""
Function: Definition of all input parameters

Created on Fri Jun 12 

@author: Wiebke G 
"""

from dataclasses import dataclass 

#========================================
# ELECTROLYSER 
#========================================
@dataclass
class ElyConfig:
    p_nom: float                   = 10.0          # Nominal Power [MW]. Fixed (not a decision variable).

    lifetime: float                = 30             # [years]
    min_load_share: float          = 0.20           # Minimum load [-] als Anteil von p_nom (Tabelle 2: 20 %).
    max_start_stop_cycles: float | None = 500     # [1/a]
    max_starts_per_day: float | None = 5          # [1/d]
    efficiency: float              = 0.577          # [-] Systemwirkungsgrad (bezogen auf Hu). Tabelle 2: 60 % im ersten Betriebsjahr, ueber die Stacklebensdauer gemittelt und degradationsbedingt korrigiert auf 57,7 %.
    LHV_H2: float                  = 33.33           # [MWh/t] unterer Heizwert (Hu) von Wasserstoff - physikalische Konstante
    capex: float                   = 2075.0        # [k€/MW] Tabelle 2: reine Anlagen-CAPEX OHNE Stackwechsel (2075 €/kW). Die Stackwechsel werden vom Modell selbst berechnet und aufgeschlagen - siehe ely.py, effective_ely_capex.
    opex_per_mw: float             = 51.88          # [k€/MW/a] Tabelle 2: 51,88 €/kW/a (absoluter Wert, kein CAPEX-Anteil)
    stack_replacment_cost: float   = 290.0          # [k€/MW] Tabelle 2: 290 €/kW je Stackwechsel. Wird alle stack_lifetime Jahre faellig, auf t=0 diskontiert und den CAPEX zugeschlagen (siehe ely.py, effective_ely_capex).
    stack_lifetime: float          = 10             # [a] Tabelle 2: Lebensdauer eines Stacks - bestimmt, in welchen Jahren ein Stackwechsel anfaellt.
    min_h2_annual: float           = 0.0            # Mindestjahresmenge H2 [t] (0 = keine)

@dataclass
class BessConfig:
    c_rate: float                  = 0.5            # Power-to-capacity ration (0.5 -> 2h stoarge)
    eta_charge: float              = 0.95           # Charging Efficency 
    eta_discharge: float           = 0.95           # Discharging Efficency 
    soc_initial_share: float       = 0.5            # Initial SOE [%]
    soc_min_share: float           = 0.10           # Entladetiefe (Tabelle 3: 10 %): minimal zulaessiger Ladezustand als Anteil der (gealterten) Kapazitaet - der Speicher darf also bis auf 10 % entladen werden, 90 % der Kapazitaet sind nutzbar. Siehe bess.py, con_bess_soc_min.
    lifetime: float                = 14             # [years]
    capacity_eol_share: float      = 0.70           # Restkapazitaet am Lebensdauerende [-] (SOH_EOL), z.B. 0.70 = 70 % der Nennkapazitaet. Wird in compute_bess_params zu einer ueber die Lebensdauer gemittelten Verfuegbarkeit (SOH_avg) verrechnet, siehe bess.py.
    capex: float                   = 200.0          # [k€/MWh] Tabelle 3: 200 €/kWh - KAPAZITAETSbezogen. Die Investition ergibt sich als capex * (P_nom_bess / c_rate), also ueber die Energiekapazitaet, nicht ueber die Leistung.
    opex_share: float              = 0.01           # [1/a] Tabelle 3: 1 % der BESS-CAPEX pro Jahr (Anteil, kein absoluter Wert)
    max_cycles_per_year: float     = 500            # Max. full-equivalent cycles [1/a] (Tabelle 3)
    max_cycles_per_day: float      = 2              # Max. full-equivalent cycles [1/d] (Tabelle 3)
    cycle_life: float              = 7000.0        # [-] Tabelle 3: max. 7000 Zyklen bis End-of-Life, Grundlage der 14 Jahre Lebensdauer. Nur zur Dokumentation - die Lebensdauer geht ueber lifetime ein.
    p_nom: float                   = 20.0           # Nominal Power [MW]. Fixed (not a decision variable).

#========================================
# ELECTRICTY MARKETS AND REGULATION 
#========================================
@dataclass
class PPAConfig:
    price_wind: float              = 0.085           # PPA Price Wind [k€/MWh]
    price_pv: float                = 0.060            # PPA Price PV [k€/MWh]
    p_nom_wind_max: float          = 40.0           # Upper bound on P_nom_wind decision variable [MW] -
    p_nom_pv_max: float            = 40.0           # Upper bound on P_nom_pv decision variable [MW] -
    p_nom_wind_fixed: float | None = None           # If set, fixes P_nom_wind to this value instead of sizing it
    p_nom_pv_fixed: float | None   = None           # If set, fixes P_nom_pv to this value instead of sizing it

@dataclass
class Redispatch13kConfig:
    price_13k: float               = 0.02383           # Redispatch Energy Price [k€/MWh] (Tabelle 5: 25 €/MWh; Sensitivitaet 0 / 50 €/MWh)
    price_cap_13k: float           = 0.313           # DA price cap for reimbursement [k€/MWh]
    rd_region: str                 = "H2"            # Entlastungszone. Tabelle 5: Referenzwert H2, Sensitivitaetswerte T2 / T5. Moeglich: T1-T6, H1-H2
   
@dataclass
class RfnboConfig:
    rfnbo_threshold: float         = 0.020            # DA price threshold [k€/MWh]
    eua_price_factor: float        = 0.36             # factor x EUA price threshold
    correlation_mode: str          = "monthly"        # monthly | hourly
    enforce_correlation: bool      = True             # If False, the RFNBO correlation constraint (con_correlation/con_correlation_hourly in markets.py) is skipped entirely - PPA/DA sourcing is then unconstrained by RFNBO rules and correlation_mode is ignored for the constraint. The correlation post-check in main.py still runs and reports (informationally) whether it would have held.
    correlation_ely_only: bool     = False             # Only relevant when enforce_correlation is True. If True, only da_to_ely (net of its rd_da_ely exemption) counts as "demand needing PPA backing" in con_correlation/con_correlation_hourly - da_to_bess is excluded, so the BESS can buy/sell on the DA market freely without needing matching PPA-to-DA sales to justify it. If False (default), the correlation covers the whole plant (Ely + BESS), as before.


@dataclass
class MarketConfig:
    use_ppa: bool                  = True
    use_da_buy: bool               = True
    use_ppa_da_sell: bool          = True
    use_bess_da_sell: bool         = True
    use_rd: bool                   = False
    use_curtailment: bool          = True             # allow PPA power to go unused (ppa_curtailed > 0)

#========================================
# ECONOMICS (WACC, degradation, H2 price)
#========================================

@dataclass
class EconomicsConfig:
    wacc: float                     = 0.08            # [-] real, vor Steuern
    price_h2: float                 = 6.00             # H2 sales price [€/kg] (Tabelle 6)
    sell_curtailment_ex_post: bool  = False

#========================================
# KAPAZITAETSFAKTOR-SZENARIO (PV/WIND)
#========================================
@dataclass
class ScenarioConfig:
    # "Base Case" liest wie bisher aus PV_CF_2021-2025.xlsx / Wind_CF_2021-2025.xlsx
    # (Blatt = Jahr, alle Jahre 2021-2025 verfuegbar). "Hohe VLH" / "Niedrige VLH"
    # lesen stattdessen aus PV_CF_Szenarien.xlsx / Wind_CF_Szenarien.xlsx (Blatt
    # = Szenarioname) - diese Dateien enthalten nur das Jahr 2025.
    cf_scenario: str                = "Niedrige VLH"     # "Base Case" | "Hohe VLH" | "Niedrige VLH"