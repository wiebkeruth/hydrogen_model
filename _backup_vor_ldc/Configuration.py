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
    p_nom: float                   = 100.0          # Nennleistung [MW]. Fest, wenn p_nom_variable False ist
    p_nom_variable: bool           = False          # True -> P_nom_ely wird zur Entscheidungsvariablen
    p_nom_max: float               = 1000.0         # Obergrenze fuer P_nom_ely [MW] (nur bei p_nom_variable)
    lifetime: float                = 20             # [a]

    # --- Mindestlast ------------------------------------------------------
    # Wird NICHT als Nebenbedingung erzwungen. Die Effizienzkurve unten macht
    # den Bereich darunter unwirtschaftlich, sodass das Optimum stets bei
    # P_ely = 0 oder oberhalb dieser Schwelle liegt. Der Wert dient der
    # Kontrolle in main.py (h_below_min_load muss 0 bleiben) und als Untergrenze
    # des Bereichs, fuer den die Kurve hergeleitet wurde.
    min_load_share: float          = 0.20           # [-] Anteil der Nennleistung

    # --- Effizienzkurve ---------------------------------------------------
    # Konkave Huelle als  H2 <= a_k * P_ely + b_k * p_nom , ein Tupel (a, b)
    # je Segment. a in [t/MWh], b in [t/h je MW Nennleistung].
    # Hergeleitet in ely_effizienzkurve.py aus Buttler & Spliethoff (2018),
    # Abschnitt 5.1, alkalische Elektrolyse, Mittelwertfall:
    #   Stack 4,5 kWh/Nm3 bei Nennlast und 3,8 bei 25 % Last,
    #   Utilities 0,6 kWh/Nm3 lastunabhaengig konstant.
    # Systemverbrauch 56,742 MWh/t bei Nennlast, Effizienzoptimum bei 79,6 %,
    # 99,26 MWh/t bei 20 % Last. Prüfprotokoll: ely_effizienzkurve_herleitung.md
    #
    # ACHTUNG: Der 20-%-Punkt liegt unterhalb der unteren Stuetzstelle der
    # Quelle (0,1 A/cm2 entspricht 25 % Nennlast) und ist damit extrapoliert.
    # Die Quelle weist darauf hin, dass der Faraday-Wirkungsgrad dort
    # zusaetzlich einbricht - der Wert ist also eher zu guenstig.
    #
    # b ist in den unteren Segmenten negativ. Genau daraus folgt, dass H2 bei
    # kleiner Leistung negativ wird und der Bereich unterhalb der Mindestlast
    # nie gewaehlt wird. Die Abschaltschwelle des Modells liegt dadurch bei
    # rund 0,138 k EUR/MWh statt bei den oekonomisch korrekten 0,107 - in
    # diesem Band laeuft die Anlage weiter, obwohl sie Verlust macht. Diesen
    # Modellfehler in der Arbeit ausweisen.
    curve_segments: tuple          = (
        (0.023034, -0.002592),   # 20-30 %
        (0.021715, -0.002196),   # 30-40 %
        (0.020600, -0.001750),   # 40-50 %
        (0.019640, -0.001270),   # 50-60 %
        (0.018804, -0.000768),   # 60-70 %
        (0.018066, -0.000252),   # 70-80 %
        (0.017408,  0.000274),   # 80-90 %
        (0.016818,  0.000806),   # 90-100 %
    )
    power_consumption_100: float   = 56.742         # [MWh/t] bei Nennlast - Reporting/Fallback (Lcoh.py, cashflow.py)

    capex: float                   = 1_300.0        # [k EUR/MW]
    opex: float                    = 0.044          # Anteil CAPEX [1/a]
    stack_replacment_cost: float   = 200.0          # [k EUR/MW]
    stack_lifetime: float          = 10             # [a]
    min_h2_annual: float           = 0.0            # Mindestjahresmenge H2 [t] (0 = keine)

#========================================
# BESS
#========================================
@dataclass
class BessConfig:
    c_rate: float                  = 0.5            # Power-to-capacity ration (0.5 -> 2h stoarge)
    eta_charge: float              = 0.95           # Charging Efficency 
    eta_discharge: float           = 0.95           # Discharging Efficency 
    soc_initial_share: float       = 0.5            # Initial SOE [%]
    lifetime: float                = 13             # [years]
    capex: float                   = 300            # [k€/MW]
    opex: float                    = 15           # [k€/MW/a]
    max_cycles_per_year: float     = 700            # Max. full-equivalent cycles [1/a] (degradation limit)
    max_cycles_per_day: float      = 3              # Max. full-equivalent cycles [1/d] (independent daily budget)
    cycle_life: float              = 10000.0        # Full-equivalent cycles to end-of-life per datasheet
    replacement_cost_share: float  = 0.0            # Share of first-installation CAPEX incurred on replacement (inverter/container/EPC survive the cell swap)
    p_nom_bess_max: float          = 200.0          # Upper bound on the P_nom_bess decision variable [MW]
    p_nom_bess_fixed: float |None  = None        # If set, fixes P_nom_bess to this value instead of sizing it

#========================================
# ELECTRICTY MARKETS AND REGULATION 
#========================================
@dataclass
class PPAConfig:
    price_wind: float              = 0.076           # PPA Price Wind [k€/MWh]
    price_pv: float                = 0.06            # PPA Price PV [k€/MWh]
    p_nom_wind_max: float          = 192.0           # Upper bound on P_nom_wind decision variable [MW] -
    p_nom_pv_max: float            = 162.0           # Upper bound on P_nom_pv decision variable [MW] -
    p_nom_wind_fixed: float | None = None            # If set, fixes P_nom_wind to this value instead of sizing it 
    p_nom_pv_fixed: float | None   = None            # If set, fixes P_nom_pv to this value instead of sizing it 

@dataclass
class Redispatch13kConfig:
    price_13k: float               = 0.025           # Redispatch Energy Price [k€/MWh]
    price_cap_13k: float           = 0.450           # DA price cap for reimbursement [k€/MWh]
    rd_region: str                 = "T2"            # T1-T6, H1-H2
   
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
    use_bess: bool                 = True
    use_da_buy: bool               = True
    use_da_sell: bool              = True
    use_ppa_da_sell: bool          = True
    use_bess_da_sell: bool         = True
    use_rd: bool                   = True
    use_curtailment: bool          = True             # allow PPA power to go unused (ppa_curtailed > 0)

#========================================
# 20-YEAR CASHFLOW / ECONOMICS
#========================================

@dataclass
class EconomicsConfig:
    share_equity: float             = 0.20            # Equity share [-]
    i_equity: float                 = 0.12            # Required equity return [-] - MUST be a REAL rate (no inflation), both cashflow.py and Lcoh.py discount in real terms
    i_debt: float                   = 0.06            # Debt interest rate [-] - MUST be a REAL rate (no inflation), same real-terms convention as i_equity above
    i_reserve: float                = 0.02            # Return on stack reserve fund [-]
    degradation_rate: float         = 0.01            # Annual efficiency degradation [-]
    price_h2: float                 = 6.00             # H2 sales price [€/kg]
    trafo_capex: float              = 0.0             # Transformer cost [k€/MW]
    inflation_rate: float           = 0.02            # Annual OPEX cost escalation [-] (deliberate exception to the same real-terms convention as i_equity and i_debt above)
    sell_curtailment_ex_post: bool  = False         # Post-hoc reporting assumption ONLY - does not feed back into the dispatch model. When True, curtailed PPA power (ppa_curtailed) is credited as if sold at the DA price in every hour with da_price > 0, added to the DA revenue used for NPV/IRR/cashflow and the LCOH DA credit (see extract_results in main.py).