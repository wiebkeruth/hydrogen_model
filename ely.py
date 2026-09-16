""""
Function: operating behaviour and limitations Ely

Created on Fri Jun 12
Ueberarbeitet: konstanter Systemwirkungsgrad (ElyConfig.efficiency) statt
lastabhaengiger Effizienzkurve - H2[t] = efficiency * P_ely[t] / LHV_H2, mit
P_ely[t] als GESAMTER elektrischer Leistungsaufnahme des Elektrolyseurs
(keine separate BOP/Stack-Aufteilung mehr). Die binaere Mindestlastbedingung
(z_ely) ist fest verbaut, kein weicher/optionaler Modus mehr: bei z_ely=0
ist der Elektrolyseur exakt abgeschaltet (P_ely=0, damit ueber con_h2 auch
H2=0), bei z_ely=1 ist die Leistung zwischen min_load_share*p_nom und p_nom
frei waehlbar.

@author: Wiebke G
"""

from dataclasses import dataclass
import pyomo.environ as pyo
from Configuration import EconomicsConfig, ElyConfig


# ==================================
# PRE-CALCULATIONS
# ==================================

@dataclass
class ElyParams:
    """Derived financial parameters computed from ElyConfig."""
    annuity_factor:      float
    capex_specific:      float   # [k€/MW] Anlagen-CAPEX inkl. der auf t=0 diskontierten Stackwechsel (siehe effective_ely_capex)
    capex_annual:        float   # bei ely.p_nom [k€/a]
    opex_annual:         float   # bei ely.p_nom [k€/a]
    fixed_cost_annual:   float


def effective_ely_capex(ely: ElyConfig, wacc: float) -> float:
    """
    Spezifische Investitionskosten des Elektrolyseurs [k€/MW] inklusive der
    Stackwechsel.

    Ein Stack haelt ely.stack_lifetime Jahre, die Anlage insgesamt
    ely.lifetime Jahre. Damit faellt in den Jahren stack_lifetime,
    2*stack_lifetime, ... ein Stackwechsel zu ely.stack_replacment_cost an,
    solange dieser Zeitpunkt ECHT vor dem Ende der Anlagenlebensdauer liegt
    (ein Wechsel exakt am Laufzeitende waere sinnlos). Diese Auszahlungen
    werden mit dem WACC auf t=0 diskontiert und den CAPEX zugeschlagen -
    dadurch erfasst der eine Annuitaetenfaktor ueber ely.lifetime auch die
    Stackwechsel.

    Beispiel (Tabelle 2): 2075 k€/MW, Stackwechsel 290 k€/MW alle 10 Jahre,
    Anlagenlebensdauer 30 a, WACC 8 %:
        2075 + 290/1.08^10 + 290/1.08^20 = 2271,5 k€/MW
    """
    capex = ely.capex
    if ely.stack_replacment_cost and ely.stack_lifetime and ely.stack_lifetime > 0:
        year = ely.stack_lifetime
        while year < ely.lifetime - 1e-9:
            capex += ely.stack_replacment_cost / (1 + wacc) ** year
            year += ely.stack_lifetime
    return capex


def compute_ely_params(ely: ElyConfig
                       , economics: EconomicsConfig
                       ) -> ElyParams:
    """
    Compute annuity and annual cost parameters.

    Parameters
    ----------
    ely : ElyConfig – Electrolyser configuration.

    Returns
    -------
    ElyParams – All derived parameters needed for model setup.
    """
    wacc = economics.wacc
    # CAPEX inkl. der auf t=0 diskontierten Stackwechsel (siehe oben).
    capex_specific = effective_ely_capex(ely, wacc)
    capex_total    = capex_specific * ely.p_nom
    # OPEX ist ein absoluter Wert je MW und Jahr (Tabelle 2: 51,88 EUR/kW/a),
    # kein Anteil an der CAPEX.
    opex_annual    = ely.opex_per_mw * ely.p_nom
    annuity_factor = (wacc * (1 + wacc) ** ely.lifetime) / \
                     ((1 + wacc) ** ely.lifetime - 1)
    capex_annual   = annuity_factor * capex_total
    fixed_cost_annual = capex_annual + opex_annual

    return ElyParams(
        annuity_factor    = annuity_factor,
        capex_specific    = capex_specific,
        capex_annual      = capex_annual,
        opex_annual       = opex_annual,
        fixed_cost_annual = fixed_cost_annual,
    )


# ==================================
# PYOMO VARIABLES
# ==================================

def add_ely_variables(model: pyo.ConcreteModel, ely: ElyConfig) -> None:
    """Add all electrolyser-related Pyomo variables to the model."""
    # Die Nennleistung des Elektrolyseurs ist exogen vorgegeben (Kapitel
    # 1.2.1: nur die PPA-Dimensionierung ist endogen). Sie bleibt als
    # fixierte Pyomo-Variable im Modell, damit die Auswertung sie wie jede
    # andere Groesse auslesen kann - eine Entscheidungsvariable ist sie nicht.
    model.P_nom_ely = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, ely.p_nom))
    model.P_nom_ely.fix(ely.p_nom)

    model.P_ely = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Gesamtleistung Ely [MW]
    model.H2    = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # H2-Erzeugung [t/h]

    # Ein Binaer pro Stunde: 1 = an (Leistung in [min_load_share*p_nom,
    # p_nom] frei waehlbar), 0 = exakt aus (P_ely=0, folglich H2=0 ueber
    # con_h2). Fest verbaut - macht aus dem LP immer ein MILP.
    model.z_ely = pyo.Var(model.T, domain=pyo.Binary)

    # Stetige Hilfsvariable je Zeitschritt zur Erfassung eines Startvorgangs
    # (Gleichung 5.16). Muss NICHT binaer deklariert werden: u_ely taucht nur
    # in der nach oben beschraenkenden Zyklenbedingung auf, der Solver hat
    # also keinen Anreiz, sie ueber ihr Minimum max(0, z[t]-z[t-1]) hinaus zu
    # erhoehen - im Optimum nimmt sie damit automatisch 0 oder 1 an.
    if ely.max_start_stop_cycles is not None or ely.max_starts_per_day is not None:
        model.u_ely = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 1))


# ==================================
# PYOMO CONSTRAINTS
# ==================================

def add_ely_constraints(model: pyo.ConcreteModel,
                        ely: ElyConfig,
                        params: ElyParams,
                        df=None) -> None:
    """
    Add all electrolyser constraints to the model.

    Requires model.T and the ely variables (add_ely_variables) to already exist.

    Parameters
    ----------
    df : pd.DataFrame, optional – Period time series (needs 'timestamp'),
        used to group hours by calendar day for the daily start budget.
        Required only if ely.max_starts_per_day is not None.
    """
    # Leistung nur zwischen 0 (aus) und p_nom (an) - die Mindestlast unten
    # schneidet den unteren Teil dieses Intervalls weiter ab.
    def _max_load(model, t):
        return model.P_ely[t] <= ely.p_nom * model.z_ely[t]
    model.con_max_load = pyo.Constraint(model.T, rule=_max_load)

    # Harte Mindestlast: P_ely[t] in {0} u [min_load_share*p_nom, p_nom].
    def _min_load(model, t):
        return model.P_ely[t] >= ely.min_load_share * ely.p_nom * model.z_ely[t]
    model.con_min_load = pyo.Constraint(model.T, rule=_min_load)

    # Konstanter Systemwirkungsgrad (Gleichung 5.17): H2 = efficiency *
    # P_ely / LHV_H2. Als Gleichheit formuliert; H2[t]=0 folgt bei
    # z_ely[t]=0 automatisch aus P_ely[t]=0 (con_max_load), keine gesonderte
    # Kopplung noetig.
    def _h2(model, t):
        return model.H2[t] == ely.efficiency * model.P_ely[t] / ely.LHV_H2
    model.con_h2 = pyo.Constraint(model.T, rule=_h2)

    # --- Start-Stopp-Zyklenbudget (Gleichung 5.16 + Folgebedingung) -------
    # Alterung des Elektrolyseurs, approximiert ueber die Anzahl der
    # Startvorgaenge: u_ely[t] >= z_ely[t] - z_ely[t-1] erzwingt bei einem
    # Uebergang von Stillstand zu Betrieb u_ely[t] = 1 und ist sonst trivial
    # erfuellt; die Summe der Startereignisse ist ueber das Jahr begrenzt.
    if ely.max_start_stop_cycles is not None or ely.max_starts_per_day is not None:
        t_first = min(model.T)

        def _start(model, t):
            if t == t_first:
                # Zustand vor Beginn des Betrachtungszeitraums: Anlage aus -
                # ein Betrieb in der ersten Stunde zaehlt also als Start.
                return model.u_ely[t] >= model.z_ely[t]
            return model.u_ely[t] >= model.z_ely[t] - model.z_ely[t - 1]
        model.con_start = pyo.Constraint(model.T, rule=_start)

        if ely.max_start_stop_cycles is not None:
            def _start_budget(model):
                return sum(model.u_ely[t] for t in model.T) <= ely.max_start_stop_cycles
            model.con_start_budget = pyo.Constraint(rule=_start_budget)

        # --- taegliches Startbudget (unabhaengig vom Jahresbudget oben) ---
        if ely.max_starts_per_day is not None:
            dates = df["timestamp"].dt.date
            days  = dates.unique().tolist()
            T_by_day = {d: [t for t in model.T if dates[t] == d] for d in days}

            def _start_budget_daily(model, d):
                return sum(model.u_ely[t] for t in T_by_day[d]) <= ely.max_starts_per_day
            model.con_start_budget_daily = pyo.Constraint(days, rule=_start_budget_daily)

    # --- optionale Mindestjahresmenge H2 ---------------------------------
    def _min_h2(model):
        return sum(model.H2[t] for t in model.T) >= ely.min_h2_annual
    model.con_min_h2 = pyo.Constraint(rule=_min_h2)
