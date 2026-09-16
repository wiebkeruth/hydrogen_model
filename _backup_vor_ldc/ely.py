""""
Function: operating behaviour and limitations Ely

Created on Fri Jun 12

@author: Wiebke G

Effizienzmodell des Elektrolyseurs (Monolith), Kurve in ABSOLUTEN MW:

    H2[t] <= a_k * P_ely[t] + b_k * p_nom        fuer jedes Segment k

Die Kurve ist ueber [min_load_share * p_nom, p_nom] konkav. Weil H2 maximiert
wird, bindet in jedem Punkt automatisch das niedrigste Segment - die
Darstellung als Satz oberer Schranken ist damit exakt und braucht weder SOS2
noch Binaervariablen.

H2 ist bewusst als Reals deklariert, darf also negativ werden. Der mit p_nom
skalierte Achsenabschnitt b_k bildet den lastunabhaengigen Eigenverbrauch ab;
bei P_ely = 0 folgt daraus H2 < 0. Zusammen mit der Konkavitaet sorgt das
dafuer, dass sich ein Betriebspunkt zwischen 0 und der Mindestlast nie lohnt:
Die steilste Segmentsteigung liegt am unteren Rand der Kurve. Faellt der
Grenzertrag darunter, faellt er ueberall darunter, und der Optimierer geht
auf den Rand P_ely = 0 statt auf einen Zwischenwert.

Preis dafuer: Eine Stillstandsstunde erscheint mit H2 = min_k(b_k) * p_nom,
also einem fiktiven Erloesverlust, und die Abschaltschwelle des Modells liegt
hoeher als die oekonomisch korrekte. Die Auswertung in main.py trennt deshalb
H2_opt (roh, passend zur Zielfunktion) von H2_bereinigt (auf null geklemmt).
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
    capex_annual:        float   # at ely.p_nom - only meaningful when p_nom_variable is False
    opex_annual:         float
    fixed_cost_annual:   float


def compute_ely_params(ely: ElyConfig, economics: EconomicsConfig) -> ElyParams:
    """Compute annuity and annual fixed costs."""
    capex_total    = ely.capex * ely.p_nom
    opex_annual    = capex_total * ely.opex
    wacc = economics.share_equity * economics.i_equity + (1 - economics.share_equity) * economics.i_debt
    annuity_factor = (wacc * (1 + wacc) ** ely.lifetime) / ((1 + wacc) ** ely.lifetime - 1)
    capex_annual   = annuity_factor * capex_total
    return ElyParams(annuity_factor    = annuity_factor,
                     capex_annual      = capex_annual,
                     opex_annual       = opex_annual,
                     fixed_cost_annual = capex_annual + opex_annual)


# ==================================
# PYOMO VARIABLES
# ==================================

def add_ely_variables(model: pyo.ConcreteModel, ely: ElyConfig) -> None:
    """Add all electrolyser-related Pyomo variables to the model."""
    if ely.p_nom_variable:
        model.P_nom_ely = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, ely.p_nom_max))
    else:
        model.P_nom_ely = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, ely.p_nom))
        model.P_nom_ely.fix(ely.p_nom)

    # Leistungsaufnahme des Elektrolyseursystems [MW]. Diese Variable
    # bilanziert markets.py gegen PPA, Day-Ahead und Batterie.
    model.P_ely = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # Wasserstofferzeugung [t/h]. Reals, nicht NonNegativeReals - siehe
    # Modulkopf: der negative Ast in Stillstandsstunden ist der Mechanismus,
    # der den Bereich unterhalb der Mindestlast unattraktiv macht.
    model.H2 = pyo.Var(model.T, domain=pyo.Reals)


# ==================================
# PYOMO CONSTRAINTS
# ==================================

def add_ely_constraints(model: pyo.ConcreteModel,
                        ely: ElyConfig,
                        params: ElyParams,
                        time_share: float) -> None:
    """Add all electrolyser constraints. Requires model.T and the ely variables."""

    def _cap(model, t):
        return model.P_ely[t] <= model.P_nom_ely
    model.con_cap = pyo.Constraint(model.T, rule=_cap)

    # --- Effizienzkurve, konkave Huelle in absoluten MW -------------------
    model.S_curve = pyo.Set(initialize=range(len(ely.curve_segments)))

    def _curve(model, t, k):
        a, b = ely.curve_segments[k]
        return model.H2[t] <= a * model.P_ely[t] + b * ely.p_nom
    model.con_curve = pyo.Constraint(model.T, model.S_curve, rule=_curve)

    # --- optionale Mindestjahresmenge H2 ---------------------------------
    def _min_h2(model):
        return sum(model.H2[t] for t in model.T) >= ely.min_h2_annual * time_share
    model.con_min_h2 = pyo.Constraint(rule=_min_h2)
