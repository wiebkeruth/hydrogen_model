"""
Wasserstoffgestehungskosten (LCOH), einjaehrig und undiskontiert:

    LCOH = (CAPEX_annuisiert + OPEX + Strombezugskosten
            - DA-Erloese - §13k-Erstattung) / H2-Jahresmenge

CAPEX wird ueber den WACC-Annuitaetenfaktor (crf) auf ein Jahr umgelegt,
alle anderen Positionen sind bereits Jahreswerte aus dem optimierten
Referenzjahr. Es gibt keine Mehrjahresbetrachtung, keine Diskontierung
ueber die Projektlaufzeit, keine Ersatzinvestitionen und keine Restwerte -
jede Kostenposition wird zusaetzlich einzeln durch die H2-Jahresmenge
geteilt und als eigener Anteil an den Gestehungskosten ausgewiesen.

@author: Wiebke G
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ely import effective_ely_capex


# ============================================================
# ZINS-HELFER
# ============================================================

def crf(rate: float, n_years: float) -> float:
    """Kapitalwiedergewinnungsfaktor (Annuitaetenfaktor)."""
    if n_years <= 0:
        raise ValueError("n_years must be > 0")
    if rate == 0:
        return 1.0 / n_years
    return rate * (1 + rate) ** n_years / ((1 + rate) ** n_years - 1)


# ============================================================
# ERGEBNIS
# ============================================================

@dataclass
class LcohResult:
    lcoh:               float          # [EUR/kg]
    h2_annual_t:        float          # [t]
    wacc:               float          # [-]
    components_eur:     dict           # Jahreskosten je Position [EUR/a]
    components_per_kg:  dict           # Anteil je Position [EUR/kg]
    capex_ely_specific: float          # [k€/MW] Ely-CAPEX inkl. diskontierter Stackwechsel

    def as_dataframe(self) -> pd.DataFrame:
        rows = [
            {"Position": k, "Jahreskosten [EUR/a]": v,
             "Anteil LCOH [EUR/kg]": self.components_per_kg[k]}
            for k, v in self.components_eur.items()
        ]
        rows.append({
            "Position": "Summe / LCOH",
            "Jahreskosten [EUR/a]": sum(self.components_eur.values()),
            "Anteil LCOH [EUR/kg]": self.lcoh,
        })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        return (
            f"LCOH: {self.lcoh:8.3f} EUR/kg  "
            f"(H2-Menge: {self.h2_annual_t:.1f} t/a, WACC: {self.wacc:.2%}, "
            f"Ely-CAPEX inkl. Stackwechsel: {self.capex_ely_specific:.1f} k EUR/MW)"
        )


# ============================================================
# HAUPTRECHNUNG
# ============================================================

def compute_lcoh(
    p_nom_ely: float,          # [MW]  Ely-Nennleistung
    p_nom_bess: float,         # [MW]  BESS-Leistung
    cost_ppa: float,           # [k€/a]  PPA-Kosten (auf Erzeugung)
    cost_da: float,            # [k€/a]  DA-Einkauf
    revenue_da: float,         # [k€/a]  DA-Verkauf (Erloes)
    cost_rd: float,            # [k€/a]  §13k-Erstattung, bereits NEGATIV (Gutschrift), wie in main.py
    h2_annual_t: float,        # [t/a]  H2-Jahresmenge
    ely, bess, economics, redispatch,
    revenue_da_curtailment: float = 0.0,   # [k€/a]  Ex-post-Erloes aus abgeregelter PPA-Energie (0, wenn der Schalter aus ist)
    verbose: bool = True,
) -> LcohResult:
    """
    Einjaehrige LCOH-Rechnung ohne Diskontierung ueber mehrere Jahre.

    CAPEX (Elektrolyseur, BESS, Netzanschluss/Trafo) wird ueber den
    WACC-Annuitaetenfaktor (crf, mit der jeweiligen Komponenten-Lebensdauer)
    auf einen Jahreswert umgelegt. OPEX, Strombezugskosten, DA-Erloese und
    die §13k-Erstattung sind bereits Jahreswerte aus dem optimierten
    Dispatch (main.py, extract_results) und gehen unveraendert ein.
    """
    r = economics.wacc

    crf_ely  = crf(r, ely.lifetime)
    crf_bess = crf(r, bess.lifetime) if p_nom_bess > 0 else 0.0

    # Anlagen-CAPEX zzgl. der auf t=0 diskontierten Stackwechsel (Tabelle 2:
    # 2075 EUR/kW + Wechsel a 290 EUR/kW in Jahr 10 und 20) - siehe ely.py.
    capex_ely_specific = effective_ely_capex(ely, r)                # [k€/MW]
    capex_ely_total  = capex_ely_specific * p_nom_ely * 1000        # [EUR]
    capex_bess_total = bess.capex * (p_nom_bess / bess.c_rate) * 1000 if p_nom_bess > 0 else 0.0
    capex_grid_total = redispatch.gridconnection_capex * p_nom_bess * 1000

    capex_ely_ann  = capex_ely_total  * crf_ely
    capex_bess_ann = capex_bess_total * crf_bess
    # Netzanschluss/Trafo: dieselbe Annuisierung wie beim Elektrolyseur
    # (keine eigene Lebensdauer definiert).
    capex_grid_ann = capex_grid_total * crf_ely

    # OPEX Ely: absoluter Wert je MW und Jahr (Tabelle 2: 51,88 EUR/kW/a).
    # OPEX BESS: Anteil an der BESS-CAPEX (Tabelle 3: 1 %/a).
    opex_ely_ann  = ely.opex_per_mw * p_nom_ely * 1000              # [EUR/a]
    opex_bess_ann = bess.opex_share * capex_bess_total              # [EUR/a]

    strom_ppa_eur = cost_ppa * 1000                                 # k€ -> EUR
    strom_da_eur  = cost_da  * 1000
    da_erlose_eur = revenue_da * 1000
    rd_eur        = cost_rd * 1000                                  # bereits negativ

    h2_kg = h2_annual_t * 1000

    components_eur = {
        "CAPEX Elektrolyseur":            capex_ely_ann,
        "CAPEX BESS":                      capex_bess_ann,
        "CAPEX Netzanschluss/Trafo":       capex_grid_ann,
        "OPEX Elektrolyseur":              opex_ely_ann,
        "OPEX BESS":                       opex_bess_ann,
        "PPA-Kosten":                      strom_ppa_eur,
        "DA-Einkauf":                      strom_da_eur,
        "DA-Verkauf (Erloes)":            -da_erlose_eur,
        # Nachtraeglich bewertete Abregelungsmenge - nur besetzt, wenn
        # economics.sell_curtailment_ex_post aktiv ist (siehe main.py).
        "DA-Verkauf Curtailment (ex post)": -revenue_da_curtailment * 1000,
        "§13k-Erstattung":                 rd_eur,
    }

    total_eur = sum(components_eur.values())
    lcoh = total_eur / h2_kg if h2_kg > 0 else float("nan")

    components_per_kg = {
        k: (v / h2_kg if h2_kg > 0 else float("nan"))
        for k, v in components_eur.items()
    }

    result = LcohResult(
        lcoh=lcoh, h2_annual_t=h2_annual_t, wacc=r,
        components_eur=components_eur, components_per_kg=components_per_kg,
        capex_ely_specific=capex_ely_specific,
    )
    if verbose:
        print(result.summary())
    return result
