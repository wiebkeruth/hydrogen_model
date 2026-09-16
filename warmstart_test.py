""""
Misst, wie viel der Warmstart im BESS-Sweep tatsaechlich bringt.

Rechnet eine kurze Folge von BESS-Stufen zweimal durch - einmal ohne und
einmal mit Warmstart - und stellt die Laufzeiten gegenueber. Gemessen wird
mit demselben Modell, denselben Daten und demselben Gap wie im echten Lauf,
nur mit wenigen Stufen, damit das Ergebnis in ueberschaubarer Zeit vorliegt.

Wichtig: Nur Gurobi verwertet eine Startloesung. Laeuft das Skript mit HiGHS,
meldet es das und die beiden Durchlaeufe sind zwangslaeufig gleich schnell.

Aufruf (im Modell-Ordner, gleiche Python-Umgebung wie main.py):
    python warmstart_test.py

@author: Wiebke G
"""

import time
from dataclasses import replace

import pyomo.environ as pyo

import main as main_mod
import model as model_mod
from Configuration import BessConfig, PPAConfig

# =========================================================================
# EINSTELLUNGEN
# =========================================================================
STUFEN_MW = [0, 10, 20, 30]   # Folge von BESS-Groessen, die verglichen wird
C_RATE    = 0.5
STUNDEN   = None              # None = volles Jahr; sonst z. B. 2184 fuer ein Quartal
# =========================================================================


def _reihe(df, stufen, warmstart: bool):
    """Rechnet die Stufenfolge einmal durch und gibt die Zeiten je Stufe zurueck."""
    zeiten, ergebnisse = [], []
    z_vorstufe = None

    for mw in stufen:
        bess = replace(BessConfig(), c_rate=C_RATE, p_nom=float(mw))
        t0 = time.time()
        model, db, solve_time, converged, status = model_mod.solve_period(
            df, main_mod.ely, bess, PPAConfig(), main_mod.rfnbo,
            main_mod.market, main_mod.economics,
            warmstart_z=z_vorstufe if warmstart else None,
        )
        zeiten.append(time.time() - t0)
        ergebnisse.append(db)
        z_vorstufe = {t: round(pyo.value(model.z_ely[t])) for t in model.T}

    return zeiten, ergebnisse


def main():
    df = main_mod.load_data(year=main_mod.YEAR, market=main_mod.redispatch)
    if STUNDEN is not None:
        df = df.iloc[:STUNDEN].reset_index(drop=True)

    print("\n" + "=" * 72)
    print(f"WARMSTART-MESSUNG  |  Solver: {model_mod.SOLVER_NAME}  |  "
          f"Gap: {model_mod.MIP_REL_GAP:.3%}  |  {len(df)} Stunden")
    print(f"Stufen: {STUFEN_MW} MW bei c-rate {C_RATE}")
    print("=" * 72)

    print("\n--- Durchlauf 1: OHNE Warmstart ---")
    zeit_ohne, db_ohne = _reihe(df, STUFEN_MW, warmstart=False)

    print("\n--- Durchlauf 2: MIT Warmstart ---")
    zeit_mit, db_mit = _reihe(df, STUFEN_MW, warmstart=True)

    print("\n" + "=" * 72)
    print("ERGEBNIS")
    print("=" * 72)
    print(f'{"BESS [MW]":>10}{"ohne [s]":>12}{"mit [s]":>12}{"Ersparnis":>12}'
          f'{"Deckungsbeitrag identisch?":>30}')
    print("-" * 72)
    for mw, to, tm, do, dm in zip(STUFEN_MW, zeit_ohne, zeit_mit, db_ohne, db_mit):
        gleich = "ja" if abs(do - dm) < max(1e-6, abs(do) * model_mod.MIP_REL_GAP) else \
                 f"nein ({do:.1f} vs {dm:.1f})"
        ersparnis = f"{(1 - tm / to) * 100:+.0f} %" if to > 0 else "-"
        print(f"{mw:>10}{to:>12.1f}{tm:>12.1f}{ersparnis:>12}{gleich:>30}")
    print("-" * 72)

    summe_ohne, summe_mit = sum(zeit_ohne), sum(zeit_mit)
    print(f'{"GESAMT":>10}{summe_ohne:>12.1f}{summe_mit:>12.1f}'
          f'{(1 - summe_mit / summe_ohne) * 100:>11.0f} %')
    print()
    # Die erste Stufe hat nie eine Startloesung - sie verfaelscht den Vergleich.
    if len(STUFEN_MW) > 1:
        o, m = sum(zeit_ohne[1:]), sum(zeit_mit[1:])
        print(f"Ohne die erste Stufe (die nie einen Warmstart hat): "
              f"{o:.1f} s -> {m:.1f} s  ({(1 - m / o) * 100:+.0f} %)")
    hochrechnung = len(main_mod.BESS_SWEEP_STEPS_MW) * len(main_mod.BESS_SWEEP_C_RATES)
    print(f"\nHochgerechnet auf den vollen Sweep ({hochrechnung} Laeufe): "
          f"{summe_ohne / len(STUFEN_MW) * hochrechnung / 3600:.1f} h ohne, "
          f"{summe_mit / len(STUFEN_MW) * hochrechnung / 3600:.1f} h mit Warmstart")


if __name__ == "__main__":
    main()
