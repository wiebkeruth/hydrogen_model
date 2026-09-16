""""
HINWEIS: Der BESS-Groessen-Sweep ist jetzt direkt in main.py integriert
(main.run_bess_sweep) und landet als eigenes Excel-Blatt "BESS_Sweep" in
derselben Ergebnisdatei wie der Hauptlauf (main.main()) - siehe main.py,
RUN_BESS_SWEEP / BESS_SWEEP_STEPS_MW / BESS_SWEEP_C_RATES.

Dieses eigenstaendige Skript (frueher: eigene Dateien/Ordner pro
BESS-Groesse inkl. Cashflow-Export) wird dadurch nicht mehr benoetigt und
ist hier nur noch als duenner Aufruf des main.py-Workflows erhalten.

Aufruf (im Modell-Ordner, mit derselben Python-Umgebung wie main.py):
    python bess_sweep.py

@author: Wiebke G
"""

import main as main_mod

if __name__ == "__main__":
    main_mod.main()
