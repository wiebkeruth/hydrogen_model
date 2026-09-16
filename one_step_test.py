import time
from dataclasses import replace
import main as main_mod
from Configuration import BessConfig, PPAConfig

main_mod.bess = replace(BessConfig(), p_nom=50.0)
main_mod.ppa  = PPAConfig()
t0 = time.time()
df_results = main_mod.main()
print("ELAPSED_TOTAL_S", time.time() - t0)
print(df_results["Year"].get("LCOH_final"), df_results["Year"].get("solve_time_s"))
