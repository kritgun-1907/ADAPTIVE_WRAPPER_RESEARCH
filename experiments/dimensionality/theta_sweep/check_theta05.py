"""Does the box reproduce the stored theta=0.5 numbers (ind17, LogReg, seeds 0-1)?"""
import json, math, sys
REF = json.loads(open(sys.argv[1]).read())
got = {str(r["seed"]): r["arms"] for r in json.load(open(sys.argv[2]))}
worst, k_ok = 0.0, True
for seed, arms in REF.items():
    for a, v in arms.items():
        g = got[seed][a]
        if [int(x) for x in g["k"]] != v["k"]:
            k_ok = False
            print(f"  seed {seed} {a}: k differs  stored={v['k']}  box={g['k']}")
        for x, y in zip(v["auc"], g["auc"]):
            if not (math.isnan(x) and math.isnan(y)):
                worst = max(worst, abs(x - y))
print(f"REPLICATION  k identical: {k_ok}   max |dAUC| = {worst:.2e}")
print("REPLICATION  PASS" if (k_ok and worst < 1e-6) else
      "REPLICATION  DIFFERS (not necessarily an error: laptop was ARM, box is x86; look at how large)")
