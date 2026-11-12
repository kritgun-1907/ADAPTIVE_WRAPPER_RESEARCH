"""Build tier_d_theta.py from tier_d_fullnovelty.py with a handful of exact-match edits.
The original file is never touched, so its provenance hash and checkpoints stay valid."""
import pathlib, sys
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
s = src.read_text()

def sub(old, new):
    global s
    n = s.count(old)
    assert n == 1, f"patch target must occur exactly once, found {n}: {old!r}"
    s = s.replace(old, new)

# 1. theta comes from the environment, with NO default (cannot silently run at 0.5)
sub("MIN_F, THETA = 3, 0.5", 'MIN_F = 3\nTHETA = float(os.environ["FN_THETA"])')
# 2. separate checkpoint store (theta is in the provenance config, so each theta gets its own generation)
sub('"tier_d_fullnovelty", PROV)', '"theta_sweep", PROV)')
# 3. sparse_pso reads THETA from ITS OWN module; loky workers re-import it at 0.5, so set it inside the worker
sub("def run_unit(ds, model_name, seed):\n",
    "def run_unit(ds, model_name, seed):\n"
    "    assert set(ACTIVE_ARMS) <= {'all', 'sparse_pso', 'mrmr', 'lasso', 'rfe'}, "
    "'theta only reaches sparse_pso; other arms would still use 0.5'\n"
    "    _m = sys.modules[sparse_pso.__module__]\n"
    "    assert hasattr(_m, 'THETA'), 'sparse_pso module has no THETA global: theta would not reach the search'\n"
    "    _m.THETA = THETA\n")
# 4. separate output files
sub('tag = "" if set(ACTIVE_ARMS) == set(ARMS) else "_" + "-".join(sorted(ACTIVE_ARMS))',
    'tag = f"_theta{THETA:.2f}_" + "-".join(present) + "_" + "-".join(MODEL_NAMES)\n'
    '    os.makedirs("results/theta_sweep", exist_ok=True)')
assert s.count("results/tier_d_fullnovelty{tag}.json") == 2
s = s.replace("results/tier_d_fullnovelty{tag}.json", "results/theta_sweep/fn{tag}.json")
# 5. banner
sub('print("  FULL-NOVELTY OrPSOC LADDER at high p")',
    'print(f"  THETA SWEEP  theta={THETA}  (sparse_pso + matched filters only)")')
dst.write_text(s)
print("wrote", dst)
