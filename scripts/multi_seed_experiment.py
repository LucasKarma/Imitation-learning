import json
import subprocess
import copy
import os
import glob
import argparse

SEEDS = [1, 2, 3]
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS = {
    "bc":  os.path.join(BASE_DIR, "configs/bc_lift.json"),
    "iql": os.path.join(BASE_DIR, "configs/iql_lift.json"),
}
N_ROLLOUTS = 50

def find_latest_checkpoint(output_dir):
    pattern = os.path.join(output_dir, "**", "model_epoch_2000.pth")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)

def run_training(algo, seed):
    config_path = CONFIGS[algo]
    with open(config_path) as f:
        cfg = json.load(f)

    output_dir = os.path.join(BASE_DIR, "outputs", f"{algo}_seed{seed}")
    data_path = os.path.join(BASE_DIR, "data", "low_dim_v141.hdf5")

    cfg["train"]["seed"] = seed
    cfg["train"]["output_dir"] = output_dir
    cfg["train"]["data"] = data_path
    cfg["experiment"]["name"] = f"{algo}_lift_seed{seed}"

    tmp_config = os.path.join(BASE_DIR, f".tmp_{algo}_seed{seed}.json")
    with open(tmp_config, "w") as f:
        json.dump(cfg, f, indent=4)

    print(f"\n{'='*60}")
    print(f"  Training {algo.upper()} | seed={seed}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")
    subprocess.run(
        ["python", "-m", "robomimic.scripts.train", "--config", tmp_config],
        check=True,
    )
    os.remove(tmp_config)
    return output_dir

def run_eval(algo, ckpt_path, seed):
    print(f"\n--- Evaluating {algo.upper()} seed={seed} ({N_ROLLOUTS} rollouts) ---")
    script = os.path.join(BASE_DIR, "scripts", f"{algo}_rollout.py")
    result = subprocess.run(
        ["python", script, "--ckpt", ckpt_path, "--n_rollouts", str(N_ROLLOUTS)],
        capture_output=True, text=True, check=True,
    )
    print(result.stdout)

    lines = result.stdout.strip().split("\n")
    sr_line = [l for l in lines if "Success rate" in l][0]
    rew_line = [l for l in lines if "Avg reward" in l][0]

    sr_frac = sr_line.split("=")[-1].strip().rstrip("%")
    sr = float(sr_frac) / 100.0
    avg_rew = float(rew_line.split(":")[-1].strip())
    return sr, avg_rew

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, only evaluate existing checkpoints")
    args = parser.parse_args()

    results = {}

    for algo in ["bc", "iql"]:
        results[algo] = []
        for seed in SEEDS:
            output_dir = os.path.join(BASE_DIR, "outputs", f"{algo}_seed{seed}")

            if not args.eval_only:
                run_training(algo, seed)

            ckpt = find_latest_checkpoint(output_dir)
            if ckpt is None:
                print(f"WARNING: No checkpoint found for {algo} seed={seed} in {output_dir}")
                continue

            sr, avg_rew = run_eval(algo, ckpt, seed)
            results[algo].append({"seed": seed, "sr": sr, "reward": avg_rew})

    print(f"\n\n{'='*60}")
    print(f"  MULTI-SEED RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Algo':<6} {'Seed':>4} {'SR':>8} {'Avg Reward':>12}")
    print("-" * 34)

    import numpy as np
    for algo in ["bc", "iql"]:
        for r in results[algo]:
            print(f"{algo.upper():<6} {r['seed']:>4} {r['sr']*100:>7.1f}% {r['reward']:>12.3f}")
        srs = [r["sr"] for r in results[algo]]
        rews = [r["reward"] for r in results[algo]]
        if srs:
            print(f"{algo.upper():<6} {'mean':>4} {np.mean(srs)*100:>7.1f}% {np.mean(rews):>12.3f}")
            print(f"{'':>6} {'±std':>4} {np.std(srs)*100:>7.1f}% {np.std(rews):>12.3f}")
        print("-" * 34)
