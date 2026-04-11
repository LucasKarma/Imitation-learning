"""
Demo quantity ablation: train BC and IQL on 20/50/100/200 demos,
evaluate each with 50 rollouts, and plot results.
"""
import json
import subprocess
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS = {
    "bc":  os.path.join(BASE_DIR, "configs/bc_lift.json"),
    "iql": os.path.join(BASE_DIR, "configs/iql_lift.json"),
}
DEMO_COUNTS = [20, 50, 100, 200]
N_ROLLOUTS = 50

def data_path(n):
    if n == 200:
        return os.path.join(BASE_DIR, "data", "low_dim_v141.hdf5")
    return os.path.join(BASE_DIR, "data", f"low_dim_{n}demos.hdf5")

def find_checkpoint(output_dir):
    pattern = os.path.join(output_dir, "**", "model_epoch_2000.pth")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)

def run_training(algo, n_demos):
    config_path = CONFIGS[algo]
    with open(config_path) as f:
        cfg = json.load(f)

    output_dir = os.path.join(BASE_DIR, "outputs", f"{algo}_{n_demos}demos")
    cfg["train"]["seed"] = 1
    cfg["train"]["output_dir"] = output_dir
    cfg["train"]["data"] = data_path(n_demos)
    cfg["experiment"]["name"] = f"{algo}_{n_demos}demos"

    tmp_config = os.path.join(BASE_DIR, f".tmp_{algo}_{n_demos}demos.json")
    with open(tmp_config, "w") as f:
        json.dump(cfg, f, indent=4)

    print(f"\n{'='*60}")
    print(f"  Training {algo.upper()} | {n_demos} demos")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")
    subprocess.run(
        ["python", "-m", "robomimic.scripts.train", "--config", tmp_config],
        check=True,
    )
    os.remove(tmp_config)
    return output_dir

def run_eval(algo, ckpt_path, n_demos):
    print(f"\n--- Evaluating {algo.upper()} {n_demos} demos ({N_ROLLOUTS} rollouts) ---")
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
    results = {"bc": [], "iql": []}

    for algo in ["bc", "iql"]:
        for n in DEMO_COUNTS:
            output_dir = os.path.join(BASE_DIR, "outputs", f"{algo}_{n}demos")
            run_training(algo, n)

            ckpt = find_checkpoint(output_dir)
            if ckpt is None:
                print(f"WARNING: No checkpoint for {algo} {n} demos")
                results[algo].append({"n": n, "sr": 0, "reward": 0})
                continue

            sr, avg_rew = run_eval(algo, ckpt, n)
            results[algo].append({"n": n, "sr": sr, "reward": avg_rew})

    # Print summary
    print(f"\n\n{'='*60}")
    print(f"  DEMO ABLATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Algo':<6} {'Demos':>5} {'SR':>8} {'Avg Reward':>12}")
    print("-" * 35)
    for algo in ["bc", "iql"]:
        for r in results[algo]:
            print(f"{algo.upper():<6} {r['n']:>5} {r['sr']*100:>7.1f}% {r['reward']:>12.3f}")
        print("-" * 35)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    colors = {'bc': '#4C72B0', 'iql': '#DD8452'}

    for algo in ["bc", "iql"]:
        ns = [r["n"] for r in results[algo]]
        srs = [r["sr"] * 100 for r in results[algo]]
        rews = [r["reward"] for r in results[algo]]

        axes[0].plot(ns, srs, 'o-', color=colors[algo], label=algo.upper(),
                     linewidth=2, markersize=8)
        axes[1].plot(ns, rews, 'o-', color=colors[algo], label=algo.upper(),
                     linewidth=2, markersize=8)

    axes[0].set_xlabel('Number of Demonstrations', fontsize=13)
    axes[0].set_ylabel('Success Rate (%)', fontsize=13)
    axes[0].set_title('Success Rate vs Demo Quantity', fontsize=12)
    axes[0].set_xticks(DEMO_COUNTS)
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Number of Demonstrations', fontsize=13)
    axes[1].set_ylabel('Average Reward', fontsize=13)
    axes[1].set_title('Average Reward vs Demo Quantity', fontsize=12)
    axes[1].set_xticks(DEMO_COUNTS)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "results", "demo_ablation.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close()
