import argparse
import robosuite as suite
from robosuite import load_controller_config
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

parser = argparse.ArgumentParser()
parser.add_argument("--bc_ckpt", type=str, required=True, help="Path to BC checkpoint (.pth)")
parser.add_argument("--iql_ckpt", type=str, required=True, help="Path to IQL checkpoint (.pth)")
args = parser.parse_args()

device = TorchUtils.get_torch_device(try_to_use_cuda=False)

def load_policy(ckpt_path):
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)
    return policy

def process_obs(raw_obs):
    return {
        'object':              raw_obs['object-state'],
        'robot0_eef_pos':      raw_obs['robot0_eef_pos'],
        'robot0_eef_quat':     raw_obs['robot0_eef_quat'],
        'robot0_gripper_qpos': raw_obs['robot0_gripper_qpos'],
    }

def run_experiment(policy, seeds, label):
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name="Lift", robots="Panda",
        controller_configs=controller_config,
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, reward_shaping=True,
    )
    successes, rewards_all = 0, []
    for seed in seeds:
        np.random.seed(seed)
        obs = env.reset()
        policy.start_episode()
        total_reward, success = 0, False
        for _ in range(400):
            action = policy(process_obs(obs))
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if env._check_success():
                success = True
                break
        successes += int(success)
        rewards_all.append(total_reward)
        print(f"  [{label}] seed={seed:3d}: success={success}, reward={total_reward:.2f}")
    env.close()
    return successes, rewards_all

scenarios = {
    "Seed Group A (0-19)":    list(range(0, 20)),
    "Seed Group B (100-119)": list(range(100, 120)),
    "Seed Group C (200-219)": list(range(200, 220)),
}

print("Loading policies...")
bc_policy  = load_policy(args.bc_ckpt)
iql_policy = load_policy(args.iql_ckpt)

results = {}
for name, seeds in scenarios.items():
    print(f"\n=== {name} ===")
    bc_succ,  bc_rew  = run_experiment(bc_policy,  seeds, "BC ")
    iql_succ, iql_rew = run_experiment(iql_policy, seeds, "IQL")
    results[name] = {
        "bc_sr": bc_succ/len(seeds), "iql_sr": iql_succ/len(seeds),
        "bc_rew": np.mean(bc_rew),   "iql_rew": np.mean(iql_rew),
    }

print("\n\n========== Generalization Results ==========")
print(f"{'Scenario':<28} {'BC SR':>6} {'IQL SR':>7} {'BC Rew':>8} {'IQL Rew':>9}")
print("-" * 62)
for name, r in results.items():
    print(f"{name:<28} {r['bc_sr']*100:>5.1f}% {r['iql_sr']*100:>6.1f}% {r['bc_rew']:>8.2f} {r['iql_rew']:>9.2f}")
