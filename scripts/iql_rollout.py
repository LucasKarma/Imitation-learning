import robosuite as suite
from robosuite import load_controller_config
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

CKPT = "/Users/shenliuxu/robomimic_outputs/iql_lift/iql_lift_ph/20260410151318/models/model_epoch_2000.pth"

device = TorchUtils.get_torch_device(try_to_use_cuda=False)
policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=CKPT, device=device, verbose=True)

controller_config = load_controller_config(default_controller="OSC_POSE")
env = suite.make(
    env_name="Lift", robots="Panda",
    controller_configs=controller_config,
    has_renderer=False, has_offscreen_renderer=False,
    use_camera_obs=False, reward_shaping=True,
)

def process_obs(raw_obs):
    return {
        'object':              raw_obs['object-state'],
        'robot0_eef_pos':      raw_obs['robot0_eef_pos'],
        'robot0_eef_quat':     raw_obs['robot0_eef_quat'],
        'robot0_gripper_qpos': raw_obs['robot0_gripper_qpos'],
    }

n_rollouts = 50
successes = 0
rewards_all = []

for i in range(n_rollouts):
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
    print(f"Rollout {i+1:02d}: success={success}, reward={total_reward:.2f}")

print(f"\n=== IQL Results ===")
print(f"Success rate: {successes}/{n_rollouts} = {successes/n_rollouts*100:.1f}%")
print(f"Avg reward:   {np.mean(rewards_all):.3f}")
