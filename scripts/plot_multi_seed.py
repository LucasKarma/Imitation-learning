import matplotlib.pyplot as plt
import numpy as np

# Multi-seed results
algos = ['BC', 'IQL']
sr_mean = [76.0, 88.0]
sr_std  = [3.3, 5.9]
rew_mean = [15.20, 13.62]
rew_std  = [1.57, 1.31]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# --- Success Rate ---
colors = ['#4C72B0', '#DD8452']
bars1 = axes[0].bar(algos, sr_mean, yerr=sr_std, capsize=8,
                     color=colors, edgecolor='black', linewidth=0.8, width=0.5)
axes[0].set_ylabel('Success Rate (%)', fontsize=13)
axes[0].set_title('Overall Success Rate (3 seeds × 50 rollouts)', fontsize=12)
axes[0].set_ylim(0, 105)
for bar, mean, std in zip(bars1, sr_mean, sr_std):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1.5,
                 f'{mean:.1f}%', ha='center', fontsize=12, fontweight='bold')

# --- Avg Reward ---
bars2 = axes[1].bar(algos, rew_mean, yerr=rew_std, capsize=8,
                     color=colors, edgecolor='black', linewidth=0.8, width=0.5)
axes[1].set_ylabel('Average Reward', fontsize=13)
axes[1].set_title('Average Reward (3 seeds × 50 rollouts)', fontsize=12)
axes[1].set_ylim(0, 22)
for bar, mean, std in zip(bars2, rew_mean, rew_std):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                 f'{mean:.2f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/multi_seed_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: results/multi_seed_comparison.png")
plt.close()

# --- Per-seed breakdown ---
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))

seeds = [1, 2, 3]
bc_sr  = [80.0, 76.0, 72.0]
iql_sr = [82.0, 86.0, 96.0]
bc_rew  = [14.023, 17.414, 14.171]
iql_rew = [15.102, 13.844, 11.920]

x = np.arange(len(seeds))
width = 0.3

axes2[0].bar(x - width/2, bc_sr, width, label='BC', color='#4C72B0', edgecolor='black', linewidth=0.8)
axes2[0].bar(x + width/2, iql_sr, width, label='IQL', color='#DD8452', edgecolor='black', linewidth=0.8)
axes2[0].set_xlabel('Training Seed', fontsize=13)
axes2[0].set_ylabel('Success Rate (%)', fontsize=13)
axes2[0].set_title('Success Rate by Seed', fontsize=12)
axes2[0].set_xticks(x)
axes2[0].set_xticklabels([f'Seed {s}' for s in seeds])
axes2[0].set_ylim(0, 105)
axes2[0].legend()

axes2[1].bar(x - width/2, bc_rew, width, label='BC', color='#4C72B0', edgecolor='black', linewidth=0.8)
axes2[1].bar(x + width/2, iql_rew, width, label='IQL', color='#DD8452', edgecolor='black', linewidth=0.8)
axes2[1].set_xlabel('Training Seed', fontsize=13)
axes2[1].set_ylabel('Average Reward', fontsize=13)
axes2[1].set_title('Average Reward by Seed', fontsize=12)
axes2[1].set_xticks(x)
axes2[1].set_xticklabels([f'Seed {s}' for s in seeds])
axes2[1].set_ylim(0, 22)
axes2[1].legend()

plt.tight_layout()
plt.savefig('results/multi_seed_per_seed.png', dpi=150, bbox_inches='tight')
print("Saved: results/multi_seed_per_seed.png")
plt.close()
