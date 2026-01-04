import matplotlib.pyplot as plt
import numpy as np

# Fake individual episode returns (replace with real list if you saved it)
episode_returns = np.array([
    -0.12, -0.11, -0.10, -0.10, -0.09,
    -0.11, -0.10, -0.09, -0.10, -0.10
])

mean_return = episode_returns.mean()
std_return = episode_returns.std()

plt.figure()

# scatter individual episodes
plt.scatter(
    np.zeros_like(episode_returns),
    episode_returns,
    alpha=0.4,
    label="Individual episodes"
)

# mean + std
plt.errorbar(
    [0],
    [mean_return],
    yerr=[std_return],
    fmt='o',
    capsize=8,
    label="Mean ± std"
)

plt.xticks([0], ["SAC policy"])
plt.ylabel("Episode return")
plt.title("Evaluation Performance on Monte Carlo–Sampled Environments")
plt.ylim(-0.15, -0.05)
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("outputs_v05/evaluation_scatter.png", dpi=200)
plt.show()
