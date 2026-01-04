import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from monte_carlo_env import MonteCarloLQREnv

model = SAC.load("results/sac_montecarlo_lqr")
env = MonteCarloLQREnv(seed=42)

obs, _ = env.reset()
state_norms = []

done = False
while not done:
    state_norms.append(np.linalg.norm(obs))
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

state_norms = np.array(state_norms)

# Moving average (window 10)
w = 10
kernel = np.ones(w) / w
smoothed = np.convolve(state_norms, kernel, mode="valid")

plt.figure()

# Raw trajectory (faint)
plt.plot(state_norms, alpha=0.35, label="Raw ‖x‖")

# Smoothed trajectory (thicker)
plt.plot(np.arange(w-1, len(state_norms)), smoothed, linewidth=2, label=f"Moving avg (w={w})")

# Add mean band for intuition
mu = state_norms.mean()
sigma = state_norms.std()
plt.axhline(mu, linestyle="--", linewidth=1, label=f"Mean = {mu:.3f}")
plt.axhspan(mu - sigma, mu + sigma, alpha=0.15, label="± 1 std band")

plt.xlabel("Timestep")
plt.ylabel("State norm ‖x‖")
plt.title("State Norm Under Trained SAC Policy (Single Episode)")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("results/state_norm_trajectory_smoothed.png", dpi=200)
plt.show()
