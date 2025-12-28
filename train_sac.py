import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from monte_carlo_env import MonteCarloLQREnv


class RewardLoggerCallback(BaseCallback):
    """
    Logs episode rewards over time from Monitor info.
    """
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_at_end = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.timesteps_at_end.append(self.num_timesteps)
        return True


def moving_average(x, w=20):
    if len(x) < w:
        return np.array(x, dtype=float)
    x = np.array(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    out_dir = "outputs_v05"
    os.makedirs(out_dir, exist_ok=True)

    env = MonteCarloLQREnv(
        n=4,
        episode_len=200,
        action_scale=1.0,
        process_noise_std=0.03,
        param_sigma=0.05,
        obs_clip=5.0,
        seed=0,
    )
    env = Monitor(env)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=2_000,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=0,
    )

    cb = RewardLoggerCallback()
    total_steps = 50_000
    model.learn(total_timesteps=total_steps, callback=cb)

    model_path = os.path.join(out_dir, "sac_montecarlo_lqr")
    model.save(model_path)

    rewards = cb.episode_rewards
    ts = cb.timesteps_at_end

    plt.figure()
    plt.plot(ts, rewards, label="Episode return")
    ma = moving_average(rewards, w=20)
    if len(ma) > 0 and len(ts) >= len(ma):
        plt.plot(ts[-len(ma):], ma, label="Moving avg (20)")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode return")
    plt.title("SAC on Monte Carlo LQR: Learning Curve")
    plt.legend()

    fig_path = os.path.join(out_dir, "learning_curve.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")

    np.savez(
        os.path.join(out_dir, "training_logs.npz"),
        timesteps=np.array(ts),
        episode_returns=np.array(rewards),
        episode_lengths=np.array(cb.episode_lengths),
    )

    print(f"\nSaved model: {model_path}.zip")
    print(f"Saved plot : {fig_path}")
    print(f"Saved logs : {os.path.join(out_dir, 'training_logs.npz')}")


if __name__ == "__main__":
    main()
