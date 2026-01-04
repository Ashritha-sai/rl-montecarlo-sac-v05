import numpy as np
from stable_baselines3 import SAC
from monte_carlo_env import MonteCarloLQREnv


def rollout(model, env, n_episodes=20):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
        returns.append(ep_ret)
    return np.array(returns, dtype=float)


def main():
    env = MonteCarloLQREnv(seed=123)
    model = SAC.load("results/sac_montecarlo_lqr")

    rets = rollout(model, env, n_episodes=30)
    print("Evaluation over 30 episodes")
    print(f"Mean return: {rets.mean():.2f}")
    print(f"Std  return: {rets.std():.2f}")
    print(f"Min/Max    : {rets.min():.2f} / {rets.max():.2f}")


if __name__ == "__main__":
    main()
