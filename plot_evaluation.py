import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from monte_carlo_env import MonteCarloLQREnv
from evaluate import rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation returns for trained SAC policy.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs_v05/sac_montecarlo_lqr",
        help="Path to saved SB3 SAC model (without .zip).",
    )
    parser.add_argument("--episodes", type=int, default=30, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=123, help="Environment seed.")
    parser.add_argument("--out_dir", type=str, default="outputs_v05", help="Where to save the plot/data.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    env = MonteCarloLQREnv(seed=args.seed)
    model = SAC.load(args.model_path)

    returns = rollout(model, env, n_episodes=args.episodes)
    mean_return = float(returns.mean())
    std_return = float(returns.std())

    # Save raw returns for traceability/reproducibility
    np.save(os.path.join(args.out_dir, "evaluation_returns.npy"), returns)

    # Scatter with small horizontal jitter so points don't stack
    rng = np.random.default_rng(0)
    x0 = 0.0
    jitter = rng.normal(loc=0.0, scale=0.03, size=len(returns))
    xs = x0 + jitter

    plt.figure()
    plt.scatter(xs, returns, alpha=0.8, label="Episode returns")
    plt.errorbar(
        [x0],
        [mean_return],
        yerr=[std_return],
        fmt="o",
        capsize=8,
        label="Mean ± std",
    )

    plt.xticks([x0], ["SAC policy"])
    plt.ylabel("Episode return")
    plt.title("Evaluation Performance on Monte Carlo–Sampled Environments")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=9, frameon=False)

    # Set y-limits based on observed values (so it never looks “flat” or “zoomed wrong”)
    y_min = float(returns.min())
    y_max = float(returns.max())
    margin = 0.1 * max(1e-6, (y_max - y_min))
    plt.ylim(y_min - margin, y_max + margin)

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "evaluation_scatter.png")
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"Saved: {out_path}")
    print(f"Saved raw returns: {os.path.join(args.out_dir, 'evaluation_returns.npy')}")
    print(f"Mean return: {mean_return:.4f} | Std: {std_return:.4f} | Min/Max: {y_min:.4f}/{y_max:.4f}")


if __name__ == "__main__":
    main()
