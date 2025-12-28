# SAC on a Monte Carlo-Simulated Stochastic Control Environment

This project trains Soft Actor-Critic (SAC) using Stable-Baselines3 on a custom Gymnasium environment
where episode dynamics are randomized via Monte Carlo sampling.

## Files
- `monte_carlo_env.py`: Monte Carlo stochastic environment (stable linear dynamics + noise)
- `train_sac.py`: trains SAC and saves learning curve + logs
- `evaluate.py`: evaluates trained policy over multiple episodes

## Setup
```bash
pip install -U gymnasium stable-baselines3 torch matplotlib numpy
