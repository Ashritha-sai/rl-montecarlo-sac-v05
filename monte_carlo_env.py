import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MonteCarloLQREnv(gym.Env):
    """
    A simple stochastic continuous-control environment (Monte Carlo per episode).

    State: x in R^n
    Action: u in R^n (bounded)
    Episode dynamics sampled at reset():
        x_{t+1} = A x_t + B u_t + noise
    Reward: scaled negative quadratic cost
        r_t = -0.1 * (x^T Q x + u^T R u)

    Key stability features (for sane training):
    - spectral radius of A capped below 1
    - smaller initial state
    - bounded observations (clipping)
    - reward scaling
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n: int = 4,
        episode_len: int = 200,
        action_scale: float = 1.0,
        process_noise_std: float = 0.03,
        param_sigma: float = 0.05,
        obs_clip: float = 5.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.n = int(n)
        self.episode_len = int(episode_len)
        self.action_scale = float(action_scale)
        self.process_noise_std = float(process_noise_std)
        self.param_sigma = float(param_sigma)
        self.obs_clip = float(obs_clip)

        # Action space
        self.action_space = spaces.Box(
            low=-self.action_scale,
            high=self.action_scale,
            shape=(self.n,),
            dtype=np.float32,
        )

        # Observation space: bounded (we clip observations)
        self.observation_space = spaces.Box(
            low=-self.obs_clip,
            high=self.obs_clip,
            shape=(self.n,),
            dtype=np.float32,
        )

        # Cost matrices
        self.Q = np.eye(self.n, dtype=np.float32)
        self.R = 0.2 * np.eye(self.n, dtype=np.float32)  # stronger action penalty encourages stability

        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.x = np.zeros(self.n, dtype=np.float32)
        self.A = np.eye(self.n, dtype=np.float32)
        self.B = np.eye(self.n, dtype=np.float32)

    def _sample_episode_params(self):
        """
        Monte Carlo sampling of episode dynamics.
        We enforce stability by capping spectral radius well below 1.
        """
        base_A = 0.80 * np.eye(self.n, dtype=np.float32)
        perturb = self.param_sigma * self.rng.normal(size=(self.n, self.n)).astype(np.float32)
        A = base_A + perturb

        # Hard cap eigenvalue magnitude to keep stable
        eigvals = np.linalg.eigvals(A)
        max_abs = float(np.max(np.abs(eigvals)))
        target = 0.95  # safely < 1
        if max_abs > target:
            A = (A / max_abs) * target

        # Keep B simple (control works consistently across episodes)
        B = np.eye(self.n, dtype=np.float32)

        self.A, self.B = A.astype(np.float32), B.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._sample_episode_params()

        # Smaller initial state (prevents immediate reward explosions)
        self.x = (0.05 * self.rng.normal(size=(self.n,))).astype(np.float32)
        self.t = 0

        info = {"A": self.A.copy(), "B": self.B.copy()}
        obs = np.clip(self.x.copy(), -self.obs_clip, self.obs_clip)
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        noise = (self.process_noise_std * self.rng.normal(size=(self.n,))).astype(np.float32)
        x_next = self.A @ self.x + self.B @ action + noise

        # Quadratic costs
        cost_state = float(self.x.T @ self.Q @ self.x)
        cost_action = float(action.T @ self.R @ action)

        # Reward scaling keeps magnitudes sane for learning
        reward = -0.1 * (cost_state + cost_action)

        self.x = x_next.astype(np.float32)
        self.t += 1

        terminated = False
        truncated = self.t >= self.episode_len

        info = {
            "cost_state": cost_state,
            "cost_action": cost_action,
        }

        obs = np.clip(self.x.copy(), -self.obs_clip, self.obs_clip)
        return obs, reward, terminated, truncated, info
