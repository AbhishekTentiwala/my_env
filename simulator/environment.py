from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .state import initialize_env
from .scenarios import get_scenario_config
from .step import step_environment


class _LocalEnvBase:
    """Minimal fallback base class when no RL backend is installed."""


class _LocalBox:
    def __init__(self, low, high, shape=None, dtype=np.float32):
        low_arr = np.asarray(low, dtype=dtype)
        high_arr = np.asarray(high, dtype=dtype)

        if shape is None:
            if low_arr.shape != high_arr.shape:
                raise ValueError("low/high shapes must match when shape is omitted")
            self.shape = low_arr.shape
            self.low = low_arr
            self.high = high_arr
        else:
            self.shape = tuple(shape)
            self.low = np.broadcast_to(low_arr, self.shape).astype(dtype)
            self.high = np.broadcast_to(high_arr, self.shape).astype(dtype)

        self.dtype = dtype

    def sample(self):
        return np.random.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x):
        x_arr = np.asarray(x, dtype=self.dtype)
        return (
            x_arr.shape == self.shape
            and np.all(np.isfinite(x_arr))
            and np.all(x_arr >= self.low)
            and np.all(x_arr <= self.high)
        )


def _resolve_backend():
    # Prefer OpenEnv framework if available; otherwise use meta-openenv then gym backends.
    try:
        from openenv.env import Env as OpenEnvBase  # type: ignore

        return "openenv", OpenEnvBase, _LocalBox
    except ImportError:
        pass

    try:
        import meta_openenv as openenv  # type: ignore

        env_base = getattr(openenv, "Env", _LocalEnvBase)
        spaces = getattr(openenv, "spaces", None)
        box_cls = getattr(spaces, "Box", _LocalBox) if spaces is not None else _LocalBox
        return "meta_openenv", env_base, box_cls
    except ImportError:
        pass

    try:
        import gymnasium as gym_lib  # type: ignore

        return "gymnasium", gym_lib.Env, gym_lib.spaces.Box
    except ImportError:
        pass

    try:
        import gym as gym_lib  # type: ignore

        return "gym", gym_lib.Env, gym_lib.spaces.Box
    except ImportError:
        pass

    return "local", _LocalEnvBase, _LocalBox


BACKEND_NAME, EnvBase, BoxSpace = _resolve_backend()


@dataclass(frozen=True)
class ActionScale:
    theta_boundary_min: float = 0.01
    theta_boundary_max: float = 0.45
    fertilizer_n_max: float = 0.02
    fertilizer_p_max: float = 0.01
    fertilizer_k_max: float = 0.02


class PlantEnvironment(EnvBase):
    """Meta-openenv-first wrapper around the plant-soil simulator."""

    metadata = {"render_modes": ["text"], "render_fps": 10}

    def __init__(
        self,
        Nr: int = 100,
        dt: float = 0.01,
        max_steps: int = 1000,
        normalize_observation: bool = False,
        scenario: str = "medium",
        action_scale: ActionScale | None = None,
        params_override: dict[str, Any] | None = None,
        initial_state_override: dict[str, Any] | None = None,
    ):
        super().__init__()
        self._nr = Nr
        self._dt = dt
        self._max_steps = max_steps
        self._normalize_observation = normalize_observation
        self._scenario = scenario
        scenario_cfg = get_scenario_config(scenario=scenario)

        if action_scale:
            self._action_scale = action_scale
        else:
            ranges = scenario_cfg.action_ranges
            self._action_scale = ActionScale(
                theta_boundary_min=ranges["theta_boundary"][0],
                theta_boundary_max=ranges["theta_boundary"][1],
                fertilizer_n_max=ranges["fertilizer_N"][1],
                fertilizer_p_max=ranges["fertilizer_P"][1],
                fertilizer_k_max=ranges["fertilizer_K"][1],
            )

        self._initial_state_override = dict(scenario_cfg.initial_state_override)
        if initial_state_override:
            self._initial_state_override.update(initial_state_override)

        _, params = initialize_env(Nr=self._nr)
        self.params = params
        self.params.update(scenario_cfg.params_override)
        if params_override:
            self.params.update(params_override)

        self.state: dict[str, Any] | None = None
        self._step_count = 0

        self.action_space = BoxSpace(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        obs_dim = (4 * self._nr) + 4
        self.observation_space = BoxSpace(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            np.random.seed(seed)

        self.state, params = initialize_env(
            Nr=self.params.get("Nr", self._nr),
            R=self.params.get("R", 0.1),
            r0=self.params.get("r0", 0.002),
            seed=seed,
            initial_state_override=self._initial_state_override,
        )
        params.update(self.params)
        self.params = params
        self._step_count = 0

        info = {
            "step": self._step_count,
            "biomass": self.state["C_p"],
            "root_length": self.state["L"],
            "LAI": self.state["LAI"],
            "backend": BACKEND_NAME,
        }
        return self._get_observation(), info

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Environment must be reset before calling step().")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (4,):
            raise ValueError(f"Expected action shape (4,), got {action.shape}")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        mapped_action = self._map_action(action)

        self.state, reward, terminated, info = step_environment(
            self.state,
            mapped_action,
            self.params,
            dt=self._dt,
        )
        self._step_count += 1

        truncated = self._step_count >= self._max_steps
        info = {
            **info,
            "step": self._step_count,
            "action_physical": mapped_action,
            "terminated": terminated,
            "truncated": truncated,
        }
        return self._get_observation(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.state is None:
            return "Environment not initialized. Call reset() first."

        theta_avg = float(np.mean(self.state["theta"]))
        return (
            f"step={self._step_count} "
            f"C_p={self.state['C_p']:.6f} "
            f"L={self.state['L']:.6f} "
            f"LAI={self.state['LAI']:.6f} "
            f"theta_avg={theta_avg:.6f}"
        )

    def close(self):
        self.state = None

    def _map_action(self, action: np.ndarray) -> dict[str, float]:
        scale = self._action_scale
        theta_range = scale.theta_boundary_max - scale.theta_boundary_min
        return {
            "theta_boundary": float(scale.theta_boundary_min + theta_range * action[0]),
            "fertilizer_N": float(scale.fertilizer_n_max * action[1]),
            "fertilizer_P": float(scale.fertilizer_p_max * action[2]),
            "fertilizer_K": float(scale.fertilizer_k_max * action[3]),
        }

    def _get_observation(self) -> np.ndarray:
        assert self.state is not None
        obs = np.concatenate(
            [
                self.state["theta"],
                self.state["C_N"],
                self.state["C_P"],
                self.state["C_K"],
                np.asarray(
                    [
                        self.state["C_s"],
                        self.state["C_p"],
                        self.state["L"],
                        self.state["LAI"],
                    ],
                    dtype=np.float64,
                ),
            ]
        )

        if self._normalize_observation:
            obs = self._normalize(obs)

        return obs.astype(np.float32)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        theta_end = self._nr
        n_end = theta_end + self._nr
        p_end = n_end + self._nr
        k_end = p_end + self._nr

        obs[:theta_end] = obs[:theta_end] / max(self.params.get("theta_s", 0.45), 1e-8)

        n_ref = max(self.params.get("C_boundary_N", 0.01) * 10.0, 1e-8)
        p_ref = max(self.params.get("C_boundary_P", 0.002) * 10.0, 1e-8)
        k_ref = max(self.params.get("C_boundary_K", 0.006) * 10.0, 1e-8)
        obs[theta_end:n_end] = obs[theta_end:n_end] / n_ref
        obs[n_end:p_end] = obs[n_end:p_end] / p_ref
        obs[p_end:k_end] = obs[p_end:k_end] / k_ref

        obs[k_end + 0] = obs[k_end + 0] / 2.0
        obs[k_end + 1] = obs[k_end + 1] / max(self.params.get("target_biomass", 2.5), 1e-8)
        obs[k_end + 2] = obs[k_end + 2] / 5.0
        obs[k_end + 3] = obs[k_end + 3] / 10.0
        return obs


def make_plant_environment(**kwargs) -> PlantEnvironment:
    return PlantEnvironment(**kwargs)