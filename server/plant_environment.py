"""Plant-soil environment implementation for plant_soil_env."""

from uuid import uuid4

import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PlantAction, PlantObservation
    from ..simulator.state import initialize_env
    from ..simulator.step import step_environment
except ImportError:
    from models import PlantAction, PlantObservation
    from simulator.state import initialize_env
    from simulator.step import step_environment


class PlantEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._sim_state = None
        self._params = None
        self._dt = 0.01
        self._max_steps = 1000
        self._action_ranges = {
            "theta_boundary": (0.01, 0.45),
            "fertilizer_N": (0.0, 0.02),
            "fertilizer_P": (0.0, 0.01),
            "fertilizer_K": (0.0, 0.02),
        }

    def reset(self) -> PlantObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._sim_state, self._params = initialize_env(seed=None)
        return self._build_observation(
            reward=0.0,
            terminated=False,
            truncated=False,
            growth=0.0,
            target_error=float(abs(self._sim_state["C_p"] - self._params["target_biomass"])),
            u_eff=float(self._sim_state.get("U_eff", 0.0) or 0.0),
        )

    def step(self, action: PlantAction) -> PlantObservation:  # type: ignore[override]
        if self._sim_state is None or self._params is None:
            self.reset()

        self._state.step_count += 1
        mapped_action = self._map_action(action)
        self._sim_state, reward, terminated, info = step_environment(
            self._sim_state,
            mapped_action,
            self._params,
            dt=self._dt,
        )

        truncated = self._state.step_count >= self._max_steps
        done = bool(terminated or truncated)

        return self._build_observation(
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            growth=float(info.get("growth", 0.0)),
            target_error=float(info.get("target_error", 0.0)),
            u_eff=float(info.get("U_eff", 0.0) or 0.0),
            done=done,
            metadata={"step": self._state.step_count, "action_physical": mapped_action},
        )

    def _map_action(self, action: PlantAction) -> dict:
        def to_physical(normalized_value: float, lo: float, hi: float) -> float:
            x = float(np.clip(normalized_value, 0.0, 1.0))
            return lo + (hi - lo) * x

        return {
            "theta_boundary": to_physical(action.theta_boundary_norm, *self._action_ranges["theta_boundary"]),
            "fertilizer_N": to_physical(action.fertilizer_n_norm, *self._action_ranges["fertilizer_N"]),
            "fertilizer_P": to_physical(action.fertilizer_p_norm, *self._action_ranges["fertilizer_P"]),
            "fertilizer_K": to_physical(action.fertilizer_k_norm, *self._action_ranges["fertilizer_K"]),
        }

    def _build_observation(
        self,
        reward: float,
        terminated: bool,
        truncated: bool,
        growth: float,
        target_error: float,
        u_eff: float,
        done: bool = False,
        metadata: dict | None = None,
    ) -> PlantObservation:
        assert self._sim_state is not None
        metadata = metadata or {}
        return PlantObservation(
            theta_mean=float(np.mean(self._sim_state["theta"])),
            c_n_mean=float(np.mean(self._sim_state["C_N"])),
            c_p_mean=float(np.mean(self._sim_state["C_P"])),
            c_k_mean=float(np.mean(self._sim_state["C_K"])),
            storage_carbon=float(self._sim_state["C_s"]),
            biomass=float(self._sim_state["C_p"]),
            root_length=float(self._sim_state["L"]),
            lai=float(self._sim_state["LAI"]),
            growth=float(growth),
            target_error=float(target_error),
            u_eff=float(u_eff),
            terminated=bool(terminated),
            truncated=bool(truncated),
            done=bool(done),
            reward=float(reward),
            metadata=metadata,
        )

    @property
    def state(self) -> State:
        return self._state
