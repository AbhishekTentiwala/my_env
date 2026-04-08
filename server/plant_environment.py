"""Plant-soil environment implementation for plant_soil_env."""

from uuid import uuid4

import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PlantAction, PlantObservation
    from ..simulator.scenarios import get_scenario_config
    from ..simulator.state import initialize_env
    from ..simulator.step import step_environment
except ImportError:
    from models import PlantAction, PlantObservation
    from simulator.scenarios import get_scenario_config
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
        self._scenario_cycle = ["easy", "medium", "hard"]
        self._scenario_index = 0
        self._current_scenario = "medium"
        self._current_species = "tomato"
        self._action_ranges = {
            "theta_boundary": (0.02, 0.42),
            "fertilizer_N": (0.0, 0.04),
            "fertilizer_P": (0.0, 0.02),
            "fertilizer_K": (0.0, 0.05),
        }

    def _next_cycled_scenario(self) -> str:
        scenario = self._scenario_cycle[self._scenario_index % len(self._scenario_cycle)]
        self._scenario_index += 1
        return scenario

    def _select_scenario(self, task: str | None, scenario: str | None) -> None:
        requested = scenario or self._next_cycled_scenario()
        cfg = get_scenario_config(task=task, scenario=requested)
        self._current_scenario = cfg.scenario
        self._current_species = cfg.species
        self._action_ranges = dict(cfg.action_ranges)
        self._scenario_state_override = dict(cfg.initial_state_override)
        self._scenario_params_override = dict(cfg.params_override)

    def reset(self, **kwargs) -> PlantObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        scenario = kwargs.get("scenario")
        task = kwargs.get("task")
        seed = kwargs.get("seed")

        self._select_scenario(task=task, scenario=scenario)
        self._sim_state, self._params = initialize_env(
            seed=seed,
            initial_state_override=self._scenario_state_override,
        )
        self._params.update(self._scenario_params_override)

        return self._build_observation(
            reward=0.0,
            terminated=False,
            truncated=False,
            growth=0.0,
            target_error=float(abs(self._sim_state["C_p"] - self._params["target_biomass"])),
            u_eff=float(self._sim_state.get("U_eff", 0.0) or 0.0),
            metadata={"scenario": self._current_scenario, "species": self._current_species},
        )

    def step(self, action: PlantAction) -> PlantObservation:  # type: ignore[override]
        if self._sim_state is None or self._params is None:
            self.reset()

        self._state.step_count += 1
        
        def scale_action(val: float, r_min: float, r_max: float) -> float:
            # Map [1, 500] scale to real physical bounds [r_min, r_max]
            # Clip to 1-500 first just in case
            val_clipped = max(1.0, min(500.0, val))
            # Normalize to 0-1
            norm = (val_clipped - 1.0) / 499.0
            return r_min + norm * (r_max - r_min)

        mapped_action = {
            "theta_boundary": scale_action(action.theta_boundary, self._action_ranges["theta_boundary"][0], self._action_ranges["theta_boundary"][1]),
            "fertilizer_N": scale_action(action.fertilizer_N, self._action_ranges["fertilizer_N"][0], self._action_ranges["fertilizer_N"][1]),
            "fertilizer_P": scale_action(action.fertilizer_P, self._action_ranges["fertilizer_P"][0], self._action_ranges["fertilizer_P"][1]),
            "fertilizer_K": scale_action(action.fertilizer_K, self._action_ranges["fertilizer_K"][0], self._action_ranges["fertilizer_K"][1]),
        }

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
            metadata={
                "step": self._state.step_count,
                "action_physical": mapped_action,
                "scenario": self._current_scenario,
                "species": self._current_species,
            },
        )

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
