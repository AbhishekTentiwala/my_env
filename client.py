"""My Env environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import PlantAction, PlantObservation


class PlantEnv(EnvClient[PlantAction, PlantObservation, State]):
    """Client for my_env plant-soil environment."""

    def _step_payload(self, action: PlantAction) -> Dict:
        return {
            "theta_boundary": action.theta_boundary,
            "fertilizer_N": action.fertilizer_N,
            "fertilizer_P": action.fertilizer_P,
            "fertilizer_K": action.fertilizer_K,
        }

    def _parse_result(self, payload: Dict) -> StepResult[PlantObservation]:
        obs_data = payload.get("observation", {})
        observation = PlantObservation(
            theta_mean=obs_data.get("theta_mean", 0.0),
            c_n_mean=obs_data.get("c_n_mean", 0.0),
            c_p_mean=obs_data.get("c_p_mean", 0.0),
            c_k_mean=obs_data.get("c_k_mean", 0.0),
            storage_carbon=obs_data.get("storage_carbon", 0.0),
            biomass=obs_data.get("biomass", 0.0),
            root_length=obs_data.get("root_length", 0.0),
            lai=obs_data.get("lai", 0.0),
            growth=obs_data.get("growth", 0.0),
            target_error=obs_data.get("target_error", 0.0),
            u_eff=obs_data.get("u_eff", 0.0),
            terminated=obs_data.get("terminated", payload.get("done", False)),
            truncated=obs_data.get("truncated", False),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
