"""Data models for my_env based on the plant-soil simulator."""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class PlantAction(Action):
    """Normalized control action for one simulation step."""

    theta_boundary_norm: float = Field(default=0.5, ge=0.0, le=1.0)
    fertilizer_n_norm: float = Field(default=0.0, ge=0.0, le=1.0)
    fertilizer_p_norm: float = Field(default=0.0, ge=0.0, le=1.0)
    fertilizer_k_norm: float = Field(default=0.0, ge=0.0, le=1.0)


class PlantObservation(Observation):
    """Summary observation of plant and soil state."""

    theta_mean: float = Field(default=0.0)
    c_n_mean: float = Field(default=0.0)
    c_p_mean: float = Field(default=0.0)
    c_k_mean: float = Field(default=0.0)
    storage_carbon: float = Field(default=0.0)
    biomass: float = Field(default=0.0)
    root_length: float = Field(default=0.0)
    lai: float = Field(default=0.0)
    growth: float = Field(default=0.0)
    target_error: float = Field(default=0.0)
    u_eff: float = Field(default=0.0)
    terminated: bool = Field(default=False)
    truncated: bool = Field(default=False)
