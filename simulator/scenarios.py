from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScenarioConfig:
    scenario: str
    species: str
    params_override: dict[str, Any]
    initial_state_override: dict[str, Any]
    action_ranges: dict[str, tuple[float, float]]


SCENARIOS: dict[str, ScenarioConfig] = {
    "easy": ScenarioConfig(
        scenario="easy",
        species="lettuce",
        params_override={
            "K_theta": 0.16,
            "Imax_N": 3.0e-5,
            "Imax_P": 1.2e-5,
            "Imax_K": 2.8e-5,
            "k_g": 0.065,
            "r_m": 0.004,
            "delta": 0.008,
            "k_L": 0.04,
            "target_biomass": 2.0,
        },
        initial_state_override={
            "theta": 0.12,
            "C_N": 0.003,
            "C_P": 0.0015,
            "C_K": 0.003,
            "C_s": 0.03,
            "C_p": 0.08,
            "L": 1.0,
        },
        action_ranges={
            "theta_boundary": (0.04, 0.45),
            "fertilizer_N": (0.0, 0.05),
            "fertilizer_P": (0.0, 0.02),
            "fertilizer_K": (0.0, 0.05),
        },
    ),
    "medium": ScenarioConfig(
        scenario="medium",
        species="tomato",
        params_override={
            "target_biomass": 10.0,
        },
        initial_state_override={
            "theta": 0.08,
            "C_N": 0.0018,
            "C_P": 0.0009,
            "C_K": 0.0018,
            "C_s": 0.02,
            "C_p": 0.05,
            "L": 0.85,
        },
        action_ranges={
            "theta_boundary": (0.02, 0.42),
            "fertilizer_N": (0.0, 0.04),
            "fertilizer_P": (0.0, 0.018),
            "fertilizer_K": (0.0, 0.05),
        },
    ),
    "hard": ScenarioConfig(
        scenario="hard",
        species="wheat",
        params_override={
            "K_theta": 0.26,
            "Imax_N": 2.6e-5,
            "Imax_P": 0.8e-5,
            "Imax_K": 2.0e-5,
            "k_g": 0.038,
            "r_m": 0.006,
            "delta": 0.014,
            "k_L": 0.065,
            "delta_L": 0.006,
            "target_biomass": 14.0,
        },
        initial_state_override={
            "theta": 0.04,
            "C_N": 0.0008,
            "C_P": 0.00035,
            "C_K": 0.0008,
            "C_s": 0.015,
            "C_p": 0.04,
            "L": 0.7,
        },
        action_ranges={
            "theta_boundary": (0.01, 0.3),
            "fertilizer_N": (0.0, 0.025),
            "fertilizer_P": (0.0, 0.01),
            "fertilizer_K": (0.0, 0.025),
        },
    ),
}

TASK_TO_SCENARIO = {
    "plant_growth_easy": "easy",
    "plant_growth_medium": "medium",
    "plant_growth_hard": "hard",
    "water_efficiency": "medium",
    "nutrient_balance": "hard",
}


def resolve_scenario_name(task: str | None = None, scenario: str | None = None) -> str:
    if scenario and scenario in SCENARIOS:
        return scenario
    if task and task in TASK_TO_SCENARIO:
        return TASK_TO_SCENARIO[task]
    if task:
        task_lower = task.lower()
        if "easy" in task_lower:
            return "easy"
        if "hard" in task_lower:
            return "hard"
        if "medium" in task_lower:
            return "medium"
    return "medium"


def get_scenario_config(task: str | None = None, scenario: str | None = None) -> ScenarioConfig:
    resolved = resolve_scenario_name(task=task, scenario=scenario)
    return SCENARIOS[resolved]
