"""Simple sync client inference example for plant_soil_env."""

from plant_soil_env import PlantAction, PlantEnv


def main():
    with PlantEnv(base_url="http://localhost:8000").sync() as env:
        reset_result = env.reset()
        print("reset biomass:", reset_result.observation.biomass)

        step_result = env.step(
            PlantAction(
                theta_boundary_norm=0.4,
                fertilizer_n_norm=0.2,
                fertilizer_p_norm=0.1,
                fertilizer_k_norm=0.3,
            )
        )
        print("step reward:", step_result.reward)


if __name__ == "__main__":
    main()
