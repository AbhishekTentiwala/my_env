# Plant Soil Env

OpenEnv environment package for plant-soil simulation.

This folder is ready to be developed in its own git branch and tested both
locally and in Docker.

## Branch Workflow (Recommended)

From repo root:

```bash
git checkout -b env
```

If the branch already exists:

```bash
git checkout env
```

## Validate OpenEnv Project

From this folder:

```bash
cd plant_soil_env
../.venv/bin/openenv validate
```

Expected result includes:

```text
[OK] plant_soil: Ready for multi-mode deployment
```

## Run Server Locally (No Docker)

Terminal 1:

```bash
cd plant_soil_env
../.venv/bin/python -m server.app --port 8000
```

Terminal 2 health check:

```bash
curl -s http://localhost:8000/health
```

Useful endpoints:

- Health: http://localhost:8000/health
- API docs: http://localhost:8000/docs
- Web UI: http://localhost:8000/web

## Docker Build and Run

Build image:

```bash
cd plant_soil_env
docker build -t plant_soil_env-env:latest -f Dockerfile .
```

Run container:

```bash
docker rm -f plant_soil_env_live >/dev/null 2>&1 || true
docker run -d --name plant_soil_env_live -p 8000:8000 plant_soil_env-env:latest
```

Check container status/logs:

```bash
docker ps --filter name=plant_soil_env_live
docker logs --tail 60 plant_soil_env_live
```

## End-to-End Client Smoke Test

From repo root:

```bash
PYTHONPATH=$PWD ./.venv/bin/python - <<'PY'
from plant_soil_env import PlantEnv, PlantAction

with PlantEnv(base_url='http://localhost:8000').sync() as env:
	reset_result = env.reset()
	step_result = env.step(
		PlantAction(
			theta_boundary_norm=0.4,
			fertilizer_n_norm=0.2,
			fertilizer_p_norm=0.1,
			fertilizer_k_norm=0.3,
		)
	)
	print('reset_biomass', round(reset_result.observation.biomass, 6))
	print('step_reward', round(float(step_result.reward), 6))
PY
```

## Quick Local Logic Test (No Server)

```bash
PYTHONPATH=$PWD ./.venv/bin/python - <<'PY'
from plant_soil_env.server.plant_environment import PlantEnvironment
from plant_soil_env.models import PlantAction

env = PlantEnvironment()
obs = env.reset()
print('local_reset', round(obs.biomass, 6))

obs2 = env.step(
	PlantAction(
		theta_boundary_norm=0.4,
		fertilizer_n_norm=0.2,
		fertilizer_p_norm=0.1,
		fertilizer_k_norm=0.3,
	)
)
print('local_step_reward', round(float(obs2.reward), 6))
PY
```

## Cleanup

Stop/remove test container:

```bash
docker rm -f plant_soil_env_live
```

## Notes

- Keep branch-scoped changes inside this folder when preparing PRs.
- Run validate and at least one smoke test before pushing.
