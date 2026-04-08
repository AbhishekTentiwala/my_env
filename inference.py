"""Inference Script for Plant Environment
===================================
MANDATORY - Ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
"""

import os
import textwrap
import json
import re
from typing import List, Optional
from dotenv import load_dotenv

from openai import OpenAI
from client import PlantEnv
from models import PlantAction

import asyncio

# Load environment variables from .env file
load_dotenv()

# ----------------------------------------------------------------------------
# Required Environment Variables
# ----------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") if os.getenv("HF_TOKEN") else os.getenv("API_KEY")

TASK_NAME = os.getenv("TASK_NAME", "plant_growth")
TASK_NAMES = [
    t.strip()
    for t in os.getenv(
        "TASK_NAMES",
        "plant_growth_easy,plant_growth_medium,plant_growth_hard",
    ).split(",")
    if t.strip()
]
BENCHMARK = os.getenv("BENCHMARK", "openenv_plant_environment")
MAX_STEPS = 50  # Note: The server allows up to 1000 steps max
MAX_TOKENS = 150
TEMPERATURE = 0.7
SUCCESS_SCORE_THRESHOLD = 0.1
MIN_NON_BOUNDARY_SCORE = 0.001
MAX_NON_BOUNDARY_SCORE = 0.999

def build_system_prompt(species: str, scenario: str) -> str:
        return textwrap.dedent(
                f"""
                You are an AI managing a {species} plant growth simulator.
                Scenario difficulty is {scenario}. At each step, you receive plant and soil state
                (biomass, LAI, soil moisture, nutrients).

                You must output a STRICT JSON object representing control parameters.

                CRITICAL INSTRUCTION: Provide all parameters on a scale of 1.0 to 500.0.

                The parameters are:
                - theta_boundary (soil moisture boundary) [Scale: 1.0 to 500.0]
                - fertilizer_N (nitrogen) [Scale: 1.0 to 500.0]
                - fertilizer_P (phosphorus) [Scale: 1.0 to 500.0]
                - fertilizer_K (potassium) [Scale: 1.0 to 500.0]

                Example format:
                {{
                    "theta_boundary": 250.0,
                    "fertilizer_N": 100.0,
                    "fertilizer_P": 50.0,
                    "fertilizer_K": 120.0
                }}

                Goal: maximize {species} growth reward for this {scenario} scenario.
                Reply ONLY with valid JSON and nothing else.
                """
        ).strip()


TASK_CONFIG = {
        "plant_growth_easy": {"scenario": "easy", "species": "lettuce"},
        "plant_growth_medium": {"scenario": "medium", "species": "tomato"},
        "plant_growth_hard": {"scenario": "hard", "species": "wheat"},
}

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.6f} rewards={rewards_str}", flush=True)


def log_grader(task: str, score: float, grader: str = "normalized_reward") -> None:
    print(f"[GRADER] task={task} grader={grader} score={score:.6f}", flush=True)

def build_user_prompt(step: int, obs: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current Observation: {json.dumps(obs, indent=2)}
        Previous steps history:
        {history_block}
        Send your next action strictly as a JSON object.
        """
    ).strip()

def get_action_from_llm(
    client: OpenAI,
    step: int,
    obs: dict,
    history: List[str],
    system_prompt: str,
) -> tuple[PlantAction, str, Optional[str]]:
    user_prompt = build_user_prompt(step, obs, history)
    action_str = "{}"
    error = None
    action_obj = PlantAction(
        theta_boundary=250.0,
        fertilizer_N=10.0,
        fertilizer_P=10.0,
        fertilizer_K=10.0,
    )
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        action_str = (completion.choices[0].message.content or "").strip()
        
        match = re.search(r'\{.*\}', action_str, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            action_obj = PlantAction(
                theta_boundary=float(parsed.get('theta_boundary', 250.0)),
                fertilizer_N=float(parsed.get('fertilizer_N', 10.0)),
                fertilizer_P=float(parsed.get('fertilizer_P', 10.0)),
                fertilizer_K=float(parsed.get('fertilizer_K', 10.0)),
            )
        else:
            error = "Could not locate valid JSON in model's response."
    except Exception as exc:
        error = f"Error code: 401 - {exc}" if "401" in str(exc) or "Unauthorized" in str(exc) else str(exc)
        
    # Ensure standard string format for logging output (no new lines)
    clean_action_str = action_str.replace('\n', '').replace('\r', '').replace(' ', '')
    return action_obj, clean_action_str, error

def _clamp_score_to_open_interval(score: float) -> float:
    """Keep score strictly in (0, 1) to satisfy external validators."""
    return min(max(score, MIN_NON_BOUNDARY_SCORE), MAX_NON_BOUNDARY_SCORE)


async def run_single_task(client: OpenAI, task_name: str) -> None:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False
    norm_score = MIN_NON_BOUNDARY_SCORE
    task_cfg = TASK_CONFIG.get(task_name, {"scenario": "medium", "species": "tomato"})
    system_prompt = build_system_prompt(species=task_cfg["species"], scenario=task_cfg["scenario"])

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with PlantEnv(base_url="https://dexter2012-plant-env.hf.space") as env:
            result = await env.reset(task=task_name, scenario=task_cfg["scenario"])
            obs_obj = result.observation
            obs_dict = obs_obj.model_dump()
            done = False

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                action_obj, action_str, error = get_action_from_llm(
                    client,
                    step,
                    obs_dict,
                    history,
                    system_prompt,
                )

                step_result = await env.step(action_obj)
                obs_obj = step_result.observation
                obs_dict = obs_obj.model_dump()

                reward = step_result.reward or 0.0
                done = getattr(obs_obj, "done", False) or getattr(obs_obj, "terminated", False)

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action_str, reward=reward, done=done, error=error)
                history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            raw_score = sum(rewards) / (MAX_STEPS * 10)
            norm_score = _clamp_score_to_open_interval(raw_score)
            success = norm_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] env error: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=norm_score, rewards=rewards)
    log_grader(task=task_name, score=norm_score)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    tasks = TASK_NAMES if len(TASK_NAMES) >= 3 else [
        "plant_growth_easy",
        "plant_growth_medium",
        "plant_growth_hard",
    ]
    for task in tasks:
        await run_single_task(client=client, task_name=task)

if __name__ == "__main__":
    asyncio.run(main())
