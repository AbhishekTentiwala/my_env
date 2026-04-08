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

# Load environment variables from .env file
load_dotenv()

# ----------------------------------------------------------------------------
# Required Environment Variables
# ----------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") if os.getenv("HF_TOKEN") else os.getenv("API_KEY")

TASK_NAME = os.getenv("TASK_NAME", "plant_growth")
BENCHMARK = os.getenv("BENCHMARK", "openenv_plant_environment")
MAX_STEPS = 1000  # Note: The server allows up to 1000 steps max
MAX_TOKENS = 150
TEMPERATURE = 0.7
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI managing a tomato plant growth simulator.
    At each step, you receive the plant and soil state (biomass, LAI, soil moisture, nutrients).
    You are required to output a STRICT JSON object representing your chosen physical parameters.
    
    CRITICAL INSTRUCTION: You must provide all parameters on a scale of 1.0 to 500.0 to simulate real-world interactions between the soil and the plant!
    
    The parameters are:
    - theta_boundary (soil moisture boundary) [Scale: 1.0 to 500.0]
    - fertilizer_N (nitrogen) [Scale: 1.0 to 500.0]
    - fertilizer_P (phosphorus) [Scale: 1.0 to 500.0]
    - fertilizer_K (potassium) [Scale: 1.0 to 500.0]
    
    Example format:
    {
      "theta_boundary": 250.0,
      "fertilizer_N": 100.0,
      "fertilizer_P": 50.0,
      "fertilizer_K": 120.0
    }
    Your goal is to maximize the tomato plant's growth and score the highest reward. 
    Reply ONLY with valid JSON and nothing else.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

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

def get_action_from_llm(client: OpenAI, step: int, obs: dict, history: List[str]) -> tuple[PlantAction, str, Optional[str]]:
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
                {"role": "system", "content": SYSTEM_PROMPT},
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

def main() -> None:
    # 1. Initiate client
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    norm_score = 0.0
    
    # 2. Open standard episode logging
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        with PlantEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset()
            obs_obj = result.observation
            obs_dict = obs_obj.model_dump()
            done = False
            
            for step in range(1, MAX_STEPS + 1):
                if done:
                    break
                    
                # 3. Model Request
                action_obj, action_str, error = get_action_from_llm(client, step, obs_dict, history)
                
                # 4. Step Environment
                step_result = env.step(action_obj)
                obs_obj = step_result.observation
                obs_dict = obs_obj.model_dump()
                
                reward = step_result.reward or 0.0
                # Fallback to true if simulator marks standard done behavior
                done = getattr(obs_obj, "done", False) or getattr(obs_obj, "terminated", False)
                
                rewards.append(reward)
                steps_taken = step
                
                # 5. Log Step
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)
                history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")
                
            # Score conversion logic mapping [0, 1] range depending on max score available 
            # Modify the max threshold divisor according to known rewards boundaries of this env
            score = sum(rewards)
            norm_score = min(max(score / (MAX_STEPS * 10), 0.0), 1.0)
            success = norm_score >= SUCCESS_SCORE_THRESHOLD
            
    except Exception as e:
        print(f"[DEBUG] env error: {e}", flush=True)

    # 6. Log Exit
    log_end(success=success, steps=steps_taken, score=norm_score, rewards=rewards)

if __name__ == "__main__":
    main()
