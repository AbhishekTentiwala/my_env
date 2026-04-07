from .water_dynamics import update_water_dynamics
from .nutrients import update_nutrient_dynamics
from .root_length import update_root_length
from .storage_carbon import update_storage_carbon
from .structure_carbon import update_structural_carbon
from .lai import update_LAI

def step_environment(state, action, params, dt=0.01):
    """
    Full RL step function for plant–soil system.

    Pipeline:
        Water → Nutrients → Storage Carbon → Biomass → LAI → Root

    Reward:
        Weighted combination of:
            - Biomass increment (primary)
            - Distance from target biomass

    Parameters
    ----------
    state : dict
    action : dict
    params : dict
    dt : float

    Returns
    -------
    state : dict
    reward : float
    done : bool
    info : dict
    """

    # ----------------------------
    # Store previous biomass
    # ----------------------------
    C_p_old = state["C_p"]

    # ----------------------------
    # 1. Water update
    # ----------------------------
    state = update_water_dynamics(state, action, params, dt)

    # ----------------------------
    # 2. Nutrient update
    # ----------------------------
    state = update_nutrient_dynamics(state, action, params, dt)

    # ----------------------------
    # 3. Storage carbon update
    # ----------------------------
    state = update_storage_carbon(state, params, dt)

    # ----------------------------
    # 4. Structural carbon update
    # ----------------------------
    state = update_structural_carbon(state, params, dt)

    # ----------------------------
    # 5. LAI update
    # ----------------------------
    state = update_LAI(state, params)

    # ----------------------------
    # 6. Root growth update
    # ----------------------------
    state = update_root_length(state, params, dt)

    # ----------------------------
    # Compute reward
    # ----------------------------
    C_p_new = state["C_p"]

    # Biomass increment
    delta_Cp = C_p_new - C_p_old

    # Target biomass
    target = params.get("target_biomass", 1.0)

    # Distance penalty
    error = abs(C_p_new - target)

    # ----------------------------
    # Reward function
    # ----------------------------
    # The previous reward calculation stagnated because target error was
    # continuously penalizing the agent at every step with a huge magnitude.
    # Now, we use raw scale increments for step reward.
    
    # Scale delta biomass to a reasonable stepwise value (e.g., multiplier of 10.0
    # since biomass increment per step dt might be very small)
    reward = delta_Cp * 10.0

    # ----------------------------
    # Termination conditions
    # ----------------------------
    done = False

    # plant death
    if C_p_new < 1e-4:
        done = True
        reward -= 1.0  # Sparse terminal penalty for death

    # success condition
    if C_p_new >= target:
        done = True
        reward += 1.0  # Sparse terminal reward for reaching target

    # ----------------------------
    # Info dict
    # ----------------------------
    info = {
        "biomass": C_p_new,
        "growth": delta_Cp,
        "target_error": error,
        "root_length": state["L"],
        "LAI": state["LAI"],
        "U_eff": state.get("U_eff", None)
    }

    return state, reward, done, info