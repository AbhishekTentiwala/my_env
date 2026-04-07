import numpy as np

def update_structural_carbon(state, params, dt=0.01):
    """
    Update structural carbon (biomass) C_p for one timestep.

    Governing equation:
        dC_p/dt = α k_g C_s - δ C_p * (C_crit / (C_s + ε))

    where:
        α k_g C_s        : growth from storage carbon
        δ C_p * (...)    : stress-induced decay (low energy → higher death)

    Parameters
    ----------
    state : dict
        Must contain:
            - "C_s": storage carbon
            - "C_p": structural carbon (biomass)

    params : dict
        Model parameters:
            - "k_g": growth rate from storage
            - "alpha": allocation fraction to structure
            - "delta": base death rate
            - "C_crit": critical storage threshold
            - "epsilon": small constant for stability

    dt : float
        Time step

    Returns
    -------
    state : dict
        Updated "C_p"
    """

    C_s = state["C_s"]
    C_p = state["C_p"]

    # ----------------------------
    # Parameters
    # ----------------------------
    k_g = params.get("k_g", 0.05)
    alpha = params.get("alpha", 0.7)
    delta = params.get("delta", 0.02)
    C_crit = params.get("C_crit", 0.05)
    epsilon = params.get("epsilon", 1e-6)

    # ----------------------------
    # Growth term
    # ----------------------------
    growth = alpha * k_g * C_s

    # ----------------------------
    # Stress-dependent death
    # ----------------------------
    stress_factor = C_crit / (C_s + epsilon)
    death = delta * C_p * stress_factor

    # ----------------------------
    # Net change
    # ----------------------------
    dC_p_dt = growth - death

    C_p_new = C_p + dt * dC_p_dt

    # ----------------------------
    # Stability clamp
    # ----------------------------
    C_p_new = max(C_p_new, 0.0)

    state["C_p"] = C_p_new

    return state