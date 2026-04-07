import numpy as np
from .root_length import compute_root_growth_rate

def update_storage_carbon(state, params, dt=0.01):
    """
    Update storage carbon C_s for one timestep.

    Governing equation:
        dC_s/dt = A - k_g C_s - r_m C_p - c_r dL/dt

    where:
        A       : photosynthesis (assimilation)
        k_g C_s : carbon used for growth
        r_m C_p : maintenance respiration
        c_r dL/dt : root growth cost

    Parameters
    ----------
    state : dict
        Must contain:
            - "C_s": storage carbon
            - "C_p": structural biomass
            - "L": root length
            - "LAI": leaf area index
            - "theta": water profile
            - "C_N", "C_P", "C_K": nutrients

    params : dict
        Model parameters:
            - k_g, r_m, c_r
            - photosynthesis parameters

    dt : float
        Time step

    Returns
    -------
    state : dict
        Updated "C_s"
    """

    C_s = state["C_s"]
    C_p = state["C_p"]
    LAI = state["LAI"]

    theta = state["theta"]

    # ----------------------------
    # Parameters
    # ----------------------------
    k_g = params.get("k_g", 0.05)
    r_m = params.get("r_m", 0.01)
    c_r = params.get("c_r", 0.02)

    k = params.get("light_ext_coeff", 0.5)

    # photosynthesis params
    Vcmax = params.get("Vcmax", 1.0)
    J = params.get("J", 1.0)
    Kc = params.get("Kc", 0.5)
    g_max = params.get("g_max", 1.0)
    K_theta = params.get("K_theta", 0.2)

    # ----------------------------
    # Water effect (stomatal conductance)
    # ----------------------------
    theta_mean = np.mean(theta)
    g_s = g_max * theta_mean / (theta_mean + K_theta + 1e-12)

    C_leaf = g_s  # proportional

    # ----------------------------
    # Photosynthesis
    # ----------------------------
    W_c = Vcmax * (C_leaf / (C_leaf + Kc))
    W_j = J * (C_leaf / (4 * (C_leaf + Kc)))

    A = min(W_c, W_j) * (1 - np.exp(-k * LAI))

    # ----------------------------
    # Root growth rate (approx)
    # ----------------------------
    dL_dt = compute_root_growth_rate(state, params)["dL_dt"]

    # ----------------------------
    # Carbon balance
    # ----------------------------
    dC_s_dt = A - k_g * C_s - r_m * C_p - c_r * dL_dt

    C_s_new = C_s + dt * dC_s_dt

    # stability
    C_s_new = max(C_s_new, 0.0)

    state["C_s"] = C_s_new

    return state