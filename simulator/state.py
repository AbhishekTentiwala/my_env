import numpy as np


def _to_profile(value, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == ():
        return np.full(size, float(arr), dtype=np.float64)
    if arr.shape == (size,):
        return arr.astype(np.float64)
    raise ValueError(f"Expected scalar or shape ({size},), got {arr.shape}")

def initialize_env(
    Nr=100,
    R=0.1,
    r0=0.002,
    seed=None,
    initial_state_override=None,
):
    """
    Initialize RL environment state for plant-soil model.

    Returns:
        state (dict)
        params (dict)
    """

    # Seed hook for deterministic environment initialization.
    if seed is not None:
        np.random.seed(seed)

    # ----------------------------
    # Spatial grid
    # ----------------------------
    r = np.linspace(r0, R, Nr)

    # ----------------------------
    # Soil water (starts dry, requiring immediate watering)
    # ----------------------------
    theta = np.ones(Nr) * 0.05

    # ----------------------------
    # Nutrients (depleted soil profile)
    # ----------------------------
    C_N = np.ones(Nr) * 0.001  # Nitrogen
    C_P = np.ones(Nr) * 0.0005 # Phosphorus
    C_K = np.ones(Nr) * 0.001  # Potassium

    # ----------------------------
    # Carbon states (young seedling)
    # ----------------------------
    C_s = 0.02  # storage carbon minimal
    C_p = 0.05  # structural biomass minimal

    # ----------------------------
    # Root system
    # ----------------------------
    L = 0.8     # initial root length

    # ----------------------------
    # Leaf Area Index
    # ----------------------------
    c_L = 3.0
    beta = 0.8
    LAI = c_L * (C_p ** beta)

    # ----------------------------
    # Pack state
    # ----------------------------
    state = {
        "r": r,
        "theta": theta,
        "C_N": C_N,
        "C_P": C_P,
        "C_K": C_K,
        "C_s": C_s,
        "C_p": C_p,
        "L": L,
        "LAI": LAI
    }

    # ----------------------------
    # Model parameters (Tomato-like baseline)
    # ----------------------------
    params = {
        "r0": r0,
        "R": R,
        "Nr": Nr,

        # water (Tomato wants higher water capacity, well-draining)
        "Ksat": 5e-6,
        "theta_s": 0.5,
        "k_w": 3e-5,
        "K_w": 0.25,
        "K_theta": 0.2,
        "g_max": 0.9,

        # nutrient transport
        "D_N": 7e-7,
        "D_P": 3e-7,
        "D_K": 6e-7,
        "Km_N": 0.05,
        "Km_P": 0.01,
        "Km_K": 0.04,
        
        # Base background nutrients before fertilization (set to 0 so plant depends on action)
        "C_boundary_N": 0.0,
        "C_boundary_P": 0.0,
        "C_boundary_K": 0.0,

        # uptake (Tomatoes are heavy N and K feeders)
        "Imax_N": 5e-5,
        "Imax_P": 1e-5,
        "Imax_K": 4e-5,

        # carbon (Grow fast)
        "k_g": 0.05,
        "r_m": 0.005,
        "c_r": 0.015,

        # root (Heavy rooting system)
        "k_L": 0.05,
        "delta_L": 0.004,
        "gamma_b": 0.03,
        "K_b": 0.1,
        "K_U": 5e-9,

        # death
        "delta": 0.01,

        # photosynthesis (Needs lots of sun, highly productive)
        "light_ext_coeff": 0.7,
        "Vcmax": 2.5,
        "J": 2.2,
        "Kc": 0.35,

        # structure carbon
        "alpha": 0.65,
        "C_crit": 0.05,
        "epsilon": 1e-6,

        # canopy scaling
        "c_L": 3.5,
        "beta": 0.85,

        # RL objective (Tomatoes produce huge biomass)
        "target_biomass": 10.0
    }

    if initial_state_override:
        for key, value in initial_state_override.items():
            if key in {"theta", "C_N", "C_P", "C_K"}:
                state[key] = _to_profile(value, Nr)
            elif key in {"C_s", "C_p", "L", "LAI"}:
                state[key] = float(value)

        if "LAI" not in initial_state_override:
            c_l = params.get("c_L", 3.5)
            beta_local = params.get("beta", 0.85)
            state["LAI"] = c_l * (state["C_p"] ** beta_local)

    return state, params