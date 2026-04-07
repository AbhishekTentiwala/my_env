import numpy as np

def initialize_env(
    Nr=100,
    R=0.1,
    r0=0.002,
    seed=None
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
    # Soil water (field-capacity-like starting condition)
    # ----------------------------
    theta = np.ones(Nr) * 0.28

    # ----------------------------
    # Nutrients (uniform initial, generic fertile loam profile)
    # ----------------------------
    C_N = np.ones(Nr) * 0.01   # Nitrogen
    C_P = np.ones(Nr) * 0.002  # Phosphorus
    C_K = np.ones(Nr) * 0.006  # Potassium

    # ----------------------------
    # Carbon states
    # ----------------------------
    C_s = 0.15  # storage carbon
    C_p = 0.25  # structural biomass

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
    # Model parameters (subset)
    # ----------------------------
    params = {
        "r0": r0,
        "R": R,
        "Nr": Nr,

        # water
        "Ksat": 3e-6,
        "theta_s": 0.45,
        "k_w": 2e-5,
        "K_w": 0.2,
        "K_theta": 0.18,
        "g_max": 0.8,

        # nutrient transport
        "D_N": 7e-7,
        "D_P": 3e-7,
        "D_K": 5e-7,
        "Km_N": 0.02,
        "Km_P": 0.004,
        "Km_K": 0.015,
        "C_boundary_N": 0.01,
        "C_boundary_P": 0.002,
        "C_boundary_K": 0.006,

        # uptake
        "Imax_N": 2e-5,
        "Imax_P": 6e-6,
        "Imax_K": 1.5e-5,

        # carbon
        "k_g": 0.03,
        "r_m": 0.004,
        "c_r": 0.01,

        # root
        "k_L": 0.03,
        "delta_L": 0.003,
        "gamma_b": 0.02,
        "K_b": 0.08,
        "K_U": 5e-9,

        # death
        "delta": 0.01,

        # photosynthesis
        "light_ext_coeff": 0.65,
        "Vcmax": 2.0,
        "J": 1.8,
        "Kc": 0.3,

        # structure carbon
        "alpha": 0.65,
        "C_crit": 0.04,
        "epsilon": 1e-6,

        # canopy scaling
        "c_L": 3.0,
        "beta": 0.8,

        # RL objective
        "target_biomass": 2.5
    }

    return state, params