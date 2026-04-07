import numpy as np

def compute_total_uptake(state, params):
    """
    Compute total nutrient uptake terms U_N, U_P, U_K and harmonic U_eff.

    U_i = 2*pi*r0*L * (Imax_i * C_i0 / (Km_i + C_i0))
    U_eff = (1/U_N + 1/U_P + 1/U_K)^-1
    """

    r = state["r"]
    L = state["L"]
    r0 = params["r0"]

    idx = np.argmin(np.abs(r - r0))
    root_zone_cells = max(1, int(params.get("root_zone_cells", 3)))

    # Use a small root-zone average rather than a single node to reduce
    # grid-level oscillations at the root boundary.
    if idx == 0:
        sl = slice(0, min(root_zone_cells, len(r)))
    else:
        i0 = max(0, idx - root_zone_cells // 2)
        i1 = min(len(r), i0 + root_zone_cells)
        sl = slice(i0, i1)

    concentrations = {
        "N": float(np.mean(state["C_N"][sl])),
        "P": float(np.mean(state["C_P"][sl])),
        "K": float(np.mean(state["C_K"][sl])),
    }

    Imax = {
        "N": params.get("Imax_N", 1e-6),
        "P": params.get("Imax_P", 5e-7),
        "K": params.get("Imax_K", 8e-7),
    }

    Km = {
        "N": params.get("Km_N", 0.01),
        "P": params.get("Km_P", 0.005),
        "K": params.get("Km_K", 0.01),
    }

    uptakes = {}
    for nutrient in ("N", "P", "K"):
        Ci0 = concentrations[nutrient]
        uptake_flux = Imax[nutrient] * Ci0 / (Km[nutrient] + Ci0 + 1e-12)
        uptakes[f"U_{nutrient}"] = 2.0 * np.pi * r0 * L * uptake_flux

    U_eff = 1.0 / (
        1.0 / (uptakes["U_N"] + 1e-12)
        + 1.0 / (uptakes["U_P"] + 1e-12)
        + 1.0 / (uptakes["U_K"] + 1e-12)
    )

    uptakes["U_eff"] = U_eff
    return uptakes


def compute_root_growth_rate(state, params):
    """Compute dL/dt from the lumped root equation."""

    L = state["L"]
    C_s = state["C_s"]
    r = state["r"]
    theta = state["theta"]
    r0 = params["r0"]

    # Parameters
    k_L = params.get("k_L", 0.1)
    delta_L = params.get("delta_L", 0.01)
    gamma_b = params.get("gamma_b", 0.05)
    K_b = params.get("K_b", 0.1)
    K_U = params.get("K_U", 0.05)
    K_theta = params.get("K_theta", 0.2)

    # Uptake-driven nutrient limitation
    uptake = compute_total_uptake(state, params)
    U_eff = uptake["U_eff"]

    # Root-zone water effect
    idx = np.argmin(np.abs(r - r0))
    theta_root = theta[idx]

    # Terms
    growth = k_L * C_s * (U_eff / (K_U + U_eff + 1e-12))
    decay = delta_L * L

    C_eff = C_s
    branching = (
        gamma_b
        * L
        * (C_eff / (C_eff + K_b + 1e-12))
        * (theta_root / (theta_root + K_theta + 1e-12))
    )

    dL_dt = growth - decay + branching

    diagnostics = {
        "dL_dt": dL_dt,
        "growth": growth,
        "decay": decay,
        "branching": branching,
        "theta_root": theta_root,
        **uptake,
    }
    return diagnostics


def update_root_length(state, params, dt=0.01):
    """
    Update root length L for one timestep.

    Governing equation:
        dL/dt = k_L C_s * (U_eff / (K_U + U_eff))
                - δ_L L
                + γ_b L * (C_eff / (C_eff + K_b)) * (θ / (θ + K_θ))

    where:
        growth term      : carbon + nutrient limited
        decay term       : natural root loss
        branching term   : enhanced growth via branching (carbon + water dependent)

    Parameters
    ----------
    state : dict
        Must contain:
            - "L": root length
            - "C_s": storage carbon
            - "theta": water profile
            - "C_N", "C_P", "C_K": nutrient profiles

    params : dict
        Model parameters:
            - k_L, delta_L
            - gamma_b, K_b
            - K_U (nutrient saturation)
            - K_theta (water saturation)

    dt : float
        Time step

    Returns
    -------
    state : dict
        Updated "L"
    """

    L = state["L"]
    root_diag = compute_root_growth_rate(state, params)
    dL_dt = root_diag["dL_dt"]

    L_new = L + dt * dL_dt

    # ----------------------------
    # Stability clamp
    # ----------------------------
    L_new = max(L_new, 0.0)

    state["L"] = L_new
    state.update(root_diag)

    return state