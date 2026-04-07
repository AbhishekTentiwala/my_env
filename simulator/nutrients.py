import numpy as np

def update_nutrient_dynamics(state, action, params, dt=0.01):
    """
    Update soil nutrient concentrations (N, P, K) for one timestep.

    Governing equation (for each nutrient i):
        ∂C_i/∂t = D_i * (1/r ∂/∂r (r ∂C_i/∂r))
                  - v(θ) ∂C_i/∂r
                  - S_i

    Parameters
    ----------
    state : dict
        Must contain:
            - "r": np.ndarray (Nr,)
            - "theta": np.ndarray (Nr,)
            - "C_N", "C_P", "C_K": np.ndarray (Nr,)
            - "L": float (root length)

    action : dict
        Control inputs:
            - "fertilizer_N", "fertilizer_P", "fertilizer_K" (optional)

    params : dict
        Model parameters:
            - "r0": root radius
            - "D_N", "D_P", "D_K": diffusion coefficients
            - "Imax_*", "Km_*": uptake parameters

    dt : float
        Time step

    Returns
    -------
    state : dict
        Updated nutrient fields
    """

    r = state["r"]
    theta = state["theta"]
    L = state["L"]

    Nr = len(r)
    dr = r[1] - r[0]
    root_zone_cells = max(1, int(params.get("root_zone_cells", 3)))

    r0 = params["r0"]

    # diffusion coefficients
    D = {
        "N": params.get("D_N", 1e-6),
        "P": params.get("D_P", 5e-7),
        "K": params.get("D_K", 8e-7),
    }

    # uptake params
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

    # water-driven velocity (mass flow)
    def velocity(theta):
        return 1e-4 * theta  # simple proportional model

    v = velocity(theta)

    # Dirichlet outer boundary concentrations C_i(R,t) = C_b,i(t)
    # Backward compatibility: if fertilizer_* is provided, treat it as an increment.
    outer_boundary = {
        "N": action.get("C_boundary_N", params.get("C_boundary_N", state["C_N"][-1])) + action.get("fertilizer_N", 0.0),
        "P": action.get("C_boundary_P", params.get("C_boundary_P", state["C_P"][-1])) + action.get("fertilizer_P", 0.0),
        "K": action.get("C_boundary_K", params.get("C_boundary_K", state["C_K"][-1])) + action.get("fertilizer_K", 0.0),
    }

    nutrients = ["N", "P", "K"]

    for nutrient in nutrients:
        C = state[f"C_{nutrient}"]
        C_new = C.copy()

        # ----------------------------
        # Diffusion + advection
        # ----------------------------
        for i in range(1, Nr - 1):
            dC_dr_plus = (C[i+1] - C[i]) / dr
            dC_dr_minus = (C[i] - C[i-1]) / dr

            diff_term = D[nutrient] * (
                (r[i+1] * dC_dr_plus - r[i-1] * dC_dr_minus) / (r[i] * dr)
            )

            adv_term = -v[i] * (C[i] - C[i-1]) / dr

            C_new[i] = C[i] + dt * (diff_term + adv_term)

        # ----------------------------
        # Root uptake (sink at r0)
        # ----------------------------
        idx = np.argmin(np.abs(r - r0))

        if idx == 0:
            C_root = float(np.mean(C[:min(root_zone_cells, Nr)]))
        else:
            i0 = max(0, idx - root_zone_cells // 2)
            i1 = min(Nr, i0 + root_zone_cells)
            C_root = float(np.mean(C[i0:i1]))

        uptake = (
            Imax[nutrient]
            * C_root
            / (Km[nutrient] + C_root + 1e-12)
        )

        # If root is at the first grid node, uptake is enforced through the
        # root-surface flux boundary below (avoid double-counting sink).
        if idx != 0:
            C_new[idx] -= dt * uptake * L

        # Root-surface flux BC at r0:
        # -D_i dC_i/dr = Imax_i*C_i0/(Km_i + C_i0)
        uptake_flux = uptake
        max_flux_from_gradient = D[nutrient] * C_new[1] / (dr + 1e-12)
        uptake_flux = min(uptake_flux, 0.95 * max_flux_from_gradient)
        C_new[0] = C_new[1] - dr * uptake_flux / (D[nutrient] + 1e-12)

        # ----------------------------
        # Boundary conditions
        # ----------------------------
        # outer boundary Dirichlet value
        C_new[-1] = outer_boundary[nutrient]

        # ----------------------------
        # Stability clamp
        # ----------------------------
        C_new = np.clip(C_new, 0.0, None)

        # update state
        state[f"C_{nutrient}"] = C_new

    return state