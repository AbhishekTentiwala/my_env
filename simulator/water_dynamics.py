import numpy as np

def update_water_dynamics(state, action, params, dt=0.01):
    """
    Update soil water content θ(r) for one timestep.

    Governing equation:
        ∂θ/∂t = (1/r) ∂/∂r [ r K(θ) ∂ψ/∂r ] - S_w

    Parameters
    ----------
    state : dict
        Must contain:
            - "theta": np.ndarray (Nr,)
            - "r": np.ndarray (Nr,)
            - "L": float (root length)

    action : dict
        Control inputs:
            - "theta_boundary": float (outer boundary water)

    params : dict
        Model parameters:
            - "r0": root radius
            - "Ksat": saturated conductivity
            - "theta_s": saturation water content

    dt : float
        Time step

    Returns
    -------
    state : dict
        Updated state with new "theta"
    """

    theta = state["theta"]
    r = state["r"]
    L = state["L"]

    Nr = len(r)
    dr = r[1] - r[0]

    r0 = params["r0"]
    Ksat = params["Ksat"]
    theta_s = params["theta_s"]
    kw = params.get("k_w", 1e-6)
    K_w = params.get("K_w", 0.2)

    theta_boundary = action.get("theta_boundary", 0.35)

    # ----------------------------
    # Constitutive relations
    # ----------------------------
    def K(theta):
        return Ksat * (theta / theta_s) ** 3

    def psi(theta):
        return -0.1 * (theta_s / (theta + 1e-8))

    def theta_from_psi(psi_value):
        theta_est = -0.1 * theta_s / (psi_value - 1e-12)
        return np.clip(theta_est, 0.01, theta_s)

    def water_sink(theta, r):
        sink = np.zeros_like(theta)
        idx = np.argmin(np.abs(r - r0))
        sink[idx] = kw * L * (theta[idx] / (theta[idx] + K_w + 1e-12))
        return sink

    # ----------------------------
    # Compute new theta
    # ----------------------------
    theta_new = theta.copy()
    K_vals = K(theta)
    psi_vals = psi(theta)

    for i in range(1, Nr - 1):
        dpsi_dr_plus = (psi_vals[i+1] - psi_vals[i]) / dr
        dpsi_dr_minus = (psi_vals[i] - psi_vals[i-1]) / dr

        flux_plus = r[i+1] * K_vals[i+1] * dpsi_dr_plus
        flux_minus = r[i-1] * K_vals[i-1] * dpsi_dr_minus

        diffusion = (flux_plus - flux_minus) / (r[i] * dr)

        theta_new[i] = theta[i] + dt * diffusion

    # ----------------------------
    # Apply root uptake
    # ----------------------------
    theta_new -= dt * water_sink(theta, r)

    # ----------------------------
    # Boundary conditions
    # ----------------------------
    theta_new[-1] = theta_boundary       # outer boundary

    # Root-surface flux BC: K(theta) dpsi/dr = k_w * theta/(theta + K_w)
    K0 = K(np.array([theta_new[0]]))[0]
    psi1 = psi(np.array([theta_new[1]]))[0]
    root_flux = kw * theta_new[0] / (theta_new[0] + K_w + 1e-12)
    psi0_target = psi1 - dr * root_flux / (K0 + 1e-12)
    theta_new[0] = theta_from_psi(psi0_target)

    # ----------------------------
    # Stability clamp
    # ----------------------------
    theta_new = np.clip(theta_new, 0.01, theta_s)

    # ----------------------------
    # Update state
    # ----------------------------
    state["theta"] = theta_new

    return state