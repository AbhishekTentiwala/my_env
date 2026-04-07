import numpy as np

def update_LAI(state, params):
    """
    Update Leaf Area Index (LAI) from structural carbon.

    Governing relation:
        LAI = c_L * (C_p)^β

    where:
        C_p : structural carbon (biomass)
        c_L : scaling factor (leaf area per biomass)
        β   : nonlinearity (accounts for canopy structure)

    Parameters
    ----------
    state : dict
        Must contain:
            - "C_p": structural carbon

    params : dict
        Model parameters:
            - "c_L": leaf scaling coefficient
            - "beta": exponent

    Returns
    -------
    state : dict
        Updated "LAI"
    """

    C_p = state["C_p"]

    # ----------------------------
    # Parameters
    # ----------------------------
    c_L = params.get("c_L", 3.0)
    beta = params.get("beta", 0.8)

    # ----------------------------
    # Compute LAI
    # ----------------------------
    LAI = c_L * (C_p ** beta)

    # ----------------------------
    # Stability clamp
    # ----------------------------
    LAI = max(LAI, 0.0)

    state["LAI"] = LAI

    return state