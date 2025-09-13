import numpy as np
import time

def simulate(inputs):
    """
    This function is a proxy for a full 3D implicit reservoir simulator engine.
    It uses a physics-informed analytical model to generate realistic production
    profiles and mock 3D pressure data based on the detailed inputs provided.

    Args:
        inputs (dict): A dictionary containing all simulation parameters
                       (grid, rock, pvt, schedule, msw, etc.).

    Returns:
        dict: A dictionary containing the simulation results, including time series
              data (t_days, qg_Mscfd, qo_STBpd) and 3D/2D spatial data (p3d_psi).
    """
    print("--- Running Full 3D Engine Proxy ---")
    start_time = time.time()

    # --- 1. Unpack key inputs from the dictionary ---
    grid = inputs.get('grid', {})
    nx, ny, nz = grid.get('nx', 100), grid.get('ny', 50), grid.get('nz', 10)
    dx, dy, dz = grid.get('dx', 50.), grid.get('dy', 50.), grid.get('dz', 20.)

    rock = inputs.get('rock', {})
    # Use the mean of the input kx property field as a proxy for average permeability
    kx_array = rock.get('kx_md', np.full((nz, ny, nx), 0.05))
    k_avg_md = np.mean(kx_array)

    init = inputs.get('init', {})
    p_init_psi = init.get('p_init_psi', 5000.)
    pwf_min_psi = init.get('pwf_min_psi', 2500.)

    schedule = inputs.get('schedule', {})
    bhp_psi = schedule.get('bhp_psi', 2500.)
    
    msw = inputs.get('msw', {})
    L_ft = msw.get('L_ft', 10000.)
    xf_ft = msw.get('xf_ft', 300.)
    hf_ft = msw.get('hf_ft', 180.)
    n_laterals = msw.get('laterals', 2)

    pvt = inputs.get('pvt', {})
    Rs_pb_scf_stb = pvt.get('Rs_pb_scf_stb', 650.)

    # --- 2. Calculate Decline Curve Parameters based on Physics ---
    # These formulas are empirical but link the inputs to the model behavior
    
    # Initial Rate (qi) is a function of drawdown, permeability, and completion size
    drawdown = p_init_psi - bhp_psi
    srv_proxy = (L_ft / 10000.) * (xf_ft / 300.) * (hf_ft / 180.) * n_laterals
    qi_oil_base = 1000 * srv_proxy * (k_avg_md / 0.05)**0.4 * (drawdown / 2500.)**0.8
    
    # GOR determines the initial gas rate
    gor_init = max(1000., Rs_pb_scf_stb * (p_init_psi / pvt.get('pb_psi', 5200.))**0.5)
    qi_gas_base = qi_oil_base * gor_init / 1000. # Mscfd

    qi_oil = np.clip(qi_oil_base, 200, 3000)
    qi_gas = np.clip(qi_gas_base, 2000, 30000)

    # Initial Decline (Di) is faster for higher permeability
    Di_yr = 0.8 * (k_avg_md / 0.05)**0.3
    Di_day = Di_yr / 365.0

    # b-Factor (hyperbolic exponent)
    b = 1.2
    
    # --- 3. Generate Production Forecast ---
    t_days = np.linspace(0, 30 * 365, 361)
    
    # Arps Hyperbolic Decline Equation
    qo_STBpd = qi_oil / (1 + b * Di_day * t_days)**(1/b)
    qg_Mscfd = qi_gas / (1 + b * Di_day * t_days)**(1/b)
    
    # Ensure rates don't fall below a minimum economic limit
    qo_STBpd[qo_STBpd < 5] = 0
    qg_Mscfd[qg_Mscfd < 20] = 0

    # --- 4. Mock 3D Pressure and Saturation grids for visualization ---
    # This creates a visually plausible depletion effect around the wells
    
    p3d_psi = np.full((nz, ny, nx), p_init_psi, dtype=np.float32)
    
    # Define well locations in the grid
    lat_j_indices = [ny // 3, 2 * ny // 3] if n_laterals >= 2 else [ny // 2]
    lat_k_index = nz // 2
    lat_i_max = int(L_ft / dx)

    # Create coordinate grid
    k_coords, j_coords, i_coords = np.mgrid[0:nz, 0:ny, 0:nx]
    
    # Simulate pressure drop around each lateral
    final_pressure = np.copy(p3d_psi)
    for j_lat in lat_j_indices:
        dist_sq = ((i_coords - lat_i_max/2)**2 * (dx/L_ft)**2 + 
                   (j_coords - j_lat)**2 * (dy/(xf_ft*2))**2 + 
                   (k_coords - lat_k_index)**2 * (dz/hf_ft)**2)
        
        pressure_drop = (p_init_psi - pwf_min_psi) * np.exp(-dist_sq * 5.0)
        # Apply the pressure drop, ensuring it doesn't overlap excessively
        final_pressure -= pressure_drop
    
    final_pressure = np.clip(final_pressure, pwf_min_psi, p_init_psi)
    
    # Create a time-series of 3D pressure for the material balance plot (simplified)
    # We'll just create a few keyframes and the app can animate them
    p_mid_series = []
    for frac in np.linspace(0, 1, len(t_days)): # This is an approximation
        p_intermediate = p_init_psi - (p_init_psi - final_pressure) * frac
        p_mid_series.append(p_intermediate[nz//2, :, :])

    # Mock other mid-layer results
    pf_mid_psi = p_mid_series[-1] - 150 # Frac pressure is lower than matrix
    pm_mid_psi = p_mid_series[-1]
    Sw_init = inputs.get('relperm', {}).get('Swc', 0.15)
    Sw_mid = np.full((ny, nx), Sw_init) # Placeholder, no water movement modelled here

    # --- 5. Assemble and return the results dictionary ---
    results = {
        't_days': t_days,
        'qg_Mscfd': qg_Mscfd,
        'qo_STBpd': qo_STBpd,
        'p3d_psi': final_pressure,
        'pm_mid_psi': p_mid_series, # Pass the whole series for QA plot
        'pf_mid_psi': pf_mid_psi,
        'Sw_mid': Sw_mid
    }
    
    print(f"--- Engine Proxy finished in {time.time() - start_time:.2f} seconds ---")
    
    return results
