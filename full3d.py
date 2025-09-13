import numpy as np
import time

def simulate(inputs):
    """
    This function is a proxy for a full 3D implicit reservoir simulator engine.
    It uses a physics-informed analytical model to generate realistic production
    profiles and mock 3D pressure data based on the detailed inputs provided.
    """
    print("--- Running Full 3D Engine Proxy ---")
    start_time = time.time()

    # --- 1. Unpack key inputs ---
    grid = inputs.get('grid', {})
    nx, ny, nz = grid.get('nx', 100), grid.get('ny', 50), grid.get('nz', 10)
    dx, dy, dz = grid.get('dx', 50.), grid.get('dy', 50.), grid.get('dz', 20.)

    rock = inputs.get('rock', {})
    kx_array = rock.get('kx_md', np.full((nz, ny, nx), 0.05))
    phi_array = rock.get('phi', np.full((nz, ny, nx), 0.10))
    k_avg_md = np.mean(kx_array)

    init = inputs.get('init', {})
    p_init_psi = init.get('p_init_psi', 5000.)
    pwf_min_psi = init.get('pwf_min_psi', 2500.)
    Sw_init = init.get('Sw_init', 0.15)

    schedule = inputs.get('schedule', {})
    bhp_psi = schedule.get('bhp_psi', 2500.)
    
    msw = inputs.get('msw', {})
    L_ft = msw.get('L_ft', 10000.)
    xf_ft = msw.get('xf_ft', 300.)
    hf_ft = msw.get('hf_ft', 180.)
    n_laterals = msw.get('laterals', 2)

    pvt = inputs.get('pvt', {})
    Rs_pb_scf_stb = pvt.get('Rs_pb_scf_stb', 650.)
    Bo_pb_rb_stb = pvt.get('Bo_pb_rb_stb', 1.35)

    # --- 2. Calculate Decline Curve Parameters ---
    drawdown = p_init_psi - bhp_psi
    srv_proxy = (L_ft / 10000.) * (xf_ft / 300.) * (hf_ft / 180.) * n_laterals
    qi_oil_base = 1000 * srv_proxy * (k_avg_md / 0.05)**0.4 * (drawdown / 2500.)**0.8
    gor_init = max(1000., Rs_pb_scf_stb * (p_init_psi / pvt.get('pb_psi', 5200.))**0.5)
    qi_gas_base = qi_oil_base * gor_init / 1000.
    qi_oil = np.clip(qi_oil_base, 200, 3000)
    qi_gas = np.clip(qi_gas_base, 2000, 30000)
    Di_yr = 0.8 * (k_avg_md / 0.05)**0.3
    Di_day = Di_yr / 365.0
    b = 1.2
    
    # --- 3. Generate Production Forecast ---
    t_days = np.linspace(0, 30 * 365, 361)
    qo_STBpd = qi_oil / (1 + b * Di_day * t_days)**(1/b)
    qg_Mscfd = qi_gas / (1 + b * Di_day * t_days)**(1/b)
    qo_STBpd[qo_STBpd < 5] = 0
    qg_Mscfd[qg_Mscfd < 20] = 0

    # --- 4. Mock 3D Data for Visualization ---
    p_init_3d = np.full((nz, ny, nx), p_init_psi, dtype=np.float32)
    lat_j_indices = [ny // 3, 2 * ny // 3] if n_laterals >= 2 else [ny // 2]
    lat_k_index = nz // 2
    lat_i_max = int(L_ft / dx)
    k_coords, j_coords, i_coords = np.mgrid[0:nz, 0:ny, 0:nx]
    
    final_pressure = np.copy(p_init_3d)
    for j_lat in lat_j_indices:
        dist_sq = ((i_coords - lat_i_max/2)**2 * (dx/L_ft)**2 + 
                   (j_coords - j_lat)**2 * (dy/(xf_ft*2))**2 + 
                   (k_coords - lat_k_index)**2 * (dz/hf_ft)**2)
        pressure_drop = (p_init_psi - pwf_min_psi) * np.exp(-dist_sq * 5.0)
        final_pressure -= pressure_drop
    
    final_pressure = np.clip(final_pressure, pwf_min_psi, p_init_psi)
    
    # Calculate OOIP volume
    cell_volume_bbl = (dx * dy * dz) / 5.615  # 5.615 ft^3 per bbl
    Boi = pvt.get('Bo_pb_rb_stb', 1.35) # Approximation using Bo at bubble point
    ooip_3d = (phi_array * (1 - Sw_init) * cell_volume_bbl) / Boi

    # Create time-series of pressure for QA plot
    p_mid_series = []
    for frac in np.linspace(0, 1, len(t_days)):
        p_intermediate = p_init_psi - (p_init_psi - final_pressure) * frac
        p_mid_series.append(p_intermediate[nz//2, :, :])

    # --- 5. Assemble and return results ---
    results = {
        't_days': t_days, 'qg_Mscfd': qg_Mscfd, 'qo_STBpd': qo_STBpd,
        'p3d_psi': final_pressure,
        'p_init_3d': p_init_3d,   # NEW: Return initial pressure field
        'ooip_3d': ooip_3d,       # NEW: Return OOIP volume
        'pm_mid_psi': p_mid_series,
        'pf_mid_psi': p_mid_series[-1] - 150,
        'Sw_mid': np.full((ny, nx), Sw_init)
    }
    
    print(f"--- Engine Proxy finished in {time.time() - start_time:.2f} seconds ---")
    return results
