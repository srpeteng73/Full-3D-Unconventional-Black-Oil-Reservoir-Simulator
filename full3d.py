# full3d.py (Functional placeholder version)
import numpy as np

def simulate(inputs, progress=None):
    """
    This is a functional placeholder for your real 3D engine.
    It accepts the input dictionary and returns data in the correct format,
    preventing the app from crashing.
    """
    # Get grid dimensions from the input
    nx = inputs.get('grid', {}).get('nx', 10)
    ny = inputs.get('grid', {}).get('ny', 10)
    nz = inputs.get('grid', {}).get('nz', 5)
    p_init = inputs.get('init', {}).get('p_init_psi', 5000)

    # Create dummy time-series data
    t_days = np.linspace(0, 365 * 10, 120)  # 10 years
    qg_Mscfd = 10000 * np.exp(-t_days / 1000) + 50 * np.random.randn(len(t_days))
    qo_STBpd = 800 * np.exp(-t_days / 1200) + 10 * np.random.randn(len(t_days))

    # Create dummy 3D pressure data
    p3d_psi = np.full((nz, ny, nx), p_init, dtype=float)
    # Simulate some drawdown in the center
    center_k, center_j, center_i = nz // 2, ny // 2, nx // 2
    p3d_psi[center_k, center_j, center_i] = p_init - 2000

    # Create dummy 2D mid-layer data
    pf_mid_psi = np.full((ny, nx), p_init - 500, dtype=float)
    pm_mid_psi = np.full((ny, nx), p_init - 200, dtype=float)
    Sw_mid = np.full((ny, nx), 0.25, dtype=float)

    # This is the dictionary the app expects to get back
    results = {
        't_days': t_days,
        'qg_Mscfd': qg_Mscfd,
        'qo_STBpd': qo_STBpd,
        'p3d_psi': p3d_psi,
        'pf_mid_psi': pf_mid_psi,
        'pm_mid_psi': pm_mid_psi,
        'Sw_mid': Sw_mid
    }
    
    return results
