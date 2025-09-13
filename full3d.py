import numpy as np
import time
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

def simulate(inputs):
    """
    PHASE 1a: 1D IMPLICIT FINITE-DIFFERENCE SIMULATOR
    This engine solves the single-phase diffusivity equation in 1D using an
    implicit finite-difference scheme. It models linear flow from the matrix
    to a fracture face and scales the rate for the entire well.
    """
    print("--- Running Phase 1a: 1D Implicit Finite-Difference Engine ---")
    start_time = time.time()

    # --- 1. Unpack key inputs ---
    grid = inputs.get('grid', {})
    nx, ny, nz = grid.get('nx', 300), grid.get('ny', 60), grid.get('nz', 12)
    dx, dy, dz = grid.get('dx', 40.), grid.get('dy', 40.), grid.get('dz', 15.)

    rock = inputs.get('rock', {})
    kx_array = rock.get('kx_md', np.full((nz, ny, nx), 0.05))
    phi_array = rock.get('phi', np.full((nz, ny, nx), 0.10))
    k_avg_md = np.mean(kx_array)
    phi_avg = np.mean(phi_array)

    init = inputs.get('init', {})
    p_init_psi = init.get('p_init_psi', 5800.)
    Sw_init = init.get('Sw_init', 0.15)
    
    schedule = inputs.get('schedule', {})
    bhp_psi = schedule.get('bhp_psi', 2500.)

    msw = inputs.get('msw', {})
    L_ft = msw.get('L_ft', 10000.)
    ss_ft = msw.get('stage_spacing_ft', 250.)
    hf_ft = msw.get('hf_ft', 180.)

    pvt = inputs.get('pvt', {})
    mu_oil_cp = pvt.get('muo_pb_cp', 1.2)
    ct_1psi = pvt.get('ct_o_1psi', 8e-6) # Using oil compressibility as a proxy for total

    # --- 2. Setup the 1D Simulation Grid ---
    # We model flow from the midpoint between fractures into a single fracture face.
    num_blocks = 20  # Number of grid blocks in our 1D model
    L_1d = ss_ft / 2.0  # The length of our 1D model is half the stage spacing
    dx_1d = L_1d / num_blocks
    
    # --- 3. Define Timesteps ---
    total_time_days = 30 * 365
    num_timesteps = 360
    dt_days = total_time_days / num_timesteps
    t_days = np.linspace(0, total_time_days, num_timesteps + 1)
    
    # Conversion factors for Darcy units
    k_darcy = k_avg_md / 1000.0
    alpha = 0.006328 # Conversion factor for oilfield units to Darcy

    # --- 4. Initialize Pressure and System Matrices ---
    P = np.full(num_blocks, p_init_psi)
    P_new = np.copy(P)
    
    # We use a sparse matrix for efficiency
    A = lil_matrix((num_blocks, num_blocks))
    b = np.zeros(num_blocks)

    # Calculate constant terms
    fracture_face_area = hf_ft * dx # This is a key assumption for scaling
    transmissibility_base = k_darcy * fracture_face_area / mu_oil_cp / dx_1d
    accumulation_base = (dx_1d * fracture_face_area * phi_avg * ct_1psi) / (alpha * dt_days)

    # --- 5. Assemble the Matrix (Ax = b) ---
    # This matrix is tridiagonal and constant for this simple problem.
    for i in range(num_blocks):
        # Accumulation term on the main diagonal
        A[i, i] = accumulation_base
        # Flow term to the left
        if i > 0:
            A[i, i] += transmissibility_base
            A[i, i - 1] = -transmissibility_base
        # Flow term to the right
        if i < num_blocks - 1:
            A[i, i] += transmissibility_base
            A[i, i + 1] = -transmissibility_base
            
    # --- 6. Apply Boundary Conditions ---
    # Left boundary (i=0): Constant pressure at the fracture face (BHP)
    A[0, 0] += transmissibility_base 
    
    # Right boundary (i=num_blocks-1): No-flow boundary (midpoint between fractures)
    # The default matrix assembly already handles this correctly.

    # Convert to a more efficient format for solving
    A = csc_matrix(A)

    # --- 7. Time-Stepping Loop ---
    qo_stbd_list = []
    pressure_profiles = []
    for step in range(num_timesteps):
        # Update the right-hand-side vector 'b'
        b = accumulation_base * P
        # Apply the constant pressure boundary condition to 'b'
        b[0] += transmissibility_base * 2 * bhp_psi 

        # Solve the linear system Ax = b for the new pressure P_new
        P_new = spsolve(A, b)
        
        # Calculate flow rate from the first block into the fracture face (well)
        # q = T * (P_block1 - P_well)
        q_1d = (transmissibility_base / alpha) * (P_new[0] - bhp_psi)
        
        # Scale up the rate for the entire well
        num_stages = L_ft / ss_ft
        num_frac_faces = num_stages * 2 # Two faces per fracture
        total_q = q_1d * num_frac_faces
        qo_stbd_list.append(total_q)
        
        # Save current pressure profile and update for next step
        pressure_profiles.append(np.copy(P_new))
        P = P_new

    # --- 8. Post-Process for App Compatibility ---
    qo_STBpd = np.array([0] + qo_stbd_list) # Add initial rate of 0
    # For this single-phase model, we'll just mock a GOR
    gor = Rs_pb_scf_stb * 2.0
    qg_Mscfd = qo_STBpd * gor / 1000.0
    
    # Create a mock 3D pressure field by extruding the final 1D profile
    final_1d_profile = pressure_profiles[-1]
    p3d_psi = np.full((nz, ny, nx), p_init_psi)
    # Replicate the 1D pressure profile along the Y-axis to simulate depletion
    for j in range(ny):
        dist_from_frac_center = abs(j - ny//2) * dy
        pressure_index = min(num_blocks - 1, int(dist_from_frac_center / dx_1d))
        p3d_psi[:, j, :int(L_ft / dx)] = final_1d_profile[pressure_index]
    
    results = {
        't_days': t_days, 'qg_Mscfd': qg_Mscfd, 'qo_STBpd': qo_STBpd,
        'p3d_psi': p3d_psi,
        # Other results can be mocked or derived as before
        'p_init_3d': np.full_like(p3d_psi, p_init_psi),
        'ooip_3d': np.zeros_like(p3d_psi),
        'pm_mid_psi': [p[nz//2, :, :] for p in np.linspace(np.full_like(p3d_psi, p_init_psi), p3d_psi, len(t_days))],
    }
    
    print(f"--- 1D Implicit Engine finished in {time.time() - start_time:.2f} seconds ---")
    return results
