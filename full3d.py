import numpy as np
import time
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

# --- The Grid Class (from Step 1) ---
class Grid:
    def __init__(self, inputs):
        grid_params = inputs.get('grid', {})
        self.nx = grid_params.get('nx')
        self.ny = grid_params.get('ny')
        self.nz = grid_params.get('nz')
        self.dx = grid_params.get('dx')
        self.dy = grid_params.get('dy')
        self.dz = grid_params.get('dz')
        self.num_cells = self.nx * self.ny * self.nz

    def get_idx(self, i, j, k):
        if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
            return k * (self.nx * self.ny) + j * self.nx + i
        return -1

    def cell_volume_ft3(self):
        return self.dx * self.dy * self.dz

# --- The Rock Class (from Step 2) ---
class Rock:
    def __init__(self, inputs, grid):
        rock_params = inputs.get('rock', {})
        self.poro = rock_params.get('phi', np.full(grid.num_cells, 0.1)).flatten()
        self.kx = rock_params.get('kx_md', np.full(grid.num_cells, 0.05)).flatten()
        self.ky = rock_params.get('ky_md', np.full(grid.num_cells, 0.05)).flatten()
        # For simplicity, we assume kz = 0.1 * kx
        self.kz = self.kx * 0.1

# --- The Fluid Class (from Step 2) ---
class Fluid:
    def __init__(self, inputs):
        pvt_params = inputs.get('pvt', {})
        self.pb_psi = pvt_params.get('pb_psi')
        self.Rs_pb = pvt_params.get('Rs_pb_scf_stb')
        self.Bo_pb = pvt_params.get('Bo_pb_rb_stb')
        self.muo_pb = pvt_params.get('muo_pb_cp')
        self.mug_pb = pvt_params.get('mug_pb_cp')

    def Bo(self, P):
        p = np.asarray(P, float)
        slope = -1.0e-5
        return np.where(p <= self.pb_psi, self.Bo_pb, self.Bo_pb + slope*(p - self.pb_psi))

    def Rs(self, P):
        p = np.asarray(P, float)
        return np.where(p <= self.pb_psi, self.Rs_pb, self.Rs_pb + 0.00012*(p - self.pb_psi)**1.1)

# --- NEW: The Transmissibility Class ---
class Transmissibility:
    def __init__(self, grid, rock):
        """
        Pre-calculates the geometric part of transmissibility between all cells.
        This is a performance optimization. The formula is T = (kA / L), where
        L is the distance between cell centers. We use a harmonic average for k.
        """
        self.T_x = np.zeros(grid.num_cells)
        self.T_y = np.zeros(grid.num_cells)
        self.T_z = np.zeros(grid.num_cells)
        
        conversion_factor = 0.001127 # Darcy units conversion for oilfield units
        
        # Loop through all interior cells
        for k in range(grid.nz):
            for j in range(grid.ny):
                for i in range(grid.nx):
                    m = grid.get_idx(i, j, k)
                    
                    # Transmissibility in X-direction (connection i to i+1)
                    if i < grid.nx - 1:
                        m_plus_1 = grid.get_idx(i + 1, j, k)
                        k_harmonic_avg = 2 * rock.kx[m] * rock.kx[m_plus_1] / (rock.kx[m] + rock.kx[m_plus_1])
                        area = grid.dy * grid.dz
                        self.T_x[m] = conversion_factor * k_harmonic_avg * area / grid.dx
                    
                    # Transmissibility in Y-direction (connection j to j+1)
                    if j < grid.ny - 1:
                        m_plus_1 = grid.get_idx(i, j + 1, k)
                        k_harmonic_avg = 2 * rock.ky[m] * rock.ky[m_plus_1] / (rock.ky[m] + rock.ky[m_plus_1])
                        area = grid.dx * grid.dz
                        self.T_y[m] = conversion_factor * k_harmonic_avg * area / grid.dy
                        
                    # Transmissibility in Z-direction (connection k to k+1)
                    if k < grid.nz - 1:
                        m_plus_1 = grid.get_idx(i, j, k + 1)
                        k_harmonic_avg = 2 * rock.kz[m] * rock.kz[m_plus_1] / (rock.kz[m] + rock.kz[m_plus_1])
                        area = grid.dx * grid.dy
                        self.T_z[m] = conversion_factor * k_harmonic_avg * area / grid.dz

def simulate(inputs):
    return simulate_3D_implicit(inputs)

def simulate_3D_implicit(inputs):
    """
    PHASE 1b BLUEPRINT: 3D, THREE-PHASE IMPLICIT SIMULATOR
    """
    print("--- Running Phase 1b: 3D Three-Phase Implicit Engine Blueprint ---")
    start_time = time.time()

    # --- 1. Initialize Core Components ---
    grid = Grid(inputs)
    rock = Rock(inputs, grid)
    fluid = Fluid(inputs)
    trans = Transmissibility(grid, rock)
    print("Grid, Rock, Fluid, and Transmissibility components initialized.")

    # --- 2. Initialize State Variables ---
    unknowns = np.zeros(grid.num_cells * 3) 
    
    # --- 3. Setup Timesteps ---
    total_time_days = 30 * 365
    num_timesteps = 360
    dt_days = total_time_days / num_timesteps
    t_days = np.linspace(0, total_time_days, num_timesteps + 1)
    
    # --- 4. Main Time-Stepping Loop ---
    for step in range(num_timesteps):
        
        # --- 5. Newton-Raphson Iteration Loop ---
        # PSEUDOCODE:
        # for newton_iter in range(max_newton_iter):
        #     # You would now use the pre-calculated transmissibilities inside
        #     # the residual and jacobian calculations. For example, the flow
        #     # between cell m and m+1 in the x-direction would be:
        #     # flow = trans.T_x[m] * mobility * (P[m] - P[m+1])
        #
        #     R = calculate_residuals(unknowns, old_unknowns, grid, rock, fluid, trans, dt_days)
        #     J = assemble_jacobian(unknowns, grid, rock, fluid, trans, dt_days)
        #     dx = solve_linear_system(J, -R)
        #     unknowns += dx
        #     if converged: break
        pass

    # --- 6. Post-Process and Return Results (Placeholder) ---
    qo_STBpd = 1000 * np.exp(-t_days[1:] / 500)
    qg_Mscfd = 8000 * np.exp(-t_days[1:] / 600)
    results = {
        't_days': t_days, 
        'qg_Mscfd': np.insert(qg_Mscfd, 0, 0), 
        'qo_STBpd': np.insert(qo_STBpd, 0, 0),
    }
    
    print(f"--- 3D Implicit Engine Blueprint finished in {time.time() - start_time:.2f} seconds ---")
    return results
