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
        self.kz = self.kx * 0.1

# --- The Fluid Class (from Step 2) ---
class Fluid:
    def __init__(self, inputs):
        pvt_params = inputs.get('pvt', {})
        self.pb_psi = pvt_params.get('pb_psi')
        self.Rs_pb = pvt_params.get('Rs_pb_scf_stb')
        self.Bo_pb = pvt_params.get('Bo_pb_rb_stb')
        self.muo_pb = pvt_params.get('muo_pb_cp')

    def Bo(self, P):
        p = np.asarray(P, float)
        slope = -1.0e-5
        return np.where(p <= self.pb_psi, self.Bo_pb, self.Bo_pb + slope*(p - self.pb_psi))
    
    def mu_oil(self, P):
        # For simplicity, assume constant oil viscosity for now
        return np.full_like(P, self.muo_pb)

# --- The Transmissibility Class (from Step 3) ---
class Transmissibility:
    def __init__(self, grid, rock):
        self.T_x = np.zeros(grid.num_cells)
        self.T_y = np.zeros(grid.num_cells)
        self.T_z = np.zeros(grid.num_cells)
        conversion_factor = 0.001127
        for k in range(grid.nz):
            for j in range(grid.ny):
                for i in range(grid.nx):
                    m = grid.get_idx(i, j, k)
                    if i < grid.nx - 1:
                        m_plus_1 = grid.get_idx(i + 1, j, k)
                        k_h = 2 * rock.kx[m] * rock.kx[m_plus_1] / (rock.kx[m] + rock.kx[m_plus_1])
                        self.T_x[m] = conversion_factor * k_h * grid.dy * grid.dz / grid.dx
                    if j < grid.ny - 1:
                        m_plus_1 = grid.get_idx(i, j + 1, k)
                        k_h = 2 * rock.ky[m] * rock.ky[m_plus_1] / (rock.ky[m] + rock.ky[m_plus_1])
                        self.T_y[m] = conversion_factor * k_h * grid.dx * grid.dz / grid.dy
                    if k < grid.nz - 1:
                        m_plus_1 = grid.get_idx(i, j, k + 1)
                        k_h = 2 * rock.kz[m] * rock.kz[m_plus_1] / (rock.kz[m] + rock.kz[m_plus_1])
                        self.T_z[m] = conversion_factor * k_h * grid.dx * grid.dy / grid.dz

# --- NEW: The State Class ---
class State:
    def __init__(self, inputs, grid, fluid):
        self.grid = grid
        self.fluid = fluid
        init_params = inputs.get('init', {})
        self.pressure = np.full(grid.num_cells, init_params.get('p_init_psi'))
        self.Sw = np.full(grid.num_cells, init_params.get('Sw_init'))
        self.Sg = np.zeros(grid.num_cells)
    
    def update_properties(self):
        """Calculates all pressure-dependent fluid properties for the current state."""
        self.Bo = self.fluid.Bo(self.pressure)
        self.mu_oil = self.fluid.mu_oil(self.pressure)
        # For single phase oil, relative permeability is 1 and mobility is simple
        self.oil_mobility = (1.0 / self.mu_oil) / self.Bo

# --- NEW: The Residual Calculation Function ---
def calculate_residuals(state_new, state_old, grid, rock, fluid, trans, dt_days):
    """
    Calculates the mass balance error (residual) for each cell for the oil phase.
    Residual = Accumulation - Flow_In + Flow_Out - Well_Term
    """
    residuals = np.zeros(grid.num_cells)
    
    # Calculate Accumulation term for all cells at once (vectorized)
    # Accumulation = (Vp/dt) * d(So/Bo)
    Vp = grid.cell_volume_ft3() * rock.poro / 5.615 # Pore volume in bbls
    accumulation = (Vp / dt_days) * ( (1-state_new.Sw)/state_new.Bo - (1-state_old.Sw)/state_old.Bo )
    residuals += accumulation

    # Loop through cells to calculate flow terms (this is where the work is)
    for k in range(grid.nz):
        for j in range(grid.ny):
            for i in range(grid.nx):
                m = grid.get_idx(i, j, k)
                
                # Flow in X-direction
                if i > 0:
                    m_minus_1 = grid.get_idx(i-1, j, k)
                    # Upstream mobility weighting
                    mob = state_new.oil_mobility[m_minus_1] if state_new.pressure[m_minus_1] > state_new.pressure[m] else state_new.oil_mobility[m]
                    flow = trans.T_x[m_minus_1] * mob * (state_new.pressure[m_minus_1] - state_new.pressure[m])
                    residuals[m] -= flow
                if i < grid.nx - 1:
                    m_plus_1 = grid.get_idx(i+1, j, k)
                    mob = state_new.oil_mobility[m] if state_new.pressure[m] > state_new.pressure[m_plus_1] else state_new.oil_mobility[m_plus_1]
                    flow = trans.T_x[m] * mob * (state_new.pressure[m+1] - state_new.pressure[m])
                    residuals[m] -= flow
                
                # TODO: Add flow terms for Y and Z directions similarly...
    
    # TODO: Add well terms for perforated cells...
    
    return residuals


def simulate(inputs):
    return simulate_3D_implicit(inputs)

def simulate_3D_implicit(inputs):
    """
    PHASE 1b BLUEPRINT: 3D, SINGLE-PHASE IMPLICIT SIMULATOR
    """
    print("--- Running Phase 1b: 3D Implicit Engine Blueprint ---")
    start_time = time.time()

    # --- 1. Initialize Core Components ---
    grid = Grid(inputs)
    rock = Rock(inputs, grid)
    fluid = Fluid(inputs)
    trans = Transmissibility(grid, rock)
    print("Grid, Rock, Fluid, and Transmissibility components initialized.")

    # --- 2. Initialize State Variables ---
    state_current = State(inputs, grid, fluid)
    state_current.update_properties()
    
    # --- 3. Setup Timesteps ---
    total_time_days = 30 * 365
    num_timesteps = 360
    dt_days = total_time_days / num_timesteps
    t_days = np.linspace(0, total_time_days, num_timesteps + 1)
    
    # --- 4. Main Time-Stepping Loop ---
    for step in range(num_timesteps):
        state_old = state_current
        
        # --- 5. Newton-Raphson Iteration Loop ---
        # for newton_iter in range(max_newton_iter):
        #     # Calculate residuals based on the current guess for the new state
        #     residuals = calculate_residuals(state_new_guess, state_old, grid, rock, fluid, trans, dt_days)
        #     # ... (rest of the solver) ...
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
