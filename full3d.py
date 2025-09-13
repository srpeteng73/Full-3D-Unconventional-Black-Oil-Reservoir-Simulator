import numpy as np
import time
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

# --- The Grid Class ---
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

    def cell_volume_bbl(self):
        return (self.dx * self.dy * self.dz) / 5.615

# --- The Rock Class ---
class Rock:
    def __init__(self, inputs, grid):
        rock_params = inputs.get('rock', {})
        self.poro = rock_params.get('phi', np.full(grid.num_cells, 0.1)).flatten()
        self.kx = rock_params.get('kx_md', np.full(grid.num_cells, 0.05)).flatten()
        self.ky = rock_params.get('ky_md', np.full(grid.num_cells, 0.05)).flatten()
        self.kz = self.kx * 0.1

# --- The Fluid Class ---
class Fluid:
    def __init__(self, inputs):
        pvt_params = inputs.get('pvt', {})
        self.pb_psi = pvt_params.get('pb_psi')
        self.Bo_pb = pvt_params.get('Bo_pb_rb_stb')
        self.muo_pb = pvt_params.get('muo_pb_cp')
        self.ct_oil = pvt_params.get('ct_o_1psi', 8e-6)

    def Bo(self, P):
        # A simple oil FVF model
        return self.Bo_pb * (1.0 - self.ct_oil * (P - self.pb_psi))
    
    def mu_oil(self, P):
        return np.full_like(P, self.muo_pb)

# --- The Transmissibility Class ---
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
                    # ... Y and Z calculations would go here ...

# --- The State Class ---
class State:
    def __init__(self, inputs, grid, fluid):
        self.grid = grid
        self.fluid = fluid
        init_params = inputs.get('init', {})
        self.pressure = np.full(grid.num_cells, init_params.get('p_init_psi'))
        self.update_properties()
    
    def update_properties(self):
        self.Bo = self.fluid.Bo(self.pressure)
        self.mu_oil = self.fluid.mu_oil(self.pressure)
        self.oil_mobility = (1.0 / self.mu_oil) / self.Bo

# --- The Well Class ---
class Well:
    def __init__(self, inputs, grid):
        msw_params = inputs.get('msw', {})
        schedule_params = inputs.get('schedule', {})
        self.bhp = schedule_params.get('bhp_psi')
        self.perforations = []
        # For simplicity, perforate the first few blocks of the middle row
        for i in range(int(msw_params.get('L_ft') / grid.dx)):
            self.perforations.append(grid.get_idx(i, grid.ny // 2, grid.nz // 2))
        
        # Well Index (Productivity Index) - a simple model
        self.wi = 5.0 # bbl/day/psi

# --- NEW: Jacobian Assembly Function ---
def assemble_jacobian_and_residuals(state_new, state_old, grid, rock, fluid, trans, well, dt_days):
    num_cells = grid.num_cells
    J = lil_matrix((num_cells, num_cells))
    R = np.zeros(num_cells)
    
    Vp_bbl = grid.cell_volume_bbl() * rock.poro
    
    for m in range(num_cells):
        # Accumulation Term
        accum_new = Vp_bbl[m] / (state_new.Bo[m] * dt_days)
        accum_old = Vp_bbl[m] / (state_old.Bo[m] * dt_days) * state_old.pressure[m]
        R[m] += accum_new * state_new.pressure[m] - accum_old
        J[m, m] += accum_new # d(Accum)/dP_m

        # Flow Terms (X-direction for now)
        # Connection to the left (m-1)
        if m > 0 and (m % grid.nx) != 0:
            m_minus_1 = m - 1
            mob = state_new.oil_mobility[m_minus_1] if state_new.pressure[m_minus_1] > state_new.pressure[m] else state_new.oil_mobility[m]
            flow = trans.T_x[m_minus_1] * mob * (state_new.pressure[m_minus_1] - state_new.pressure[m])
            R[m] -= flow
            J[m, m] += trans.T_x[m_minus_1] * mob # d(Flow)/dP_m
            J[m, m_minus_1] -= trans.T_x[m_minus_1] * mob # d(Flow)/dP_m-1
        # Connection to the right (m+1)
        if m < num_cells - 1 and ((m + 1) % grid.nx) != 0:
            m_plus_1 = m + 1
            mob = state_new.oil_mobility[m] if state_new.pressure[m] > state_new.pressure[m_plus_1] else state_new.oil_mobility[m_plus_1]
            flow = trans.T_x[m] * mob * (state_new.pressure[m_plus_1] - state_new.pressure[m])
            R[m] -= flow
            J[m, m] += trans.T_x[m] * mob # d(Flow)/dP_m
            J[m, m_plus_1] -= trans.T_x[m] * mob # d(Flow)/dP_m+1
    
    # Well Terms
    for perf_idx in well.perforations:
        well_flow = well.wi * state_new.oil_mobility[perf_idx] * (state_new.pressure[perf_idx] - well.bhp)
        R[perf_idx] += well_flow
        J[perf_idx, perf_idx] += well.wi * state_new.oil_mobility[perf_idx]

    return csc_matrix(J), R

def simulate(inputs):
    return simulate_3D_implicit(inputs)

def simulate_3D_implicit(inputs):
    print("--- Running Phase 1b: 3D Single-Phase Implicit Engine ---")
    start_time = time.time()

    # --- 1. Initialize Components ---
    grid = Grid(inputs)
    rock = Rock(inputs, grid)
    fluid = Fluid(inputs)
    trans = Transmissibility(grid, rock)
    well = Well(inputs, grid)
    
    # --- 2. Initialize State ---
    state_current = State(inputs, grid, fluid)
    
    # --- 3. Setup Timesteps and Output Arrays ---
    total_time_days = 30 * 365
    num_timesteps = 360
    dt_days = total_time_days / num_timesteps
    t_days = np.linspace(0, total_time_days, num_timesteps + 1)
    qo_stbd_list = np.zeros(num_timesteps)

    # --- 4. Main Time-Stepping Loop ---
    for step in range(num_timesteps):
        state_old = state_current
        state_new_guess = state_current # Start Newton iter with previous state
        
        # --- 5. Newton-Raphson Iteration Loop ---
        for newton_iter in range(10):
            # Update fluid properties for the current guess
            state_new_guess.update_properties()
            
            # Assemble Jacobian and Residuals
            J, R = assemble_jacobian_and_residuals(state_new_guess, state_old, grid, rock, fluid, trans, well, dt_days)
            
            # Solve the linear system for the update vector
            dx = spsolve(J, -R)
            
            # Update the solution
            state_new_guess.pressure += dx
            
            # Check for convergence
            if np.linalg.norm(dx) < 1e-3:
                break
        
        # --- 6. End of Timestep ---
        state_current = state_new_guess
        
        # Calculate and store total well rate
        total_q = 0
        for perf_idx in well.perforations:
            total_q += well.wi * state_current.oil_mobility[perf_idx] * (state_current.pressure[perf_idx] - well.bhp)
        qo_stbd_list[step] = total_q

    # --- 7. Post-Process and Return Results ---
    qo_STBpd = np.insert(qo_stbd_list, 0, 0)
    gor = fluid.Rs(state_current.pressure[0]) # Mock GOR
    qg_Mscfd = qo_STBpd * gor / 1000.0
    
    results = {
        't_days': t_days, 
        'qg_Mscfd': qg_Mscfd, 
        'qo_STBpd': qo_STBpd,
        'p3d_psi': state_current.pressure.reshape((grid.nz, grid.ny, grid.nx))
    }
    
    print(f"--- 3D Implicit Engine finished in {time.time() - start_time:.2f} seconds ---")
    return results
