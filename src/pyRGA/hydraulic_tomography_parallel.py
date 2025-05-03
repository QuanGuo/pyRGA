# hydrualic tomography parallel
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.sparse.linalg import cg
from .gwsolver_2D_steady_state import groundwater_solver
import time
import matplotlib.pyplot as plt

# Function to compute groundwater solution
def compute_head(well_loc, K, Q):
    return groundwater_solver(K, well_loc, Q=Q)

def hydraulic_tomography_joblib(K, well_locs, Q):
    """
    Perform hydraulic tomography using joblib for parallelization.
    """
    results = Parallel(n_jobs=8)(delayed(compute_head)(well_loc, K, Q) for well_loc in well_locs)
    return np.array(results)

def hydraulic_tomography_parallel(K, well_locs, Q):
    """
    Perform hydraulic tomography for multiple wells in parallel.

    Args:
        K (ndarray): Permeability field.
        well_locs (list): List of well locations (indices).
        Q (float): Pumping rate for all wells.

    Returns:
        ndarray: Hydraulic head solutions for all wells.
    """
    args = [(well_loc, K, Q) for well_loc in well_locs]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(compute_head_wrapper, args))
    return np.array(results)

def compute_head_wrapper(args):
    """
    Wrapper function to pass arguments to groundwater_solver.

    Args:
        args (tuple): Tuple containing (well_loc, K, Q).

    Returns:
        ndarray: Hydraulic head solution for a single well.
    """
    well_loc, K, Q = args
    return groundwater_solver(K, well_loc, Q)

def hydraulic_tomography_loop(K, well_locs, Q):
    """
    Perform hydraulic tomography for multiple wells.

    Args:
        K (ndarray): Permeability field.
        well_locs (list): List of well locations (indices).
        Q (float): Pumping rate for all wells.

    Returns:
        ndarray: Hydraulic head solutions for all wells.
    """
    
    numel = K.shape[0]
    nx = int(np.sqrt(numel))
    numnod = (nx+1)**2

    HT_heads = np.empty((len(well_locs), numnod), dtype=np.float64)

    for i, well_loc in enumerate(well_locs):
        HT_heads[i,:] = groundwater_solver(K, well_loc, Q=Q)
    return HT_heads


if __name__ == "__main__":

    # Define domain parameters
    nx, ny = 1024, 1024
    numel = nx * ny
    numnodx, numnody = nx + 1, ny + 1
    numnod = numnodx * numnody
    Lox, Loy = 320, 320
    dx, dy = Lox / nx, Loy / ny

    K = np.exp(np.random.randn(numel) * 0.1 - 2)  # Generate K without extra dimension

    K = K.reshape((nx, ny))
    K[nx//5:nx//3, ny//4:ny//4*3] = np.exp(0) 
    K[nx//3*2:nx//5*4, ny//4:ny//4*3] = np.exp(-4) 

    # Plot the solution
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(K, cmap='jet')
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title('Conductivity field')
    K = K.flatten()
    q_original = -0.02 * (64/nx)**2 # m3/s
    Q = q_original/dx/dy*3600 

    well_relative_locs = []
    horizontal_relative_locs = vertial_relative_locs = [0.25, 0.375, 0.5, 0.625, 0.75]
    for h in horizontal_relative_locs:
        for v in vertial_relative_locs:
            well_relative_locs.append((h,v))

    well_nodes = [int(x*numnodx)+int(y*numnody)*numnodx for x, y in well_relative_locs]

    t0 = time.time()
    HT_heads = hydraulic_tomography_joblib(K, well_nodes, Q)
    print("Elapsed time for parallel solving HT:", time.time() - t0)
