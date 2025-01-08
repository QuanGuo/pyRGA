import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pyamg
from scipy.sparse.linalg import cg
from concurrent.futures import ProcessPoolExecutor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import ceil
import time
from joblib import Parallel, delayed
import numpy as np

# Function to compute groundwater solution
def compute_head(well_loc, K, Q):
    return groundwater_solver(K, well_loc, Q=Q)

def hydraulic_tomography_joblib(K, well_locs, Q):
    """
    Perform hydraulic tomography using joblib for parallelization.
    """
    results = Parallel(n_jobs=8)(delayed(compute_head)(well_loc, K, Q) for well_loc in well_locs)
    return np.array(results)

def apply_dirichlet_conditions(bigk, force, dirichlet_nodes, dirichlet_values):
    """
    Apply Dirichlet boundary conditions to the stiffness matrix and force vector.

    Args:
        bigk (scipy.sparse.csr_matrix): Global stiffness matrix.
        force (ndarray): Force vector.
        dirichlet_nodes (ndarray): Indices of Dirichlet nodes.
        dirichlet_values (ndarray): Values to impose at the Dirichlet nodes.

    Returns:
        tuple:
            - bigk (scipy.sparse.csr_matrix): Updated stiffness matrix.
            - force (ndarray): Updated force vector with Dirichlet contributions applied.
    """
    dirichlet_contributions = bigk[:, dirichlet_nodes].dot(dirichlet_values)
    force -= dirichlet_contributions
    return bigk, force


def assemble_matrix(numnodx, numnody,   stiffness, K):
    """
    Assemble the global stiffness matrix for the finite element model.

    Args:
        numnodx (int): Number of nodes in the x direction.
        numnody (int): Number of nodes in the y direction.
        stiffness (ndarray): Local element stiffness matrix.
        K (ndarray): Permeability field.

    Returns:
        scipy.sparse.csr_matrix: Assembled global stiffness matrix.
    """
    numel = (numnodx - 1) * (numnody - 1)
    numnod = numnodx * numnody

    quotient, remainder = divmod(np.arange(numel), numnodx - 1)
    connect_mat = np.column_stack((
        remainder + quotient * numnodx,
        remainder + quotient * numnodx + 1,
        remainder + quotient * numnodx + numnodx,
        remainder + quotient * numnodx + numnodx + 1
    ))

    sctr_rows = connect_mat.repeat(4, axis=1).flatten()
    sctr_cols = np.tile(connect_mat, 4).flatten()
    ke_values = (stiffness * K[:, None, None]).reshape(numel, -1).flatten()
    bigk = sp.coo_matrix((ke_values, (sctr_rows, sctr_cols)), shape=(numnod, numnod)).tocsr()

    return bigk


def groundwater_solver(K, well_node, Q):
    """
    Solve the groundwater flow problem for a single well.

    Args:
        K (ndarray): Permeability field.
        well_node (int): Index of the well location in the grid.
        Q (float): Pumping rate at the well.

    Returns:
        ndarray: Hydraulic head solution for the domain.
    """
    numel = K.shape[0]
    nx = ny = int(np.sqrt(numel))
    numnodx = numnody = nx + 1
    numnod = numnodx * numnody

    stiffness = np.array([
        [0.66666667, -0.16666667, -0.16666667, -0.33333333],
        [-0.16666667, 0.66666667, -0.33333333, -0.16666667],
        [-0.16666667, -0.33333333, 0.66666667, -0.16666667],
        [-0.33333333, -0.16666667, -0.16666667, 0.66666667]
    ])

    left_boundary = np.arange(numnody) * numnodx
    right_boundary = left_boundary + (numnodx - 1)
    dirichlet_nodes = np.concatenate((left_boundary, right_boundary))

    solution_full = np.zeros(numnod)
    solution_full[left_boundary] = 0.0
    solution_full[right_boundary] = 0.0
    dirichlet_values = solution_full[dirichlet_nodes]

    bigk = assemble_matrix(numnodx, numnody, stiffness, K)
    force = np.zeros(numnod)
    force[well_node] = Q

    bigk, force = apply_dirichlet_conditions(bigk, force, dirichlet_nodes, dirichlet_values)

    mask = np.ones(numnod, dtype=bool)
    mask[dirichlet_nodes] = False

    bigk_reduced = bigk[mask, :][:, mask]
    force_reduced = force[mask]

    ml = pyamg.ruge_stuben_solver(bigk_reduced)
    solution_full[mask], _ = cg(bigk_reduced, force_reduced, M=ml.aspreconditioner())

    return solution_full


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
        # (lambda x: x**2, arr)
# map(func, *iterables) --> map object
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


def hydraulic_tomography(K, well_locs, Q):
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
    HT_heads = hydraulic_tomography(K, well_nodes, Q)
    print("Elapsed time for solving HT:", time.time() - t0)

    # t0 = time.time()
    # HT_heads = hydraulic_tomography_joblib(K, well_nodes, Q)
    # print("Elapsed time for parallel solving HT:", time.time() - t0)

    # Plot the solution
    fig, axs = plt.subplots(
        len(horizontal_relative_locs), 
        len(vertial_relative_locs), 
        figsize=(3* len(horizontal_relative_locs), 2.8 * len(vertial_relative_locs)),
        gridspec_kw={'hspace': 0.5, 'wspace': 0.5}
    )
    # fig.subplots_adjust(hspace=0.4, wspace=1)  # Adjust vertical and horizontal spacing
    axs = axs.flatten()
    for i, w in enumerate(well_relative_locs):
        head_solved = HT_heads[i].reshape((numnodx, numnody))
        im = axs[i].pcolormesh(head_solved, cmap='viridis', shading='auto')
        CT = axs[i].contour(head_solved, levels=10, colors='white')
        axs[i].clabel(CT, fontsize=10, inline=True, fmt='%.1f')
        axs[i].set_title('Pump at ({:d} m, {:d} m)'.format(int(w[0] * Lox), int(w[1] * Loy)))

        # Set ticks for the axes
        x_ticks = np.linspace(0, numnodx - 1, 5)
        x_labels = np.linspace(0, Lox, 5, dtype=int)
        axs[i].set_xticks(x_ticks)
        axs[i].set_xticklabels(x_labels)
        axs[i].set_xlabel('X (m)')

        y_ticks = np.linspace(0, numnody - 1, 5)
        y_labels = np.linspace(0, Loy, 5, dtype=int)
        axs[i].set_yticks(y_ticks)
        axs[i].set_yticklabels(y_labels)
        axs[i].set_ylabel('Y (m)')
        axs[i].tick_params(axis='y', rotation=90)

        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="7%", pad=0.05)  # Adjust size and padding
        cbar = plt.colorbar(im, cax=cax, ticks=[ceil(head_solved.min()), int(head_solved.max())])
        cbar.ax.tick_params(labelsize=8, rotation=270)  # Reduce tick font size
        cbar.set_label('Hydraulic Head (m)', fontsize=10, va="top", rotation=270)  # Adjust label padding

    plt.show()