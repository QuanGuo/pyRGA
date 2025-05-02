"""
2D Steady-State Groundwater Flow Solver Module.
This module implements finite element methods to solve steady-state groundwater flow equations.
"""

import numpy as np
import scipy.sparse as sp
import time
import matplotlib.pyplot as plt
import pyamg
from scipy.sparse.linalg import cg
from sympy import symbols, diff, integrate
import mat73
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utils import plot_comparison_and_compute_errors


# element stiffness
def elemstiff2d(nel,hx,hy):
    """
    Calculate element stiffness matrix for 2D finite elements.
    
    Args:
        nel (int): Number of elements (4 for quadrilateral)
        hx (float): Element size in x direction
        hy (float): Element size in y direction
        
    Returns:
        ndarray: Element stiffness matrix
    """
    x, y = symbols(['x', 'y'])
    fe = np.zeros((nel,nel))

    p = []
    p.append((hx-x)*(hy-y)/hx/hy) # f0: left-bottom
    p.append((x)*(hy-y)/hx/hy)    # f1: right-bottom
    p.append((hx-x)*(y)/hx/hy)    # f2: left-top
    p.append((x)*(y)/hx/hy)       # f3: right-top

    diff_p = []
    for i in range(len(p)):
        diff_p.append([diff(p[i],x),diff(p[i],y)])

    for j in range(len(p)):
        for i in range(len(p)):
            f = diff_p[i][0]*diff_p[j][0] + diff_p[i][1]*diff_p[j][1]
            int_f = integrate(f,(x,0,hx), (y,0,hy))   # integrate with scipy
            fe[j,i] = int_f
            
    return fe

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
    return force


def assemble_matrix(numnodx, numnody, stiffness, K):
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
    force = apply_dirichlet_conditions(bigk, force, dirichlet_nodes, dirichlet_values)

    mask = np.ones(numnod, dtype=bool)
    mask[dirichlet_nodes] = False

    bigk_reduced = bigk[mask, :][:, mask]
    force_reduced = force[mask]

    ml = pyamg.ruge_stuben_solver(bigk_reduced)
    solution_full[mask], _ = cg(bigk_reduced, force_reduced, M=ml.aspreconditioner())

    return solution_full

def calculate_flux(head_solved, K, numnodx, numnody, dx, dy):
    """
    Calculate flux components (q_x, q_y) from the hydraulic head and permeability field.
    
    Args:
        head_solved (ndarray): Solved hydraulic head (numnodx x numnody grid)
        K (ndarray): Permeability field for elements
        numnodx (int): Number of nodes in x direction
        numnody (int): Number of nodes in y direction
        dx (float): Element size in x direction
        dy (float): Element size in y direction
        
    Returns:
        tuple: (qx, qy) flux components in x and y directions
    """
    # Compute gradients of hydraulic head
    grad_x = ((head_solved[1:, :] - head_solved[:-1, :]) / dx)[:,1:]  # Gradient in x
    grad_y = ((head_solved[:, 1:] - head_solved[:, :-1]) / dy)[1:,:]  # Gradient in y

    # Convert element-wise K to nodal K by averaging
    Kx = K.reshape(numnodx - 1, numnody - 1)
    Ky = K.reshape(numnodx - 1, numnody - 1)

    # Calculate flux components
    qx = -Kx * grad_x
    qy = -Ky * grad_y

    return qx, qy


def test_solver_accuracy(mat_filename, plot_flag=False):
    """
    Test the solver accuracy by comparing the results with a reference solution.
    """
    mat_data = mat73.loadmat(mat_filename)
    head = mat_data['head']
    logK = mat_data['logK']
    q_original = -mat_data['Q']
    pump_well_loc = int(mat_data['pump_well_loc']) - 1 # matlab index starts from 1, numpy starts from 0
    print(pump_well_loc)
    print(logK.shape)
    print(head.shape)

    nx=ny=1024
    numnodx, numnody = nx + 1, ny + 1
    numnod = numnodx * numnody
    Lox, Loy = 320.0, 320.0 # domain real size, m
    dx, dy = Lox / nx, Loy / ny
    stiffness = elemstiff2d(4, dx, dy)
    
    K = np.exp(logK.flatten())  # Generate K without extra dimension

    # Flux density (in m³/s per m²)
    Q = q_original / dx /dy * 3600

    t0 = time.time()
    solution_full = groundwater_solver(K, pump_well_loc, Q)
    head_solved = solution_full.reshape((numnodx, numnody))
    
    print("Elapsed time for solving system:", time.time() - t0)
    
    # Compute error metrics
    L1_err = np.abs(head_solved.flatten() - head.flatten()).sum()
    L2_err = np.square(head_solved.flatten() - head.flatten()).sum()
    Max_err = np.square(head_solved.flatten() - head.flatten()).max()
        
    print(f"L1 Error: {L1_err:.6e}")
    print(f"L2 Error: {L2_err:.6e}")
    print(f"Maximum Error: {Max_err:.6e}")

    if plot_flag:
        plot_comparison_and_compute_errors(head, head_solved)


# Example usage
if __name__ == "__main__":

    test_solver_accuracy("./data/benchmark_1024.mat", plot_flag=True)
