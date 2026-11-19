"""
Groundwater Flow Solver Module.
This module implements finite element methods to solve steady-state & transient groundwater flow equations.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from sympy import symbols, diff, integrate


def build_preconditioner(matrix):
    """Build an AMG preconditioner, or return ``None`` when pyamg is absent."""
    try:
        import pyamg
    except ImportError:
        return None
    return pyamg.ruge_stuben_solver(matrix).aspreconditioner()


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

    solution_full[mask], _ = cg(
        bigk_reduced, force_reduced, M=build_preconditioner(bigk_reduced)
    )

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

def assemble_transient_matrix(numnodx, numnody, fe0, K):
    """
    Assemble the global matrix for transient groundwater flow.

    Args:
        numnodx (int): Number of nodes in x direction.
        numnody (int): Number of nodes in y direction.
        fe0 (ndarray): Element stiffness matrix.
        K (ndarray): Permeability field.
        Ss (ndarray): Specific storage field.
        dt (float): Time step.

    Returns:
        scipy.sparse.csr_matrix: Assembled global matrix for the transient flow.
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

    # Stiffness matrix for the flow term
    sctr_rows = connect_mat.repeat(4, axis=1).flatten()
    sctr_cols = np.tile(connect_mat, 4).flatten()
    ke_values = (fe0 * K[:, None, None]).reshape(numel, -1).flatten()
    bigk_flow = sp.coo_matrix((ke_values, (sctr_rows, sctr_cols)), shape=(numnod, numnod)).tocsr()

    return bigk_flow

def transient_groundwater_solver(K, well_node, Q, initial_head, dt, t_max):
    """
    Solve transient groundwater flow.

    Args:
        K (ndarray): Permeability field.
        Ss (ndarray): Specific storage field.
        Q (ndarray): Source/sink term.
        initial_head (ndarray): Initial hydraulic head.
        numnodx (int): Number of nodes in x direction.
        numnody (int): Number of nodes in y direction.
        fe0 (ndarray): Element stiffness matrix.
        dt (float): Time step.
        t_max (float): Maximum simulation time.

    Returns:
        ndarray: Hydraulic head solutions over time.
    """
    numel = K.shape[0]
    nx = ny = int(np.sqrt(numel))
    numnodx = numnody = nx + 1
    numnod = numnodx * numnody
    dx = dy = 320.0/ nx
    Ss = 1e-4 * dx * dy  # Specific storage

    stiffness = elemstiff2d(4, dx, dy)
    
    num_timesteps = int(t_max / dt)

    left_boundary = np.arange(numnody) * numnodx
    right_boundary = left_boundary + (numnodx - 1)
    dirichlet_nodes = np.concatenate((left_boundary, right_boundary))

    solution_full = np.zeros(numnod)
    solution_full[left_boundary] = 0.0
    solution_full[right_boundary] = 0.0
    dirichlet_values = solution_full[dirichlet_nodes]

    force = np.zeros(numnod)

    # Ax = b
    # A11*x1 +A12*x2 + ... A1n*xn = b1
    # A21*x1 +A22*x2 + ... A2n*xn = b2
    # ...
    # An1*x1 +An2*x2 + ... Ann*xn = bn
    
    
    # Assemble the transient matrix
    bigk = assemble_transient_matrix(numnodx, numnody, stiffness, K)
    
    # Adjust the transient matrix by adding Ss/dt to the diagonal
    bigk_transient = bigk + sp.eye(numnod, format='csr') * (Ss / dt)
    
    force = apply_dirichlet_conditions(bigk_transient, force, dirichlet_nodes, dirichlet_values)
    
    force[well_node] = Q
    
    mask = np.ones(numnod, dtype=bool)
    mask[dirichlet_nodes] = False

    bigk_reduced = bigk_transient[mask, :][:, mask]
    force_reduced = force[mask]

    # Solve the system
    preconditioner = build_preconditioner(bigk_reduced)
    
    head_over_time = np.empty((num_timesteps, numnod), dtype=np.float64)
    
    # Initial conditions
    head = initial_head.copy()[mask]
        
    # Time-stepping loop
    for step in range(num_timesteps):
        # Update right-hand side: mass_matrix @ head + Q
        rhs = (Ss / dt) * head + force_reduced

        head, _ = sp.linalg.cg(bigk_reduced, rhs, M=preconditioner)

        # Store the result for this timestep
        head_over_time[step, mask] = head       
    
    head_over_time[:, ~mask] = dirichlet_values

    return head_over_time   

def groundwater_solver_one_step(K, well_node, Q, head_i, dt):
    """
    Advance the groundwater head by one time step using the transient solver.

    Args:
        K (ndarray): Permeability field.
        well_node (int): Index of the well location in the grid.
        Q (float): Pumping rate at the well.
        head_i (ndarray): Hydraulic head at current time step (flattened array).
        dt (float): Time step size.

    Returns:
        ndarray: Hydraulic head at the next time step (flattened array).
    """
    numel = K.shape[0]
    nx = ny = int(np.sqrt(numel))
    numnodx = numnody = nx + 1
    numnod = numnodx * numnody
    dx = dy = 320.0 / nx
    Ss = 1e-4 * dx * dy  # Specific storage

    stiffness = elemstiff2d(4, dx, dy)

    left_boundary = np.arange(numnody) * numnodx
    right_boundary = left_boundary + (numnodx - 1)
    dirichlet_nodes = np.concatenate((left_boundary, right_boundary))

    solution_full = np.zeros(numnod)
    solution_full[left_boundary] = 0.0
    solution_full[right_boundary] = 0.0
    dirichlet_values = solution_full[dirichlet_nodes]

    force = np.zeros(numnod)
    force[well_node] = Q
    force = apply_dirichlet_conditions(
        assemble_transient_matrix(numnodx, numnody, stiffness, K) + sp.eye(numnod, format='csr') * (Ss / dt),
        force,
        dirichlet_nodes,
        dirichlet_values
    )

    mask = np.ones(numnod, dtype=bool)
    mask[dirichlet_nodes] = False

    bigk = assemble_transient_matrix(numnodx, numnody, stiffness, K)
    bigk_transient = bigk + sp.eye(numnod, format='csr') * (Ss / dt)
    bigk_reduced = bigk_transient[mask, :][:, mask]
    force_reduced = force[mask]

    preconditioner = build_preconditioner(bigk_reduced)
    head = head_i.copy()[mask]
    rhs = (Ss / dt) * head + force_reduced
    head_new, _ = sp.linalg.cg(bigk_reduced, rhs, M=preconditioner)

    head_full = np.zeros(numnod)
    head_full[mask] = head_new
    head_full[dirichlet_nodes] = dirichlet_values
    return head_full
