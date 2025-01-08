import numpy as np
import scipy.sparse as sp
import time
import matplotlib.pyplot as plt
import pyamg
from scipy.sparse.linalg import cg
from sympy import symbols, diff, integrate
import mat73

# element stiffness
def elemstiff2d(nel,hx,hy):
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
    Apply Dirichlet boundary conditions directly to the global stiffness matrix and force vector.

    Args:
        bigk (scipy.sparse.csr_matrix): Global stiffness matrix.
        force (ndarray): Global force vector.
        dirichlet_nodes (ndarray): Indices of Dirichlet nodes.
        dirichlet_values (ndarray): Values to impose at Dirichlet nodes.

    Returns:
        bigk (scipy.sparse.csr_matrix): Modified stiffness matrix.
        force (ndarray): Modified force vector.
    """

    # Sparse matrix slicing for boundary columns
    dirichlet_contributions = bigk[:, dirichlet_nodes].dot(dirichlet_values)
    force -= dirichlet_contributions  # Subtract Dirichlet contributions from the force vector

    # Mask for non-Dirichlet nodes
    mask = np.ones(bigk.shape[0], dtype=bool)
    mask[dirichlet_nodes] = False

    # Reduce system for non-Dirichlet nodes
    bigk_reduced = bigk[mask, :][:, mask]
    force_reduced = force[mask]

    return bigk_reduced, force_reduced


def assemble_matrix(numnodx, numnody, fe0, K, Q, dirichlet_nodes, dirichlet_values, point_source_loc):
    """
    Prepare the reduced global stiffness matrix and force vector, incorporating the effects of Dirichlet boundary conditions.

    Args:
        numnodx (int): Number of nodes in the x direction.
        numnody (int): Number of nodes in the y direction.
        fe0 (ndarray): Element stiffness matrix (4x4 for 2D elements).
        K (ndarray): Array of element permeability values.
        Q (float): Source/sink term.

    Returns:
        bigk_reduced (scipy.sparse.csr_matrix): Reduced global stiffness matrix.
        force_reduced (ndarray): Reduced force vector.
        solution_full (ndarray): Full solution vector with boundary values set to 1 (left) and 0 (right).
    """
    # Number of elements and nodes
    numel = (numnodx - 1) * (numnody - 1)
    numnod = numnodx * numnody

    # Create connectivity matrix
    quotient, remainder = divmod(np.arange(numel), numnodx - 1)
    connect_mat = np.column_stack((
        remainder + quotient * numnodx,
        remainder + quotient * numnodx + 1,
        remainder + quotient * numnodx + numnodx,
        remainder + quotient * numnodx + numnodx + 1
    ))

    # Assemble global stiffness matrix
    sctr_rows = connect_mat.repeat(4, axis=1).flatten()
    sctr_cols = np.tile(connect_mat, 4).flatten()
    ke_values = (fe0 * K[:, None, None]).reshape(numel, -1).flatten()
    bigk = sp.coo_matrix((ke_values, (sctr_rows, sctr_cols)), shape=(numnod, numnod)).tocsr()

    # Initialize force vector
    force = np.zeros(numnod)
    force[point_source_loc] = Q  # Set the source/sink term
    
    bigk_reduced, force_reduced = apply_dirichlet_conditions(bigk, force, dirichlet_nodes, dirichlet_values)

    return bigk_reduced, force_reduced


def calculate_flux(head_solved, K, numnodx, numnody, dx, dy):
    """
    Calculate flux components (q_x, q_y) from the hydraulic head and permeability field.

    Args:
        head_solved (ndarray): Solved hydraulic head (numnodx x numnody grid).
        K (ndarray): Permeability field for elements.
        numnodx (int): Number of nodes in x direction.
        numnody (int): Number of nodes in y direction.
        dx (float): Element size in x direction.
        dy (float): Element size in y direction.

    Returns:
        qx (ndarray): Flux in x direction (numnodx-1 x numnody-1 grid).
        qy (ndarray): Flux in y direction (numnodx-1 x numnody-1 grid).
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

def plot_flux_map_streamlines(head_solved, qx, qy, dx, dy):
    """
    Plot the flux map using streamlines.

    Args:
        head_solved (ndarray): Solved hydraulic head (numnodx x numnody grid).
        qx (ndarray): Flux in x direction.
        qy (ndarray): Flux in y direction.
        dx (float): Element size in x direction.
        dy (float): Element size in y direction.
    """
    numnodx, numnody = head_solved.shape

    # Create coordinate grid for plotting
    x = np.linspace(0, dx * (numnodx - 1), numnodx)
    y = np.linspace(0, dy * (numnody - 1), numnody)
    X, Y = np.meshgrid(x, y)

    # Reduce flux grid size for visualization
    # Create coordinate grid for plotting
    speed = np.sqrt(qx**2 + qy**2)
    lw = 5*speed / speed.max()
    x = np.linspace(0, dx * (numnodx - 2), numnodx-1)
    y = np.linspace(0, dy * (numnody - 2), numnody-1)
    X_mid, Y_mid = np.meshgrid(x, y)

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, head_solved, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(label="Hydraulic Head")
    plt.streamplot(X_mid, Y_mid, qy, qx, color='black', density=[0.5, 2], linewidth=1, broken_streamlines=True)
    plt.title("Streamlines with Hydraulic Head Contours")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def test_solver_accuracy(mat_filename):
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

    # Identify Dirichlet boundary nodes
    left_boundary = np.arange(numnody) * numnodx  # Left boundary nodes
    right_boundary = left_boundary + (numnodx - 1)  # Right boundary nodes
    # Incorporate Dirichlet boundary effects efficiently
    dirichlet_nodes = np.concatenate((left_boundary, right_boundary))

    # Set Dirichlet boundary values
    solution_full = np.zeros(numnod)  # Full solution vector
    solution_full[left_boundary] = 0.0  # Left boundary set to 1
    solution_full[right_boundary] = 0.0  # Right boundary set to 0
    dirichlet_values = solution_full[dirichlet_nodes]

    # Start timer
    t0 = time.time()
    
    bigk_reduced, force_reduced = assemble_matrix(numnodx, numnody, stiffness, K, Q, dirichlet_nodes, dirichlet_values, pump_well_loc)

    print("Elapsed time for preparing matrix:", time.time() - t0)
    print("Reduced matrix shape:", bigk_reduced.shape)
    print("Reduced force vector length:", len(force_reduced))

    # Use AMG as a preconditioner for CG
    ml = pyamg.ruge_stuben_solver(bigk_reduced)
    solution_reduced, info = cg(bigk_reduced, force_reduced, M=ml.aspreconditioner())

    # Check for convergence
    if info == 0:
        print("Solver converged successfully.")
    else:
        print(f"Solver did not converge. Info: {info}")

    # Create a full solution vector and reimpose Dirichlet values
    mask = np.ones(numnod, dtype=bool)
    mask[dirichlet_nodes] = False
    solution_full[mask] = solution_reduced
    head_solved = solution_full.reshape((numnodx, numnody))
    
    print("Elapsed time for solving system:", time.time() - t0)
    
    fig,ax = plt.subplots(figsize=(7,6))
    hmin, hmax = np.min(head.flatten()), np.max(head.flatten())
    im = ax.pcolormesh(head, cmap='viridis', vmin=hmin, vmax=hmax)
    lvls = np.linspace(-10,-0.1,7)
    cmp_str = 'RdBu'
    CT = ax.contour(head, levels=lvls,cmap=cmp_str)
    ax.clabel(CT,fontsize=15,inline=True,inline_spacing=1,fmt='%.1f')
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title('Matlab FEM')


    fig,ax = plt.subplots(figsize=(7,6))
    im = ax.pcolormesh(head_solved, cmap='viridis')

    # im = ax.pcolormesh(head_solved, cmap='viridis', vmin=hmin, vmax=hmax)
    CT = ax.contour(head_solved, levels=lvls,cmap=cmp_str)
    ax.clabel(CT,fontsize=15,inline=True,inline_spacing=1,fmt='%.1f')
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title('Python FEM')

    L1_err = np.abs(head_solved.flatten() - head.flatten()).sum()
    L2_err = np.square(head_solved.flatten() - head.flatten()).sum()
    Max_err = np.square(head_solved.flatten() - head.flatten()).max()

    print(L1_err)
    print(L2_err)
    print(Max_err)
    plt.show()



# Example usage
if __name__ == "__main__":

    # test_solver_accuracy("./GWSolver/benchmark_1024.mat")
    
    # Define domain parameters
    nx, ny = 64, 64 # domain resolution
    numel = nx * ny
    numnodx, numnody = nx + 1, ny + 1
    numnod = numnodx * numnody
    Lox, Loy = 320.0, 320.0 # domain real size, m
    dx, dy = Lox / nx, Loy / ny
    stiffness = elemstiff2d(4, dx, dy)
    
    K = np.exp(np.random.randn(numel) * 0.1 - 4)  # Generate K without extra dimension
    K = K.reshape((nx, ny))
    K[nx//5:nx//3, ny//4:ny//4*3] = np.exp(-5) 
    K[nx//3*2:nx//5*4, ny//4:ny//4*3] = np.exp(-8) 
    K = K.flatten()

    # Original pumping rate
    q_original = -0.02  # m³/s
    # Area associated with each node (element)
    A = Lox/nx  * Loy/ny
    # Flux density (in m³/s per m²)
    Q = q_original / A * 3600

    # Identify Dirichlet boundary nodes
    left_boundary = np.arange(numnody) * numnodx  # Left boundary nodes
    right_boundary = left_boundary + (numnodx - 1)  # Right boundary nodes
    # Incorporate Dirichlet boundary effects efficiently
    dirichlet_nodes = np.concatenate((left_boundary, right_boundary))

    # Set Dirichlet boundary values
    solution_full = np.zeros(numnod)  # Full solution vector
    solution_full[left_boundary] = 0.0  # Left boundary set to 1
    solution_full[right_boundary] = 0.0  # Right boundary set to 0
    dirichlet_values = solution_full[dirichlet_nodes]

    # Start timer
    t0 = time.time()
    
    bigk_reduced, force_reduced = assemble_matrix(numnodx, numnody, stiffness, K, Q, dirichlet_nodes, dirichlet_values, numnod//2)

    print("Elapsed time for preparing matrix:", time.time() - t0)
    print("Reduced matrix shape:", bigk_reduced.shape)
    print("Reduced force vector length:", len(force_reduced))

    # Use AMG as a preconditioner for CG
    ml = pyamg.ruge_stuben_solver(bigk_reduced)
    solution_reduced, info = cg(bigk_reduced, force_reduced, M=ml.aspreconditioner())

    # Check for convergence
    if info == 0:
        print("Solver converged successfully.")
    else:
        print(f"Solver did not converge. Info: {info}")

    # Create a full solution vector and reimpose Dirichlet values
    mask = np.ones(numnod, dtype=bool)
    mask[dirichlet_nodes] = False
    solution_full[mask] = solution_reduced
    head_solved = solution_full.reshape((numnodx, numnody))

    print("Elapsed time for solving system:", time.time() - t0)

    # Plot the solution
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(head_solved, cmap='viridis')
    CT = ax.contour(head_solved, levels=10, colors='white')
    ax.clabel(CT, fontsize=10, inline=True, fmt='%.1f')
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title('Python FEM with AMG Preconditioning')

    # Assuming K is the permeability field with (numnodx-1) x (numnody-1) elements
    qx, qy = calculate_flux(head_solved, K, numnodx, numnody, dx, dy)
    plot_flux_map_streamlines(head_solved, qx, qy, dx, dy)

    plt.show()