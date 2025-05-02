import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pyamg
import time, mat73
from sympy import symbols, diff, integrate
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


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
    
    bigk, force = apply_dirichlet_conditions(bigk_transient, force, dirichlet_nodes, dirichlet_values)
    
    force[well_node] = Q
    
    mask = np.ones(numnod, dtype=bool)
    mask[dirichlet_nodes] = False

    bigk_reduced = bigk[mask, :][:, mask]
    force_reduced = force[mask]

    # Initial conditions
    head = initial_head.copy()[mask]
    
    head_over_time = np.empty((num_timesteps, numnod), dtype=np.float64)
    
    # Solve the system
    ml = pyamg.ruge_stuben_solver(bigk_reduced)
        
    # Time-stepping loop
    for step in range(num_timesteps):
        # Update right-hand side: mass_matrix @ head + Q
        rhs = (Ss / dt) * head + force_reduced


        head, _ = sp.linalg.cg(bigk_reduced, rhs, M=ml.aspreconditioner())

        # Store the result for this timestep
        head_over_time[step, mask] = head
    
    head_over_time[:, ~mask] = dirichlet_values

    return head_over_time


def test_transient_solver_accuracy(mat_filename):
    """
    Test the solver accuracy by comparing the results with a reference solution.
    """
    mat_data = mat73.loadmat(mat_filename)
    heads = mat_data['head'][:,:, 1:]
    logK = mat_data['logK']
    q_original = -mat_data['Q']
    pump_well_loc = int(mat_data['pump_well_loc']) - 1 # matlab index starts from 1, numpy starts from 0
    print(pump_well_loc)
    print(logK.shape)
    print(heads.shape)

    nx=ny=1024
    numnodx, numnody = nx + 1, ny + 1
    numnod = numnodx * numnody
    Lox, Loy = 320.0, 320.0 # domain real size, m
    dx, dy = Lox / nx, Loy / ny

    well_node = pump_well_loc
    K = np.exp(logK.flatten())  # Generate K without extra dimension

    # Flux density (in m³/s per m²)
    Q = q_original / dx /dy * 3600

    # Start timer
    t0 = time.time()
    
    # Simulation parameters
    dt = 0.1  # Time step in hours
    t_max = 1  # Total simulation time in hours

    initial_head = np.zeros(numnod)
    
    # Solve the transient problem
    head_over_time = transient_groundwater_solver(K, well_node, Q, initial_head, dt, t_max)
        
    print("Elapsed time for solving system:", time.time() - t0)
    
 
    for i in range(10):
        head_solved = head_over_time[i]
        head_true = heads[:,:,i].flatten()
        L1_err = np.abs(head_solved.flatten() - head_true.flatten()).sum()
        L2_err = np.square(head_solved.flatten() - head_true.flatten()).sum()
        Max_err = np.square(head_solved.flatten() - head_true.flatten()).max()

        print(i, L1_err)
        print(i, L2_err)
        print(i, Max_err)
        
    head_true = head_true.reshape((numnody, numnodx))
    head_solved = head_solved.reshape((numnody, numnodx))
    fig,ax = plt.subplots(figsize=(7,6))
    hmin, hmax = np.min(head_true.flatten()), np.max(head_true.flatten())
    im = ax.pcolormesh(head_true, cmap='viridis', vmin=hmin, vmax=hmax)
    lvls = np.linspace(-10,-0.1,7)
    cmp_str = 'RdBu'
    CT = ax.contour(head_true, levels=lvls,cmap=cmp_str)
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
    plt.show()
    
# Example usage
if __name__ == "__main__":
    stiffness = np.array([  # Element stiffness matrix
        [0.66666667, -0.16666667, -0.16666667, -0.33333333],
        [-0.16666667, 0.66666667, -0.33333333, -0.16666667],
        [-0.16666667, -0.33333333, 0.66666667, -0.16666667],
        [-0.33333333, -0.16666667, -0.16666667, 0.66666667]
    ])
    
    # mat_filename = 'GWSolver/benchmark_1024_transient.mat'
    # test_transient_solver_accuracy(mat_filename)
    
    nx, ny = 1024, 1024  # Grid size
    numel = nx * ny
    numnodx, numnody = nx + 1, ny + 1
    numnod = numnodx * numnody

    # Domain and material properties
    Lox, Loy = 320, 320  # Domain size in meters
    dx, dy = Lox / nx, Loy / ny

    K = np.exp(np.random.randn(numel) * 0.1 - 4)  # Generate K without extra dimension
    K = K.reshape((nx, ny))
    K[nx//5:nx//3, ny//4:ny//4*3] = np.exp(-2) 
    K[nx//3*2:nx//5*4, ny//4:ny//4*3] = np.exp(-8) 
    K = K.flatten()
    
    q_original = -0.002 * (64/nx)**2 # m3/s
    Q = q_original/dx/dy*3600 
    
    # Simulation parameters
    dt = 1.0  # Time step in hours
    t_max = 10  # Total simulation time in hours

    well_node = numnod//2
    initial_head = np.zeros(numnod)
    
    # Solve the transient problem
    head_over_time = transient_groundwater_solver(K, well_node, Q, initial_head, dt, t_max)
    
    # Plot the solution
    head_solved = head_over_time[-1].reshape((numnody, numnodx))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(head_solved, cmap='viridis')
    CT = ax.contour(head_solved, levels=10, colors='white')
    ax.clabel(CT, fontsize=10, inline=True, fmt='%.1f')
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title('Python FEM with AMG Preconditioning')
    
    # Visualization setup for animation
    fig, ax = plt.subplots(figsize=(7, 6))
    img = ax.imshow(
        head_over_time[0].reshape((numnody, numnodx)),
        extent=(0, Lox, 0, Loy),
        origin="lower",
        cmap="viridis",
        vmin=head_over_time.min()*0.7,
        vmax=head_over_time.max(),
    )
    cbar = plt.colorbar(img, ax=ax, label="Hydraulic Head (m)")
    contour_lines = None  # To store contour plot objects

    ax.set_title("Hydraulic Head Over Time")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # Animation update function
    def update(frame):
        global contour_lines
        img.set_data(head_over_time[frame].reshape((numnody, numnodx)))
        ax.set_title(f"Hydraulic Head at Time {frame * dt:.1f} hr")

        # # Remove old contours
        # if contour_lines:
        #     for c in contour_lines.collections:
        #         c.remove()

        # Add new contours
        head = head_over_time[frame].reshape((numnody, numnodx))
        contour_lines = ax.contour(head, levels=10, colors='white', extent=(0, Lox, 0, Loy), vmin=head_over_time.min()*0.7, vmax=head.max())
        return img, contour_lines

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(head_over_time), interval=1000, blit=True)

    # Save animation as GIF
    ani.save("hydraulic_head_simulation.gif", writer=PillowWriter(fps=5))
    print("GIF saved as 'hydraulic_head_simulation.gif'")
