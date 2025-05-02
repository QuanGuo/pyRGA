"""
Reformulated Geostatistical Approach (RGA) module for hydraulic tomography analysis.
This module implements the RGA method for estimating hydraulic conductivity fields.
"""

import numpy as np
import time
import json
from scipy.linalg import svd
import os
from scipy.special import gamma
# Set working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Create figures directory if it doesn't exist
if not os.path.exists('figs'):
    os.makedirs('figs')

from hydraulic_tomography import hydraulic_tomography
from utils import plot_conductivity_fields, plot_parameters, plot_observations_vs_predictions, plot_head_fields, plot_history, plot_parameter_history
import matplotlib.pyplot as plt

def generate_random_field(n_reals=1, Nx=256, Ny=256,
                         cov_type='gaussian', mu=0, variance=1.0,
                         lx=0.1, ly=0.2, nu=0.5):
    """
    Generate 2D Gaussian random fields with specified covariance structure using spectral method

    Parameters:
    n_reals  : Number of realizations (squeezed if 1)
    Nx, Ny   : Grid dimensions
    cov_type : 'gaussian', 'exponential', or 'matern'
    mu       : Mean of the field
    variance : Field variance
    lx, ly   : Correlation lengths in x/y directions
    nu       : Smoothness parameter for Matérn covariance

    Returns:
    field    : Random field(s) with shape (n_reals, Nx, Ny) or (Nx, Ny) if n_reals=1
    """
    # Validate inputs
    allowed_types = ['gaussian', 'exponential', 'matern']
    if cov_type not in allowed_types:
        raise ValueError(f"Invalid cov_type. Choose from {allowed_types}")

    # Convert exponential to Matérn with ν=0.5
    if cov_type == 'exponential':
        cov_type = 'matern'
        nu = 0.5

    # Angular wave numbers
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=1/Nx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=1/Ny)
    Kx, Ky = np.meshgrid(kx, ky)

    # Calculate theoretical power spectrum
    if cov_type == 'gaussian':
        S = (2 * np.pi * lx * ly) * np.exp(-((lx*Kx)**2 + (ly*Ky)**2)/2)
    elif cov_type == 'matern':
        S = (2 * np.pi * lx * ly * (4*nu)**nu / gamma(nu) *
             (1 + ((lx*Kx)**2 + (ly*Ky)**2)/(2*nu)) ** -(nu + 1))

    # Normalize to match target variance
    current_power = np.sum(S)
    S *= (variance * Nx * Ny) / current_power

    # Generate white noise
    white_noise = np.random.normal(0, 1, (n_reals, Nx, Ny))

    # Spectral domain transformation
    fft_noise = np.fft.fft2(white_noise, axes=(-2, -1))
    fft_field = fft_noise * np.sqrt(S)
    field = np.fft.ifft2(fft_field, axes=(-2, -1)).real + mu

    return np.squeeze(field, axis=0) if n_reals == 1 else field

def observation_operator(hydraulic_heads, well_nodes):
    """
    Create observation operator for hydraulic tomography.
    
    Args:
        hydraulic_heads (ndarray): Matrix of hydraulic heads for each well
        well_nodes (list): List of well node indices
        
    Returns:
        ndarray: Flattened vector of head differences between well pairs
    """
    obs = np.empty((len(well_nodes), len(well_nodes) - 1))
    for i, _ in enumerate(well_nodes):
        obs[i] = hydraulic_heads[i, well_nodes[:i]+well_nodes[i+1:]].flatten()
    return obs.flatten()


def forward_model(b, V, Q, well_nodes):
    """
    Forward model for hydraulic tomography.

    Args:
        b (k,): Vector of coefficients for basis functions
        V (N_logK, k): Matrix of basis functions
        Q scaler: Pumping rates at each well
        well_nodes (ndarray): Indices of well locations in the model grid

    Returns:
        ndarray: Vector of observed hydraulic head differences between well pairs
    """
    alpha = b[:-1]
    mu = b[-1]
    s = np.squeeze(V @ alpha[:, np.newaxis],axis=1)
    hydraulic_heads = hydraulic_tomography(np.exp(s+mu), well_nodes, Q)
    yp = observation_operator(hydraulic_heads, well_nodes)
    return yp

def gauss_newton_dynamic_lambda(
    f,
    b0,
    y_obs,
    lam_init=1e-3,
    max_iter=10,
    tol=1e-6,
    history={},
    adaptive_lambda=True,
    anneal_lambda=True,
    min_lambda=1e-5
):
    """
    Gauss-Newton optimization with dynamic lambda adjustment.
    
    Args:
        f (callable): Forward model function
        b0 (ndarray): Initial parameter vector
        y_obs (ndarray): Observed data vector
        lam_init (float): Initial lambda value for regularization
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance
        history (dict): Dictionary to store optimization history
        adaptive_lambda (bool): Whether to adapt lambda based on loss
        anneal_lambda (bool): Whether to anneal lambda over iterations
        min_lambda (float): Minimum allowed lambda value
        
    Returns:
        tuple: (optimized parameters, optimization history)
    """
    b = b0.copy()
    lam = lam_init

    cumulative_resd = 0.0  # For annealing-based lambda update

    # Normalize y_obs by L2 norm
    scale = np.linalg.norm(y_obs)
    y_obs_norm = y_obs / scale

    print(f"{'Iter':<6}{'Lambda':<12}{'Loss':<15}{'Step Norm':<15}{'Time (s)':<10}")
    print("-" * 60)

    for it in range(max_iter):
        start_time = time.time()

        yp = f(b) / scale
        r = y_obs_norm - yp
        L_old = 0.5 * np.dot(r, r) / lam + 0.5 * np.dot(b[:-1], b[:-1])

        # Jacobian via finite difference
        eps = 1e-6
        J = np.zeros((len(y_obs), len(b)))
        for i in range(len(b)):
            b_eps = b.copy()
            b_eps[i] += eps
            J[:, i] = (f(b_eps) / scale - yp) / eps

        Dump = np.eye(len(b))
        Dump[-1,-1] = 0.0

        H = J.T @ J/lam + Dump
        g = J.T @ (r+J@b)/lam
        delta = np.linalg.solve(H, g)
        b_new = delta

        yp_new = f(b_new) / scale
        r_new = y_obs_norm - yp_new
        L_new = 0.5 * np.dot(r_new, r_new) / lam + 0.5 * np.dot(b_new[:-1], b_new[:-1])
        step_norm = L_old/L_new - 1

        # Calculate elapsed time
        elapsed = time.time() - start_time

        print(f"{it+1:<6}{lam:<12.2e}{L_new:<15.6e}{step_norm:<15.6e}{elapsed:<10.3f}")

        # Accumulate residual for annealing-based lambda
        cumulative_resd += np.mean((y_obs_norm - yp_new)**2)

        # Accept step if loss decreased
        if L_new < L_old:
            b = b_new
            # if adaptive_lambda:
            #     lam = lam * 2
            if step_norm < tol:
                print("✅ Converged.")
                break
        else:
            if adaptive_lambda:
                lam = lam / 10
                print("🔁 Step rejected — decreasing lambda.")

        history["alpha"].append(b[:-1].copy())
        history["mu"].append(b[-1])
        history["loss"].append(float(L_new))
        history["lambda"].append(float(lam))
        history["step_norm"].append(float(step_norm))
        history["time"].append(float(elapsed))
        history["yp"].append(yp*scale)
        with open('history.json', 'w') as history_file:
            json_history = {    
                "alpha": [arr.tolist() for arr in history["alpha"]],
                "mu": history["mu"],
                "loss": history["loss"],
                "lambda": history["lambda"],
                "step_norm": history["step_norm"],
                "time": history["time"],
                "true_alpha": history["true_alpha"].tolist(),
                "true_mu": history["true_mu"],
                "true_y": history["true_y"].tolist(),
                "yp": [arr.tolist() for arr in history["yp"]]
            }
            json.dump(json_history, history_file)

        # Annealing logic every 3 iterations
        if anneal_lambda and (it + 1) % 3 == 0:
            lam = max(cumulative_resd / (it + 1), min_lambda)

    return b, history

def load_history(filename='history.json'):
    """
    Load optimization history from JSON file and convert arrays back to numpy format.
    
    Args:
        filename (str): Path to JSON history file
        
    Returns:
        dict: History dictionary with keys:
            'b' (list of ndarrays): Parameter vectors
            'loss' (list): Loss values
            'lambda' (list): Lambda values
            'step_norm' (list): Step norm values 
            'time' (list): Computation times
            'alpha' (ndarray): True parameter vector
    """
    with open(filename, 'r') as f:
        history = json.load(f)
        
    # Convert b arrays back to numpy
    history['alpha'] = [np.array(b) for b in history['alpha']]
    history['mu'] = np.array(history['mu'])
    history['true_alpha'] = np.array(history['true_alpha'])
    history['true_y'] = np.array(history['true_y'])
    history['yp'] = [np.array(yp) for yp in history['yp']]
    return history


def head_metrics(true_heads, predicted_heads):
    mae = np.mean(np.abs(true_heads - predicted_heads))
    rmse = np.sqrt(np.mean((true_heads - predicted_heads)**2))
    mse = np.mean((true_heads - predicted_heads)**2)
    l2_relative_error = np.linalg.norm(true_heads - predicted_heads) / np.linalg.norm(true_heads)
    print(f"Heads Metrics:")
    print(f"MAE: {mae} m")
    print(f"RMSE: {rmse} m")
    print(f"MSE: {mse} m^2") 
    print(f"L2 relative error: {l2_relative_error}")
    return mae, rmse, mse, l2_relative_error

def conductivity_metrics(true_field, predicted_field, accuracy_threshold=0.1):

    l2_relative_error = np.linalg.norm(true_field - predicted_field) / np.linalg.norm(true_field)
    accuracy = np.sum(np.abs(true_field - predicted_field) < accuracy_threshold * (true_field.max()-true_field.min())) / len(true_field)
    print(f"Conductivity Field Metrics:")
    print(f"L2 relative error: {l2_relative_error}")
    print(f"Accuracy: {accuracy}")
    return l2_relative_error, accuracy

if __name__ == "__main__":
    # Main code translation
    Lox = 100
    Loy = 100
    nx = 64
    ny = 64

    numel = nx * ny
    numnodx, numnody = nx + 1, ny + 1
    numnod = numnodx * numnody
    Lox, Loy = 320, 320
    dx, dy = Lox / nx, Loy / ny

    sigma = 1.0
    lx, ly = 0.15, 0.2

    # PCA to reduce parameter dimension
    NR = 400
    k = 50
    mu = -4

    ucr = generate_random_field(
        NR,
        Nx=nx,
        Ny=ny,
        cov_type='gaussian',
        variance=sigma**2,
        lx=lx,
        ly=ly,
        mu=mu
    ).reshape(NR, -1)

    beta = np.mean(ucr)

    # Conduct SVD
    U, S, Vt = svd(ucr - beta, full_matrices=False)

    # Generate the pseudo-eigenvectors
    k = 50
    V = np.expand_dims(np.sqrt(S[:k]),axis=1)*Vt[:k]
    alpha = np.random.randn(k)
    # Generate synthetic random field
    logK = mu + (V.T @ alpha[:, np.newaxis]).flatten()

    # logK = np.exp(np.random.randn(numel) * 0.1 - 2)  # Generate logK without extra dimension
    # logK[nx//5:nx//3, ny//4:ny//4*3] = 0
    # logK[nx//3*2:nx//5*4, ny//4:ny//4*3] = -4

    K = np.exp(logK)

    q_original = -0.02 * (64/nx)**2 # m3/s
    Q = q_original/dx/dy*3600

    well_relative_locs = []
    horizontal_relative_locs = vertial_relative_locs = [0.25, 0.375, 0.5, 0.625, 0.75]
    for h in horizontal_relative_locs:
        for v in vertial_relative_locs:
            well_relative_locs.append((h,v))

    well_nodes = [int(x*numnodx)+int(y*numnody)*numnodx for x, y in well_relative_locs]

    t0 = time.time()
    hydraulic_heads = hydraulic_tomography(K, well_nodes, Q)
    print("Elapsed time for solving HT:", time.time() - t0)

    # observations
    y0 = observation_operator(hydraulic_heads, well_nodes)

    # Add random errors
    Nobv = len(y0)
    rng = np.random.default_rng()
    noise_level = 0.05
    obv_error = np.abs(y0) * rng.normal(0, noise_level, Nobv)
    y = y0 + obv_error

    # Save true values to json before optimization
    initial_history = {
        'true_alpha': alpha,
        'true_mu': mu,
        'true_y': y0,
        'alpha': [],
        'mu': [],
        'loss': [],
        'lambda': [],
        'step_norm': [],
        'time': [],
        'yp': []
    }

    b, opt_history = gauss_newton_dynamic_lambda(
        lambda b: forward_model(b, V.T, Q, well_nodes),
        b0=np.concatenate((np.zeros(k), np.array([beta]))),
        y_obs=y,
        lam_init=1e-3,
        max_iter=10,
        tol=1e-5,
        history=initial_history,
        adaptive_lambda=False,
        anneal_lambda=True,
        min_lambda=1e-5
    )

    history = load_history()

    true_heads = history['true_y']
    predicted_heads = history['yp'][-1]
    mae, rmse, mse, l2_relative_error = head_metrics(true_heads, predicted_heads)

    predicted_alpha = history['alpha'][-1]
    predicted_mu = history['mu'][-1]

    true_alpha = history['true_alpha']
    true_mu = history['true_mu']
    reconstructed_field = np.squeeze(V.T @ predicted_alpha[:, np.newaxis],axis=1) + predicted_mu
    true_field = np.squeeze(V.T @ true_alpha[:, np.newaxis]) + true_mu

    conductivity_metrics(true_field, reconstructed_field)

    # Plot the solution
    plot_conductivity_fields(reconstructed_field, true_field, nx, ny)

    # # Plot the parameter history
    # plot_parameter_history(history, V, beta)

    # # Plot the optimization history
    # plot_history(history)

    # Plot the parameters comparison
    plot_parameters(true_alpha, predicted_alpha)

    # Plot the heads comparison
    plot_observations_vs_predictions(true_heads, predicted_heads)

    true_head_field = hydraulic_tomography(np.exp(true_field), well_nodes, Q)
    predicted_head_field = hydraulic_tomography(np.exp(reconstructed_field), well_nodes, Q)


    pump_id = 0 # 0, 1, 2, 3, 4 
    plot_head_fields(true_head_field[pump_id].reshape((nx+1, ny+1)), predicted_head_field[pump_id].reshape((nx+1, ny+1)))

    mae, rmse, mse, l2_relative_error = head_metrics(true_head_field[pump_id], predicted_head_field[pump_id])

