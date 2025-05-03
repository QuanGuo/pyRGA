import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.typing as npt
import typing
from matplotlib.colors import Normalize, LogNorm

def visualiztion_one2one_3D(self, fields_prior: npt.NDArray[np.float32], fields_pred: npt.NDArray[np.float32],
                            sims: int, property_name: str, plot_range: typing.Tuple = (126, 125, 110)):
    """
    Visualize 3D fields for prior and predicted results side by side, and save them as images.

    Args:
        fields_prior (ndarray): 3D array of prior fields.
        fields_pred (ndarray): 3D array of predicted fields.
        sims (int): Simulation index or "mean" for averaging.
        property_name (str): Property name (e.g., "PORO", "PERMXY", "PERMZ").
        plot_range (tuple): Range of the plot in (x, y, z).
    """
    x_range, y_range, z_range = plot_range

    # Extract the relevant 3D field for prior and predicted results
    field_prior = self.get_field(fields_prior, property_name, sims, x_range, y_range, z_range)
    field_pred = self.get_field(fields_pred, property_name, sims, x_range, y_range, z_range)

    # Normalize and setup colormap for visualization
    if property_name == "PORO":
        norm = Normalize(vmin=field_prior.min(), vmax=field_prior.max())
    else:
        norm = LogNorm(vmin=field_prior.min(), vmax=field_prior.max())

    # Ensure the output directory exists
    os.makedirs(f'{self.output_dir}/figures', exist_ok=True)

    # Plot the predicted field
    self.plot_3D_surface(
        data=field_pred,
        property_name=property_name,
        norm=norm,
        figname=os.path.join(f'{self.output_dir}/figures', f"LANL_hm_{property_name}_{sims}_{x_range}x{y_range}x{z_range}.png")
    )

    # Plot the prior field
    self.plot_3D_surface(
        data=field_prior,
        property_name=property_name,
        norm=norm,
        figname=os.path.join(f'{self.output_dir}/figures', f"LANL_prior_{property_name}_{sims}_{x_range}x{y_range}x{z_range}.png")
    )

def plot_3D_surface(data: npt.NDArray[np.float32], property_name: str, norm, figname):
    """
    Plot a 3D surface of the given data and save the visualization.

    Args:
        data (ndarray): 3D array representing the data to be visualized.
        property_name (str): Property name to be used in titles and labels.
        norm: Normalization function for colormap.
        figname (str): Filename to save the plot.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    label_fontsize = 10
    title_fontsize = 12

    ax.zaxis.set_rotate_label(False)  # Disable automatic rotation for Z label
    ax.set_zlabel('Cell Grid ID (Z)', fontsize=label_fontsize, rotation=90)

    cmap = plt.get_cmap('viridis')  # Set colormap for visualization

    # Helper function to plot individual surfaces
    def plot_surface(array, x, y, z):
        ax.plot_surface(x, y, z, facecolors=cmap(norm(array)), rstride=1, cstride=1, shade=False)

    # Get dimensions of the data
    nx, ny, nz = data.shape

    # Plot surfaces for different slices
    z = 0
    y, x = np.meshgrid(np.arange(ny + 1), np.arange(nx + 1))
    plot_surface(np.pad(data[:, :, z], ((0, 1), (0, 1)), mode="edge"), x, y, np.full_like(x, z))

    y = ny - 1
    z, x = np.meshgrid(np.arange(nz + 1), np.arange(nx + 1))
    plot_surface(np.pad(data[:, y, :], ((0, 1), (0, 1)), mode="edge"), x, np.full_like(x, ny), z)

    x = nx - 1
    z, y = np.meshgrid(np.arange(nz + 1), np.arange(ny + 1))
    plot_surface(np.pad(data[x, :, :], ((0, 1), (0, 1)), mode="edge"), np.full_like(y, nx), y, z)

    # L-shaped surface on x=0
    z_trim, y_trim = np.meshgrid(np.arange(nz + 1), np.arange(ny // 2, ny + 1))
    plot_surface(np.pad(data[0, ny // 2:, :], ((0, 1), (0, 1)), mode="edge"), np.full_like(z_trim, 0), y_trim, z_trim)

    # # Add a colorbar
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array(data)
    # cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7, pad=0.1)

    # cbar.set_label('Permeability', fontsize=label_fontsize, rotation=270, labelpad=10)
    # cbar.ax.tick_params(labelsize=label_fontsize)

    # Set axis limits and labels
    ax.set_xlim([0, nx])
    ax.set_ylim([0, ny])
    ax.set_zlim([0, nz])
    ax.set_xlabel('Cell Grid ID (X)', fontsize=label_fontsize)
    ax.set_ylabel('Cell Grid ID (Y)', fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=label_fontsize)
    ax.tick_params(axis='z', labelsize=label_fontsize)

    # Title

    ax.set_title(f'{property_name}', fontsize=title_fontsize)
    plt.show()
    # Save the figure
    fig.savefig(figname, transparent=True)


def plot_comparison_and_compute_errors(head, head_solved):
    """
    Plot comparison between Matlab and Python FEM solutions and compute error metrics.
    Saves plots to .fig folder.
    
    Args:
        head (ndarray): Reference solution from Matlab
        head_solved (ndarray): Solution from Python solver
        
    Returns:
        tuple: L1, L2 and Max errors between solutions
    """
    # Create .fig directory if it doesn't exist
    if not os.path.exists('.fig'):
        os.makedirs('.fig')
        
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
    
    # Plot settings
    hmin, hmax = np.min(head.flatten()), np.max(head.flatten())
    lvls = np.linspace(-10,-0.1,7)
    cmp_str = 'RdBu'
    
    # First subplot - Matlab solution
    im1 = ax1.pcolormesh(head, cmap='viridis', vmin=hmin, vmax=hmax)
    CT1 = ax1.contour(head, levels=lvls, cmap=cmp_str)
    ax1.clabel(CT1, fontsize=15, inline=True, inline_spacing=1, fmt='%.1f')
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('Matlab FEM')
    
    # Second subplot - Python solution  
    im2 = ax2.pcolormesh(head_solved, cmap='viridis', vmin=hmin, vmax=hmax)
    CT2 = ax2.contour(head_solved, levels=lvls, cmap=cmp_str)
    ax2.clabel(CT2, fontsize=15, inline=True, inline_spacing=1, fmt='%.1f')
    fig.colorbar(im2, ax=ax2)
    ax2.set_title('Python FEM')
    
    # Save figure
    plt.savefig('.fig/fem_comparison.png', dpi=300, bbox_inches='tight')
    

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
    plt.contourf(X, Y, head_solved, levels=20, cmap='viridis', alpha=1)
    plt.colorbar(label="Hydraulic Head")
    plt.streamplot(X_mid, Y_mid, qy, qx, color='black', density=[0.5, 2], linewidth=1, broken_streamlines=True)
    plt.title("Streamlines with Hydraulic Head Contours")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(".fig/flux_map_streamlines.png", dpi=300, transparent=True)


def plot_history(history, save_path='./figs/optimization_history.png'):
    """
    Plot optimization history including loss, lambda, step norm and computation time.
    
    Args:
        history (dict): Dictionary containing optimization history with keys:
            'loss', 'lambda', 'step_norm', 'time'
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot loss history
    ax1.semilogy(history['loss'])
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss History')
    ax1.grid(True)

    # Plot lambda history 
    ax2.semilogy(history['lambda'])
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Lambda')
    ax2.set_title('Lambda History')
    ax2.grid(True)

    # Plot step norm history
    ax3.semilogy(history['step_norm'])
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Step Norm')
    ax3.set_title('Step Norm History')
    ax3.grid(True)

    # Plot computation time
    ax4.plot(history['time'])
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Time (s)')
    ax4.set_title('Computation Time')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def plot_parameter_history(history, V, beta=0, save_path='./figs/parameter_history.png' ):
    """
    Plot the evolution of the parameter field during optimization.
    
    Args:
        history (dict): Dictionary containing optimization history
        V (ndarray): Matrix of basis functions
        beta (float): Mean of random field (default 0)
    """
    n_iters = len(history['b'])
    n_cols = min(5, n_iters)  # Show max 5 iterations per row
    n_rows = (n_iters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
        
    for i in range(n_iters):
        row = i // n_cols
        col = i % n_cols
        
        # Transform b to parameter field
        s = V.T @ history['b'][i][:, np.newaxis] + beta
        
        im = axes[row, col].imshow(s.reshape(-1, int(np.sqrt(len(s)))), cmap='jet')
        axes[row, col].set_title(f'Iteration {i}')
        plt.colorbar(im, ax=axes[row, col])
        
    # Remove empty subplots
    for i in range(i+1, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def plot_conductivity_fields(reconstructed_field, true_field, nx, ny, save_path='./figs/conductivity_fields.png'):
        """
        Plot and compare reconstructed and true conductivity fields
        
        Args:
            reconstructed_field: The reconstructed conductivity field
            true_field: The true conductivity field 
            nx: Number of grid points in x direction
            ny: Number of grid points in y direction
            save_path: Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Get global min and max for consistent colorbar scale
        vmin = min(reconstructed_field.min(), true_field.min())
        vmax = max(reconstructed_field.max(), true_field.max())
        
        # Plot reconstructed field
        im1 = ax1.pcolormesh(reconstructed_field.reshape((nx, ny)), cmap='jet', vmin=vmin, vmax=vmax)
        fig.colorbar(im1, ax=ax1)
        ax1.set_title('Reconstructed Conductivity Field')
        
        # Plot true field  
        im2 = ax2.pcolormesh(true_field.reshape((nx, ny)), cmap='jet', vmin=vmin, vmax=vmax)
        fig.colorbar(im2, ax=ax2)
        ax2.set_title('True Conductivity Field')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def plot_parameters(true_alpha, predicted_alpha, save_path='./figs/parameters.png'):
    """
    Create a 45-degree cross-plot comparing true and predicted parameters
    
    Args:
        true_alpha: True parameter values
        predicted_alpha: Predicted parameter values 
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get axis limits
    min_val = min(np.min(true_alpha), np.min(predicted_alpha))
    max_val = max(np.max(true_alpha), np.max(predicted_alpha))
    buffer = (max_val - min_val) * 0.1
    
    # Plot 45 degree line
    ax.plot([min_val-buffer, max_val+buffer], [min_val-buffer, max_val+buffer], 
            'k--', alpha=0.5, label='Perfect Match')
    
    # Create scatter plot
    ax.scatter(true_alpha, predicted_alpha, alpha=0.6)
    
    ax.set_xlabel('True Parameters')
    ax.set_ylabel('Predicted Parameters')
    ax.set_title('Cross-plot of True vs Predicted Parameters')
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(min_val-buffer, max_val+buffer)
    ax.set_ylim(min_val-buffer, max_val+buffer)
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_observations_vs_predictions(true_heads, predicted_heads, save_path='./figs/observations_vs_predictions.png'):
    """
    Create a 45-degree cross-plot comparing true and predicted hydraulic heads
    
    Args:
        true_heads: True hydraulic head values
        predicted_heads: Predicted hydraulic head values
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get axis limits
    min_val = min(np.min(true_heads), np.min(predicted_heads))
    max_val = max(np.max(true_heads), np.max(predicted_heads))
    buffer = (max_val - min_val) * 0.1
    
    # Plot 45 degree line
    ax.plot([min_val-buffer, max_val+buffer], [min_val-buffer, max_val+buffer], 
            'k--', alpha=0.5, label='Perfect Match')
    
    # Create scatter plot
    ax.scatter(true_heads, predicted_heads, alpha=0.6)
    
    ax.set_xlabel('True Hydraulic Heads')
    ax.set_ylabel('Predicted Hydraulic Heads') 
    ax.set_title('Cross-plot of True vs Predicted Hydraulic Heads')
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(min_val-buffer, max_val+buffer)
    ax.set_ylim(min_val-buffer, max_val+buffer)
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_head_fields(true_heads, predicted_heads, save_path='./figs/head_fields.png'):
    """
    Plot and compare true and predicted hydraulic head fields
    
    Args:
        true_heads: Array of true hydraulic head values
        predicted_heads: Array of predicted hydraulic head values
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot true head field
    # Get common colorbar range
    vmin = min(np.min(true_heads), np.min(predicted_heads))
    vmax = max(np.max(true_heads), np.max(predicted_heads))
    
    # Plot true head field
    im1 = ax1.imshow(true_heads, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title('True Hydraulic Head Field')
    
    # Plot predicted head field
    im2 = ax2.imshow(predicted_heads, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title('Predicted Hydraulic Head Field')
    
    # Add colorbars for each subplot
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Example usage
if __name__ == "__main__":
    # Example usage
    nx, ny, nz = 16, 16, 8  # Grid dimensions
    K = np.exp(np.random.rand(nx, ny, nz)-4)  # Heterogeneous permeability

    norm = LogNorm(vmin=K.min(), vmax=K.max())
    plot_3D_surface(K, "PERM", norm, "example.png")