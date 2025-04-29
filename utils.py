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


# Example usage
if __name__ == "__main__":
    # Example usage
    nx, ny, nz = 16, 16, 8  # Grid dimensions
    K = np.exp(np.random.rand(nx, ny, nz)-4)  # Heterogeneous permeability

    norm = LogNorm(vmin=K.min(), vmax=K.max())
    plot_3D_surface(K, "PERM", norm, "example.png")