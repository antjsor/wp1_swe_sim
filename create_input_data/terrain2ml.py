import numpy as np
import mikeio
import os
import matplotlib.pyplot as plt
import numpy as np
# import perlin_numpy as pnp
import pandas as pd
from mikeio import Grid2D
do_plot = True  
save_plot = False
terrain_path = r'M:\phd\wps\wp1\data\raw\from_rolo\rain_generation\data\dem_5m_wb_wrd.dfs2'
terrain_dfs2 = mikeio.read(terrain_path)
terrain_data = terrain_dfs2.to_numpy().squeeze().transpose()

# Initialize the gradient array
terrain_grad = np.zeros((4, terrain_dfs2.geometry.nx, terrain_dfs2.geometry.ny))

dx = terrain_dfs2.geometry.dx
dy = terrain_dfs2.geometry.dy

# Compute the gradients
# For the gradient in the positive x_direction (east)
terrain_grad[0, :-1, :] = (terrain_data[1:, :] - terrain_data[:-1, :]) / dx
# For the gradient in the negative y_direction (south)
terrain_grad[1, :, 1:] = (terrain_data[:, :-1] - terrain_data[:, 1:]) / dy
# For the gradient in the negative x_direction (west)
terrain_grad[2, 1:, :] = (terrain_data[:-1, :] - terrain_data[1:, :]) / dx

# For the gradient in the positive y_direction (north)
terrain_grad[3, :, :-1] = (terrain_data[:, 1:] - terrain_data[:, :-1]) / dy

if do_plot:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    grad_titles = ['Gradient East', 'Gradient South', 'Gradient West', 'Gradient North']
    for i, ax in enumerate(axes):
        im = ax.imshow(terrain_grad[i], cmap='jet', origin='lower',)
        ax.set_title(grad_titles[i])
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_plot:
        plt.savefig('terrain_gradients.png')
    plt.show()
