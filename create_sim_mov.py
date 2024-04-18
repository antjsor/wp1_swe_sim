import numpy as np
import matplotlib.pyplot as plt
import mikeio
import matplotlib.animation as animation

rain_path = r'M:\phd\wps\wp1\data\generated_data\mike_sim_files\perlin\train\avg6h75_10m_noboundary\9506\9506_rain.dfs2'
terrain_path = r'M:\phd\wps\wp1\data\generated_data\mike_sim_files\perlin\train\avg6h75_10m_noboundary\9506\terrain.dfs2'
result_path = r'M:\phd\wps\wp1\data\generated_data\sim_data\perlin\train\avg6h75_10m_noboundary\Results_9506.dfs2'
rain = mikeio.open(rain_path).read().to_numpy().squeeze()
terrain = mikeio.open(terrain_path).read().to_numpy().squeeze()
height = mikeio.open(result_path).read().to_numpy().squeeze()[0]
height[np.isnan(height)] = 0

fig,ax = plt.subplots()
# images = [
#     [ax.imshow(
#         layer, animated=True,origin = 'lower', vmin = 0, vmax = np.max(height.flatten()),cmap = 'Blues')]
#     for layer in height
# ]

images = [
    [ax.imshow(
        layer, animated=True,origin = 'lower', vmin = 0, vmax = np.max(rain.flatten()),cmap = 'Blues')]
    for layer in rain
]

ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
# fig.colorbar()
animation_3d = animation.ArtistAnimation(fig, images, interval=50, blit=True)
animation_3d.save(r'C:\Users\antsor\Desktop\rain_9506.gif', writer='imagemagick', fps=60)
plt.show()