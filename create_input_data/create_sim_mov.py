import numpy as np
import matplotlib.pyplot as plt
import mikeio
import matplotlib.animation as animation

rain_path = r'M:\phd\wps\wp1\data\generated_data\mike_sim_files\perlin\train\avg6h75_10m_noboundary\10991\10991_rain.dfs2'
terrain_path = r'M:\phd\wps\wp1\data\generated_data\mike_sim_files\perlin\train\avg6h75_10m_noboundary\10991\terrain.dfs2'
result_path = r'M:\phd\wps\wp1\data\generated_data\sim_data\perlin\train\avg6h75_10m_noboundary\Results_10991.dfs2'
rain = mikeio.open(rain_path).read().to_numpy().squeeze()
terrain = mikeio.open(terrain_path).read().to_numpy().squeeze()
height = mikeio.open(result_path).read().to_numpy().squeeze()[0]
height[np.isnan(height)] = 0

fig_height,ax_height = plt.subplots()
images = [
    [ax_height.imshow(
        layer, animated=True,origin = 'lower', vmin = 0, vmax = np.max(height.flatten()),cmap = 'Blues')]
    for layer in height
]
ax_height.set_xlabel('x[m]')
ax_height.set_ylabel('y[m]')
ax_height.set_title('Event 10991 Height')
animation_3d = animation.ArtistAnimation(fig_height, images, interval=50, blit=True)
animation_3d.save(r'C:\Users\antsor\Desktop\height_10991.gif', writer='imagemagick', fps=60)
fig_rain,ax_rain = plt.subplots()
images = [
    [ax_rain.imshow(
        layer, animated=True,origin = 'lower', vmin = 0, vmax = np.max(rain.flatten()),cmap = 'Blues')]
    for layer in rain
]

ax_rain.set_xlabel('x[m]')
ax_rain.set_ylabel('y[m]')
ax_rain.set_title('Event 10991 Rain')
# fig.colorbar()
animation_3d = animation.ArtistAnimation(fig_rain, images, interval=50, blit=True)
animation_3d.save(r'C:\Users\antsor\Desktop\rain_10991.gif', writer='imagemagick', fps=60)
