import matplotlib.pyplot as plt
import numpy as np
from perlin_numpy import generate_perlin_noise_2d
import perlin_noise as pn

# np.random.seed(0)
# np.random.seed(6)


terrain_full = np.zeros(34,34)

terrain_full[1:-1,1:-1] = generate_perlin_noise_2d((32, 32), (4, 4))
plt.imshow(noise, cmap = 'terrain')

# noise = generate_perlin_noise_2d((32, 32), (4, 4))
# plt.imshow(noise, cmap = 'terrain')

# dx = int(2)
# noise_generator = pn.PerlinNoise(octaves=5, seed=1312)
# x = np.linspace(0,1022,num = dx)
# X,Y      = np.meshgrid(x,x)
# # Generator transposes somehow.
# terrain = np.array([[noise_generator([i/X, j/Y]) for j in range(len(x))] for i in range(len(x))]).transpose()

# terrain = (rain + np.abs(np.min(rain)))/np.abs(np.sum(rain))
# plt.imshow(terrain, cmap = 'terrain')


# plt.colorbar()
plt.axis('off')
plt.show()
# datafolder = os.path.abspath(r'M:\phd\wps\wp1\code\python\rolo_rain_generation_mod\data')
# output_folder  = os.path.abspath(r'M:\phd\wps\wp1\data\rain_data\mike\15-08-23')

# fdfsref    = os.path.join(datafolder,'dem_5m_wb_wrd_512.dfs2') #DEM used for getting a spatial reference for the rain files to create



# ## For real rain series

# ds_ref     = mikeio.read(fdfsref)
# grid       = ds_ref.geometry
# ds_time    = ds_ref.time
# terrain = np.zeros_like(X)




# X,Y      = np.meshgrid(x,x)

# da_terrain = mikeio.DataArray(data =terrain,
#                         geometry=grid,
#                         time = ds_time,
#                         item= mikeio.ItemInfo("Example", mikeio.EUMType.Elevation))

# ds_terrain = mikeio.Dataset([da_terrain])

# # ds_rain.to_dfs("case10_rain.dfs2")
# ds_terrain.to_dfs("flat_terrain.dfs2")
# #######################################

# noise = generate_perlin_noise_2d((32, 32), (4, 4))
# plt.imshow(noise, cmap = 'terrain')

# dx = int(2)
# noise_generator = pn.PerlinNoise(octaves=5, seed=1312)
# x = np.linspace(0,1022,num = dx)
# X,Y      = np.meshgrid(x,x)
# # Generator transposes somehow.
# terrain = np.array([[noise_generator([i/X, j/Y]) for j in range(len(x))] for i in range(len(x))]).transpose()

# terrain = (rain + np.abs(np.min(rain)))/np.abs(np.sum(rain))
# plt.imshow(terrain, cmap = 'terrain')


# plt.colorbar()
plt.axis('off')
plt.show()
# datafolder = os.path.abspath(r'M:\phd\wps\wp1\code\python\rolo_rain_generation_mod\data')
# output_folder  = os.path.abspath(r'M:\phd\wps\wp1\data\rain_data\mike\15-08-23')

# fdfsref    = os.path.join(datafolder,'dem_5m_wb_wrd_512.dfs2') #DEM used for getting a spatial reference for the rain files to create



# ## For real rain series

# ds_ref     = mikeio.read(fdfsref)
# grid       = ds_ref.geometry
# ds_time    = ds_ref.time
# terrain = np.zeros_like(X)




# X,Y      = np.meshgrid(x,x)

# da_terrain = mikeio.DataArray(data =terrain,
#                         geometry=grid,
#                         time = ds_time,
#                         item= mikeio.ItemInfo("Example", mikeio.EUMType.Elevation))

# ds_terrain = mikeio.Dataset([da_terrain])

# # ds_rain.to_dfs("case10_rain.dfs2")
# ds_terrain.to_dfs("flat_terrain.dfs2")
# #######################################