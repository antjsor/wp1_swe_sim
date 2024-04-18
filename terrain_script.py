import numpy as np
import mikeio
import os
import matplotlib.pyplot as plt
import numpy as np
# import perlin_numpy as pnp
import pandas as pd
from data_management.terrain_tools import Shift_Boundary_4_MIKE
from mikeio import Grid2D
from perlin_noise import PerlinNoise
do_plot = True  
save_plot = False
export2mike = True
terrain_type = 'perlin'

if do_plot:
    plot_path = os.path.abspath(r'M:\phd\supervisor_meetings\presentations\22_09_22\figs')
datafolder     = os.path.abspath(r'M:\phd\wps\wp1\data\raw\from_rolo\rain_generation\data')
output_folder  = os.path.abspath(r'M:\phd\wps\wp1\data\generated_data\mike_simulations_test\flat')
mike_sim_path   = os.path.abspath(r'M:\phd\wps\wp1\data\generated_data\mike_sim_files\perlin\train')
fdfsref        = os.path.join(datafolder,r'M:\phd\wps\wp1\data\raw\from_rolo\rain_generation\data\odense_bathy_132.dfs2') #DEM used for getting a spatial reference for the rain files to create
batch_path      = os.path.abspath(r'M:\phd\wps\wp1\data\generated_data\mike_sim_files\flat\train\sim_batches')
batch_name      = 'max15_30min_avg.txt'
with open(os.path.join(batch_path,batch_name), 'r') as file:
    # This will read each line, strip it of whitespace, and convert to an integer
    batch_ids = [str(line.strip()) for line in file if line.strip()]
n_pad = 2
if export2mike:
        ds_ref  = mikeio.read(fdfsref)
        grid    = ds_ref.geometry
        ds_time = ds_ref.time
        # ds_ref.plot()
        # ds_ref.geometry.nx = ds_ref.geometry.nx + n_pad

resolution = grid.nx
dx = grid.dx
perlin_generator = PerlinNoise(octaves=8)

match terrain_type:
    case 'perlin':
        xpix, ypix = grid.nx, grid.ny
        for batch_id in batch_ids:
            print(20*'#' + f' Batch ID: {batch_id} ' + 20*'#')
            raster = np.array([[perlin_generator([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)])
            vect = np.random.uniform(-1,1, 2)
            direction_vector = vect/np.sqrt(sum(element**2 for element in vect))  # This example uses the x-axis; change as needed
            y_indices, x_indices = np.indices(raster.shape[:2])*dx
            drop = 2*np.random.rand(1)
            slope = drop/(128*dx)  # Change as needed
            projection_lengths = x_indices * direction_vector[0] + y_indices * direction_vector[1]
            raster_transformed = raster + slope * projection_lengths
            raster_transformed += np.abs(np.min(raster_transformed)) + 0.1
            if export2mike:
                # da_terrain = mikeio.DataArray(data = Shift_Boundary_4_MIKE(terrain,2,0.1),
                da_terrain = mikeio.DataArray(data = Shift_Boundary_4_MIKE(raster_transformed,2,0.1),
                                                geometry=grid,
                                                item= mikeio.ItemInfo(name = terrain_type, itemtype = mikeio.EUMType.Bathymetry, unit = mikeio.EUMUnit.meter))
                # da_terrain = ds_ref.create_data_array(data = terrain)
                ds_terrain = mikeio.Dataset([da_terrain],geometry=grid)
                ds_terrain.to_dfs(os.path.join(output_folder,f"terrain_{int(resolution)}_{batch_id}.dfs2"))

# if do_plot:
#     for idx, img in enumerate(terrains):
        
#         fig,ax = plt.subplots()
#         ax.imshow(img, cmap='terrain')
        
        
#         ax.tick_params(which='both', size=0)
#         ax.minorticks_on()
#         ax.grid(which='major', color=0.3*np.array([1,1,1]), linestyle='-', linewidth=0.5)
#         ax.grid(which='minor', color=0.3*np.array([1,1,1]), linestyle='-', linewidth=0.5)
#         plt.tight_layout()
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
# if save_plot:
#     plt.savefig(os.path.join(plot_path,f"terrain_{resolution}_{idx}.png"), bbox_inches='tight', pad_inches=0)
#     plt.savefig(os.path.join(plot_path,f"terrain_{resolution}_{idx}.svg"), bbox_inches='tight', pad_inches=0)
#     plt.show()
#     plt.close()


# #######################################

