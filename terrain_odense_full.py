import numpy as np
import mikeio
import os
import matplotlib.pyplot as plt
import numpy as np
# import perlin_numpy as pnp
import pandas as pd
from data_management.terrain_tools import Shift_Boundary_4_MIKE
from mikeio import Grid2D

do_plot = True  
save_plot = False
export2mike = True
terrain_type = 'flat'
n_pad = 2
if do_plot:
    plot_path = os.path.abspath(r'M:\phd\supervisor_meetings\presentations\22_09_22\figs')
datafolder     = os.path.abspath(r'M:\phd\wps\wp1\data\generated_data\terrain_data\real')
output_folder  = os.path.abspath(r'M:\phd\wps\wp1\data\generated_data\mike_simulations_test\flat')
fdfsref        = os.path.join(datafolder,r'dem_5m_wb_wrd_crop.dfs2') #DEM used for getting a spatial reference for the rain files to create
if export2mike:
        ds_ref  = mikeio.read(fdfsref)
        grid    = ds_ref.geometry
        ds_time = ds_ref.time


# odense_np = Shift_Boundary_4_MIKE(ds_ref.to_numpy().squeeze(), n_shift=200, boundary_shift=0.0)
odense_np = ds_ref.to_numpy().squeeze()
odense_shift = Shift_Boundary_4_MIKE(odense_np, n_shift=2, boundary_shift=0.2)+0.2
if export2mike:
    # da_terrain = mikeio.DataArray(data = Shift_Boundary_4_MIKE(terrain,2,0.1),
    da_terrain = mikeio.DataArray(data = odense_shift,
                                    geometry=grid,
                                    item= mikeio.ItemInfo(name = terrain_type, itemtype = mikeio.EUMType.Bathymetry, unit = mikeio.EUMUnit.meter))
    # da_terrain = ds_ref.create_data_array(data = terrain)
    ds_terrain = mikeio.Dataset([da_terrain],geometry=grid)
    ds_terrain.to_dfs(os.path.join(output_folder,f"odense_terrain_x{int(grid.nx)}_y{int(grid.ny )}.dfs2"))