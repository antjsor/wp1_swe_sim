import numpy as np
import datetime
import os
from perlin_noise import PerlinNoise
import mikeio

# Function to apply Gaussian smoothing
from scipy.ndimage import gaussian_filter
def Shift_Boundary_4_MIKE(terrain,n_shift = 2, boundary_shift = 0.1):
    terrain[:n_shift, :] -= boundary_shift
    terrain[-(n_shift):, :] -= boundary_shift
    terrain[n_shift:-n_shift,:n_shift] -= boundary_shift
    terrain[n_shift:-n_shift,-n_shift:] -= boundary_shift
    
    return terrain

def Lin_Shift_Boundary(terrain, n_shift=3, boundary_shift=0.1):
    height, width = terrain.shape

    # Linearly change the top and bottom boundaries
    for i in range(1, n_shift + 1):
        change = boundary_shift * (n_shift - (i - 1)) / n_shift
        terrain[i - 1, :] -= change
        terrain[-i, :] -= change

    # Linearly change the left and right boundaries
    for j in range(1, n_shift + 1):
        change = boundary_shift * (n_shift - (j - 1)) / n_shift
        terrain[:, j - 1] -= change
        terrain[:, -j] -= change

    return terrain

def lin_slope_terrain2cst(terrain, boundary_value=0.1, n_cst=3, n_slope=3):
        
    def linear_shift(value, dist, max_dist):
        return value + (boundary_value - value) * (dist / max_dist)
    
    def smooth_terrain(terrain, boundary_value, n_cst):
        ### TODO have to apply boundary value twice, to not get artifacts
        ### And must do afterwards to ensure that boundary is set correct
        ### for MIKE
        terrain[:n_cst, :] = boundary_value
        terrain[-n_cst:, :] = boundary_value
        terrain[:, :n_cst] = boundary_value
        terrain[:, -n_cst:] = boundary_value
        terrain = gaussian_filter(terrain, sigma=1)
        # Apply the boundary value to a perimeter of width `n_cst`
        terrain[:n_cst, :] = boundary_value
        terrain[-n_cst:, :] = boundary_value
        terrain[:, :n_cst] = boundary_value
        terrain[:, -n_cst:] = boundary_value
        return terrain
    
    # Apply the smooth transition to the original terrain
    # Start from n_cst to create a slope over the next n_slope cells
    
    ## Problem: Det er fordi at når vi gør det
    for i in range(n_cst, n_cst + n_slope):
        dist_from_edge = n_slope+n_cst-i

        terrain[i, :] = linear_shift(terrain[n_cst + n_slope, :], dist_from_edge, n_slope)
        terrain[-(i + 1), :] = linear_shift(terrain[-(n_cst + n_slope), :], dist_from_edge, n_slope)

    for j in range(n_cst, n_cst + n_slope):
        dist_from_edge = n_slope+n_cst-j
        terrain[:, j] = linear_shift(terrain[:, n_cst + n_slope], dist_from_edge, n_slope)
        terrain[:, -(j + 1)] = linear_shift(terrain[:, -(n_cst + n_slope)], dist_from_edge, n_slope)
    
    # Apply smoothing to the terrain
    terrain = smooth_terrain(terrain, boundary_value, n_cst)

    return terrain



def Lin_Shift_Boundary2Fixed(terrain,boundary_value=0.1, n_shift=3):
    # Function to calculate the linear shift based on distance and initial value
    def linear_shift(value, dist, max_dist):
        return value + (boundary_value - value) * (dist / max_dist)
    
    from scipy.ndimage import gaussian_filter
    
    def smooth_terrain(terrain, boundary_value):

        smoothed_terrain = gaussian_filter(terrain, sigma = 1)
        smoothed_terrain[:int(n_shift/2),:] = boundary_value
        smoothed_terrain[-int(n_shift/2):,:] = boundary_value
        smoothed_terrain[:,:int(n_shift/2)] = boundary_value
        smoothed_terrain[:,-int(n_shift/2):] = boundary_value
        return smoothed_terrain
    
    # Apply linear shift to top and bottom boundaries
    for i in range(n_shift):
        dist_from_edge = n_shift-i
        terrain[i, :] = linear_shift(terrain[n_shift, :], dist_from_edge, n_shift)
        terrain[-(i + 1), :] = linear_shift(terrain[-n_shift, :], dist_from_edge, n_shift)

    for j in range(n_shift):
        dist_from_edge = n_shift-j
        terrain[:, j] = linear_shift(terrain[:, n_shift], dist_from_edge, n_shift)
        terrain[:, -(j + 1)] = linear_shift(terrain[:, -n_shift], dist_from_edge, n_shift)
    
    return smooth_terrain(terrain, boundary_value)


def Generate_Perlin_Terrain(ds_ref, sim_path,min_value = 1,var = 1,boundary_value = 0,n_shift = 10,n_slope = 10):
    perlin_generator = PerlinNoise(octaves=8)
    grid    = ds_ref.geometry
    ds_time = ds_ref.time
    resolution = grid.nx
    dx = grid.dx
    xpix, ypix = grid.nx, grid.ny
    raster = np.array([[perlin_generator([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)],dtype = np.float32)
    vect = np.random.uniform(-1,1, 2)
    direction_vector = vect/np.sqrt(sum(element**2 for element in vect))  # This example uses the x-axis; change as needed
    y_indices, x_indices = np.indices(raster.shape[:2])*dx
    drop = 2*np.random.rand(1) # total drop one one end to the next (bit more since slope is always slightly diagonal)
    slope = drop/(128*dx)
    projection_lengths = x_indices * direction_vector[0] + y_indices * direction_vector[1]
    raster += slope * projection_lengths
    raster *= var
    raster += np.abs(np.min(raster))+min_value

    return lin_slope_terrain2cst(raster,boundary_value,n_shift,n_slope)
    # da_terrain = mikeio.DataArray(data = Lin_Shift_Boundary2Fixed(raster_transformed,boundary_value,n_shift),
    #                                 geometry=grid,
    #                                 item= mikeio.ItemInfo(name = 'perlin', itemtype = mikeio.EUMType.Bathymetry, unit = mikeio.EUMUnit.meter))
    # ds_terrain = mikeio.Dataset([da_terrain],geometry=grid)
    # ds_terrain.to_dfs(os.path.join(sim_path,"terrain.dfs2"))
    

def create_real_terrain(domain, direction, slope):
    print('Not implemented yet')
    # return 
    

from mikecore.DfsFileFactory import DfsFileFactory  # type: ignore
import shutil

def terrain2mike(terrain : np.ndarray,fdfsref,sim_path,name = r'\terrain.dfs2'):
    out_name = shutil.copy(fdfsref,sim_path + name)
    dfsout=DfsFileFactory.Dfs2FileOpenEdit(out_name)
    dfsout.WriteItemTimeStep(1,0,0,terrain)
    dfsout.Close()