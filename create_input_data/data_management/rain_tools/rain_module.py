import pandas
from datetime import datetime
import numpy as np
import datetime
import os
from perlin_noise import PerlinNoise
import mikeio
from scipy.ndimage import gaussian_filter



class RainDataProcessor:
    def __init__(self, ds_ref, rain_path,n_pad = 0):
        self.ds_ref = ds_ref
        self.n_pad = n_pad
        self.rain_path = rain_path
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.rain_df = pandas.read_csv(self.rain_path, sep=';', header=None, dtype={2: str})
        self.rain_ids = self.rain_df.iloc[:, -1].unique().tolist()
        self.rain_dict, self.rain_datetime_dict = self._load_rain_data()
        self.dt_rain = 60 ### Rewrite maybe with a function that extracts it from the rain_path
          
    ########## Rain stuff #########################
    def _load_rain_data(self):
        rain_datetime_dict = {rain_id : self.rain_df.loc[self.rain_df[2]==rain_id][0] for rain_id in self.rain_ids}
        rain_dict = {rain_id : self.rain_df.loc[self.rain_df[2]==rain_id][1].values for rain_id in self.rain_ids}
        
        return rain_dict, rain_datetime_dict
    
    def zero_boundary_rain(self,rain):
        rain[:self.n_pad, :] = 0
        rain[-(self.n_pad):, :] = 0
        rain[self.n_pad:-self.n_pad,:self.n_pad] = 0
        rain[self.n_pad:-self.n_pad,-self.n_pad:] = 0
            
        return rain

    def translate_rain_2d(self,time_series, domain, velocity_true, dt, start_dry_t=0):
        velocity = np.abs(velocity_true)
        t_rain = len(time_series) * dt - dt
        eps = 1e-10
        time_delays = np.int64(np.abs(domain[0]) / (velocity[0]+eps) + np.abs(domain[1]) / (velocity[1] + eps))
        if velocity_true[0] < 0 and velocity_true[1] < 0:
            time_delays = np.flipud(np.fliplr(time_delays))
        elif velocity_true[0] < 0:
            time_delays = np.fliplr(time_delays)
        elif velocity_true[1] < 0:
            time_delays = np.flipud(time_delays)
        rain_out = []
        # Initial dry phase
        if start_dry_t > 0:
            rain_out = [np.zeros_like(domain[0]) for _ in range(0, start_dry_t, dt)]

        # Wet loop over rain series
        for tt in range(0, t_rain + dt, dt):
            rain_tt = np.zeros_like(domain[0])
            mask = tt - time_delays > 0
            times = tt - time_delays
            rain_tt[mask] = time_series[np.int64((times[mask]) / dt)]
            self.zero_boundary_rain(rain_tt)
            rain_out.append(rain_tt)

        return rain_out
########## MAKE SEPARATE CLASS FOR LOADING THE RAIN IDs and then simply call that from the others to avoid this level of complexity
# class SimProcessor:
#     def __init__(self, ds_ref, rain_path,n_pad = 0):
#         self.ds_ref = ds_ref
#         self.n_pad = n_pad
#         self.rain_path = rain_path
#         self.date_format = '%Y-%m-%d %H:%M:%S'
#         self.rain_df = pandas.read_csv(self.rain_path, sep=';', header=None, dtype={2: str})
#         self.rain_ids = self.rain_df.iloc[:, -1].unique().tolist()
#         self.rain_dict, self.rain_datetime_dict = self._load_rain_data()
        
#         self.dt_rain = 60 ### Rewrite maybe with a function that extracts it from the rain_path
          
#     ########## Rain stuff #########################
#     def _load_rain_data(self):
#         rain_datetime_dict = {rain_id : self.rain_df.loc[self.rain_df[2]==rain_id][0] for rain_id in self.rain_ids}
#         rain_dict = {rain_id : self.rain_df.loc[self.rain_df[2]==rain_id][1].values for rain_id in self.rain_ids}
        
#         return rain_dict, rain_datetime_dict
    
#     def zero_boundary_rain(self,rain):
#         rain[:self.n_pad, :] = 0
#         rain[-(self.n_pad):, :] = 0
#         rain[self.n_pad:-self.n_pad,:self.n_pad] = 0
#         rain[self.n_pad:-self.n_pad,-self.n_pad:] = 0
            
#         return rain

#     def translate_rain_2d(self,time_series, domain, velocity_true, dt, start_dry_t=0):
#         velocity = np.abs(velocity_true)
#         t_rain = len(time_series) * dt - dt
#         eps = 1e-10
#         time_delays = np.int64(np.abs(domain[0]) / (velocity[0]+eps) + np.abs(domain[1]) / (velocity[1] + eps))
#         if velocity_true[0] < 0 and velocity_true[1] < 0:
#             time_delays = np.flipud(np.fliplr(time_delays))
#         elif velocity_true[0] < 0:
#             time_delays = np.fliplr(time_delays)
#         elif velocity_true[1] < 0:
#             time_delays = np.flipud(time_delays)
#         rain_out = []
#         # Initial dry phase
#         if start_dry_t > 0:
#             rain_out = [np.zeros_like(domain[0]) for _ in range(0, start_dry_t, dt)]

#         # Wet loop over rain series
#         for tt in range(0, t_rain + dt, dt):
#             rain_tt = np.zeros_like(domain[0])
#             mask = tt - time_delays > 0
#             times = tt - time_delays
#             rain_tt[mask] = time_series[np.int64((times[mask]) / dt)]
#             self.zero_boundary_rain(rain_tt)
#             rain_out.append(rain_tt)

#         return rain_out
    
#     ############## Terrain Stuff: #######################################
#     def lin_shift_boundary2fixed(self,terrain, boundary_value=0.1, n_shift=3):
#         height, width = terrain.shape

#         # Function to calculate the linear shift based on distance and initial value
#         def linear_shift(value, dist, max_dist):
#             return value + (boundary_value - value) * (dist / max_dist)

#         def cos_interpolate(a, b, t):
#         # Use cosine interpolation instead of linear
#             cos_t = (1 - np.cos(t * np.pi)) / 2
#             return a * (1 - cos_t) + b * cos_t
        
#         def smooth_terrain(terrain, boundary_value):
#             smoothed_terrain = gaussian_filter(terrain, sigma = 1)
#             smoothed_terrain[0,:] = boundary_value
#             smoothed_terrain[-1,:] = boundary_value
#             smoothed_terrain[:,0] = boundary_value
#             smoothed_terrain[:,-1] = boundary_value
#             return smoothed_terrain

#         # Apply linear shift to top and bottom boundaries
#         for i in range(n_shift):
#             dist_from_edge = n_shift-i
#             terrain[i, :] = linear_shift(terrain[n_shift, :], dist_from_edge, n_shift)
#             terrain[-(i + 1), :] = linear_shift(terrain[-n_shift, :], dist_from_edge, n_shift)

#         # Apply linear shift to left and right boundaries
#         for j in range(n_shift):
#             dist_from_edge = n_shift-j
#             terrain[:, j] = linear_shift(terrain[:, n_shift], dist_from_edge, n_shift)
#             terrain[:, -(j + 1)] = linear_shift(terrain[:, -n_shift], dist_from_edge, n_shift)
        
        
#         return smooth_terrain(terrain, boundary_value)


#     def generate_perlin_terrain(self,mean = 1,var = 1,boundary_value = 0,n_shift = 10):
#         perlin_generator = PerlinNoise(octaves=8)
#         grid    = self.ds_ref.geometry
#         ds_time = self.ds_ref.time
#         resolution = grid.nx
#         dx = grid.dx
#         xpix, ypix = grid.nx, grid.ny
#         raster = np.array([[perlin_generator([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)])
#         vect = np.random.uniform(-1,1, 2)
#         direction_vector = vect/np.sqrt(sum(element**2 for element in vect))  # This example uses the x-axis; change as needed
#         y_indices, x_indices = np.indices(raster.shape[:2])*dx
#         drop = 2*np.random.rand(1) # total drop one one end to the next (bit more since slope is always slightly diagonal)
#         slope = drop/(128*dx)
#         projection_lengths = x_indices * direction_vector[0] + y_indices * direction_vector[1]
#         raster_transformed = raster + slope * projection_lengths
#         raster_transformed *= var
#         raster_transformed += np.abs(np.min(raster_transformed))+mean
        
#         return self.lin_shift_boundary2fixed(raster_transformed,boundary_value,n_shift)

#     # def terrain2dfs(self):
#     #     da_terrain = mikeio.DataArray(data = Lin_Shift_Boundary2Fixed(raster_transformed,boundary_value,n_shift),
#     #                                     geometry=grid,
#     #                                     item= mikeio.ItemInfo(name = 'perlin', itemtype = mikeio.EUMType.Bathymetry, unit = mikeio.EUMUnit.meter))
#     #     ds_terrain = mikeio.Dataset([da_terrain],geometry=grid)
#     #     ds_terrain.to_dfs(os.path.join(sim_path,"terrain.dfs2"))
    
#     def create_sim(self, batch_ids, start_dry_time, velocities = None)

#         ### Check if velocities and batch ids are same length:
#         grid    = self.ds_ref.geometry
#         x    = np.linspace(0,int(grid.dx*grid.nx-2),num = grid.nx)
#         y    = np.linspace(0,int(grid.dy*grid.ny-2),num = grid.ny)
#         X,Y  = np.meshgrid(x,y)
        
#         for rain_idx, rain_id in enumerate(batch_ids):
#             print(40*'#')
#             print(f'Rain id: {rain_id} ({rain_idx+1}/{len(batch_ids)})')
#             print(rain_id)
#             rain_series   = self.rain_dict[rain_id]
#             velocity = velocities[rain_idx] if velocities else np.random.uniform(0.001,10.0, 2)
#             rain_out      = self.translate_rain_2d(rain_series, [X,Y], velocity, dt=self.dt_rain, n_pad=n_pad, start_dry_t=start_dry_time)
#             rain_stack    = np.stack(rain_out)
            
#             ###### Something to do with managing the time for the simulations
#             rain_datetime = self.rain_datetime_dict[rain_id].to_list()
#             rain_t_zero   = rain_datetime[0]
#             rain_t_end    = rain_datetime[-1]  
#             rain_t_0_datetime   = datetime.strptime(rain_t_zero, date_format)
#             rain_t_end_datetime = datetime.strptime(rain_t_end, date_format)
#             rain_time = (rain_t_end_datetime - rain_t_0_datetime).seconds + start_dry_time
            
            
#             for time in range(self.dt_rain,start_dry_time + self.dt_rain, self.dt_rain):
#                 rain_datetime.append((rain_t_end_datetime + timedelta(seconds = time)).strftime(date_format))
#             ##### End of doing something with time of simulations #########################
            
            
#             # if export2mike:
#             da_rain = mikeio.DataArray(data =rain_stack * 60 * 24, # 
#                                     geometry=grid,
#                                     time = rain_datetime,
#                                     item= mikeio.ItemInfo(f'Surface rain for time series: {rain_id}', mikeio.EUMType.Precipitation_Rate, mikeio.EUMUnit.millimeter_per_day)
#                                     ) 
            
#             sim_path = os.path.join(mike_sim_path, str(int(rain_id)))
#             os.makedirs(sim_path, exist_ok=True)
#             da_rain.to_dfs(os.path.join(sim_path,f'{rain_id}_rain.dfs2'))
#             ### Generate terrain
#             terrain_tmp = self.generate_perlin_terrain(ds_ref, sim_path,1,1,boundary_value, 10)
            
#             ### Modify simulation file###################
#                 placeholders = {
#                     'Number_Of_Timesteps = 10200': f'Number_Of_Timesteps = {int(np.ceil(rain_time/dt_sim))}',
#                     'Last_Time_Step = 10200': f'Number_Of_Timesteps = {int(np.ceil(rain_time/dt_sim))}',
#                     'effrain.dfs2': f'{rain_id}_rain.dfs2',
#                     'Start_Time = 2002, 8, 2, 10, 56, 0' : f'Start_Time = {rain_t_0_datetime.year}, {rain_t_0_datetime.month}, {rain_t_0_datetime.day}, {rain_t_0_datetime.hour}, {rain_t_0_datetime.minute}, {rain_t_0_datetime.second}',
#                     'X_Range_And_Interval = 2, 128, 1': f'X_Range_And_Interval = {n_pad}, {128-n_pad}, 1',
#                     'Y_Range_And_Interval = 2, 128, 1': f'Y_Range_And_Interval = {n_pad}, {128-n_pad}, 1',
#                     # ' Value = 0': f' Value = {boundary_value+0.0}'
#                     # 'Results.dfs2':f'{sim_result_path}\Results_{rain_id}.dfs2'
#                     }
#                 # Everything is copied into the lines list since iterating directly on the input_file will cause exhaustion
#                 with open(m21template, 'r') as input_file, open(os.path.join(sim_path, 'Simulation.m21'), 'w') as output_file:
#                     lines = input_file.readlines()
#                     for placeholder, replacement in placeholders.items():
#                         replaced = False
#                         for idx, line in enumerate(lines):
#                             if placeholder in line:
#                                 if do_print:
#                                     print(f'Found {placeholder} in file')
#                                 lines[idx] = line.replace(placeholder, replacement)
#                                 replaced = True 
#                         if not replaced and do_print:
#                             print(f"No instance of: {placeholder} found in file ")
#                     output_file.writelines(lines)
                    
    # def export2hdf(self,data):
    #         rain2hdf = np.zeros([data.shape[0], data.shape[1]-2*n_pad, data.shape[2]-2*n_pad])
    #         print('rainhdf',rain2hdf.shape)
    #         for idx, rain_slice in enumerate(data):
    #             rain2hdf[idx] = rain_slice[n_pad:-n_pad, n_pad:-n_pad]
            
    #         with h5py.File(os.path.join(ml_data_path,hdf_name), "a") as hf:
    #             hf.create_dataset(str(rain_id), data=np.float32(rain2hdf))
        