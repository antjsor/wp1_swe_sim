import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mikeio
import torch
import h5py
from datetime import datetime
from datetime import timedelta
import pandas as pd
import os
import re
from data_management import Load_Rain, Translate_Rain_2D

from data_management import Generate_Perlin_Terrain, terrain2mike
import shutil
# current_drive     = os.path.splitdrive(sys.prefix)[0]
data_type = r'train' # train, val or test
# print(current_drive)
# shutil.copyfile() # Backup hvis det andet ikke virker

rain_path       =  r'M:\phd\wps\wp1\data\raw\from_rolo\rainseries\obs_events_marked_' + data_type.upper() + r'.csv'
fdfsref         = os.path.abspath(r'M:\phd\wps\wp1\data\generated_data\mike_templates\terrain\odense_bathy_76_30m.dfs2') #DEM used for getting a spatial reference for the rain files to create
m21template     = os.path.abspath(r'm:\phd\wps\wp1\data\generated_data\mike_templates\sim\Simulation_perlin_4_dmdc.m21') # Change to f current_drive for real run
mike_sim_path   = os.path.abspath(r'M:\phd\wps\wp1\data\generated_data\mike_sim_files\perlin_dmdc' + fr'\{data_type}') # Path where to save mike sim files
sim_result_path = os.path.abspath(r'F:\Data\antsor\wp1\data\generated_data\sim_data\perlin_dmdc' + fr'\{data_type}')
ml_data_path    = os.path.abspath(r'C:\phd_project\wp1\data_tmp')
batch_path      = os.path.abspath(r'M:\phd\wps\wp1\data\generated_data\mike_sim_files\perlin\train\sim_batches')
batch_name      = 'avg6h75.txt'
with open(os.path.join(batch_path,batch_name), 'r') as file:
    # This will read each line, strip it of whitespace, and convert to an integer
    batch_ids = [str(line.strip()) for line in file if line.strip()]
hdf_name = fr'rain_{os.path.splitext(batch_name)[0]}.hdf5'

make_movie  = False
export2mike = True
export2hdf  = False
do_plot     = False
do_print = False
df_velocity = pd.read_csv(r'M:\phd\wps\wp1\code\python\data_generation\misc_files\16_v_vectors.csv', index_col=0,dtype={'rain_id': str})
# np.random.seed(666)

if do_plot:
    plot_path = os.path.abspath(r'M:\phd\wps\wp1\data\generated_data\rain_data\mike\23-08-31')

## For real rain series
dt_rain = 60
start_dry_time = 10*dt_rain
vmax = 10

with open(m21template, 'r') as f:
    for line in f:
        match = re.search(r"Time_Step_Interval\s*=\s*([\d.]+)", line)
        if match:
            dt_sim = float(match.group(1))
            
if export2hdf:
    # Check if the file exists
    file_path = os.path.join(ml_data_path, hdf_name)
    if not(os.path.isfile(file_path)):
        print(f'No file with name "{file_path}", creating new file...')
        with h5py.File(os.path.join(ml_data_path,hdf_name), 'w') as f:
            pass 
    else:
        print(f'File "{file_path}"already exists, will append data sets to this.')

ds_ref     = mikeio.read(fdfsref)
grid       = ds_ref.geometry

boundary_value = 0.1
date_format = '%Y-%m-%d %H:%M:%S'
n_cst = 10 # Change according to the number of cells padded on boundary for mike
n_slope = 10
x    = np.linspace(0,int(grid.dx*grid.nx-2),num = grid.nx)
y    = np.linspace(0,int(grid.dy*grid.ny-2),num = grid.ny)

X,Y  = np.meshgrid(x,y)
rain_dict, rain_ids, rain_datetime_dict,_ = Load_Rain(rain_path)
data = dict()

for rain_idx, rain_id in enumerate(batch_ids):

    print(40*'#')
    print(f'Rain id: {rain_id} ({rain_idx+1}/{len(batch_ids)})')
    print(rain_id)
    rain_series   = rain_dict[rain_id]
    
    velocity = np.random.uniform(0.001,10.0, 2)
    rain_out      = Translate_Rain_2D(rain_series, [X,Y], velocity, dt=dt_rain, n_cst=n_cst, n_slope = n_slope,start_dry_t=start_dry_time)
    rain_stack    = np.stack(rain_out)
    rain_datetime = rain_datetime_dict[rain_id].to_list()
    rain_t_zero   = rain_datetime[0]
    rain_t_end    = rain_datetime[-1]  
    rain_t_0_datetime   = datetime.strptime(rain_t_zero, date_format)
    rain_t_end_datetime = datetime.strptime(rain_t_end, date_format)
    rain_time = (rain_t_end_datetime - rain_t_0_datetime).seconds + start_dry_time
    
    for time in range(dt_rain,start_dry_time + dt_rain, dt_rain):
        rain_datetime.append((rain_t_end_datetime + timedelta(seconds = time)).strftime(date_format))
    
    if export2mike:
        da_rain = mikeio.DataArray(data =rain_stack * 60 * 24, # 
                                geometry=grid,
                                time = rain_datetime,
                                item= mikeio.ItemInfo(f'Surface rain for time series: {rain_id}', mikeio.EUMType.Precipitation_Rate, mikeio.EUMUnit.millimeter_per_day)
                                ) 
        
        sim_path = os.path.join(mike_sim_path, str(int(rain_id)))
        os.makedirs(sim_path, exist_ok=True)
        da_rain.to_dfs(os.path.join(sim_path,f'{rain_id}_rain.dfs2'))
        
        # TODO Make a terrain class, that contains terrain2mike and terrain2hdf
        terrain = Generate_Perlin_Terrain(ds_ref, sim_path,1.1,1,boundary_value, n_cst,n_slope = 10)
        terrain2mike(terrain,fdfsref,sim_path)
        
        # Create a function for these placeholders
        placeholders = {
            'Number_Of_Timesteps = 10200': f'Number_Of_Timesteps = {int(np.ceil(rain_time/dt_sim))}',
            'Last_Time_Step = 10200': f'Number_Of_Timesteps = {int(np.ceil(rain_time/dt_sim))}',
            'effrain.dfs2': f'{rain_id}_rain.dfs2',
            'Start_Time = 2002, 8, 2, 10, 56, 0' : f'Start_Time = {rain_t_0_datetime.year}, {rain_t_0_datetime.month}, {rain_t_0_datetime.day}, {rain_t_0_datetime.hour}, {rain_t_0_datetime.minute}, {rain_t_0_datetime.second}',
            'X_Range_And_Interval = 0, 147, 1': f'X_Range_And_Interval = {int(2*n_cst+2)}, {int(grid.nx-2*n_cst-2-1)}, 1',
            'Y_Range_And_Interval = 0, 147, 1': f'Y_Range_And_Interval = {int(2*n_cst+2)}, {int(grid.ny-2*n_cst-2-1)}, 1',
            ' Value = 0': f' Value = {boundary_value+0.01}',
            'Results.dfs2':f'{sim_result_path}\Results_{rain_id}.dfs2'
            }
        # Everything is copied into the lines list since iterating directly on the input_file will cause exhaustion
        with open(m21template, 'r') as input_file, open(os.path.join(sim_path, 'Simulation.m21'), 'w') as output_file:
            lines = input_file.readlines()
            for placeholder, replacement in placeholders.items():
                replaced = False
                for idx, line in enumerate(lines):
                    if placeholder in line:
                        if do_print:
                            print(f'Found {placeholder} in file')
                        lines[idx] = line.replace(placeholder, replacement)
                        replaced = True 
                if not replaced and do_print:
                    print(f"No instance of: {placeholder} found in file ")
            output_file.writelines(lines)
    if export2hdf:
        rain2hdf = np.zeros([rain_stack.shape[0], rain_stack.shape[1]-2*n_cst-4, rain_stack.shape[2]-2*n_cst-4])
        print('rainhdf',rain2hdf.shape)
        for idx, rain_slice in enumerate(rain_stack):
            rain2hdf[idx] = rain_slice[n_cst+2:-n_cst-2, n_cst+2:-n_cst-2]
        
        with h5py.File(os.path.join(ml_data_path,hdf_name), "a") as hf:
            hf.create_dataset(str(rain_id), data=np.float32(rain2hdf))
              
if do_plot:
 for idx, img in enumerate(rain_out[10:15]):
        
        fig,ax = plt.subplots()
        ax.imshow(img, cmap='Blues',origin = 'lower')
        ax.tick_params(which='both', size=0)  # Setting size=0 to hide the tick marks but keep the grid
        ax.minorticks_on()
        # Add grids for both major and minor ticks
        ax.grid(which='major', color=0.3*np.array([1,1,1]), linestyle='-', linewidth=0.5)
        ax.grid(which='minor', color=0.3*np.array([1,1,1]), linestyle='-', linewidth=0.5)
        # Tight layout and save
        plt.tight_layout()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.savefig(os.path.join(plot_path,f"rain_64_{idx}.png"), bbox_inches='tight', pad_inches=0)
        plt.savefig(os.path.join(plot_path,f"rain_64_{idx}.svg"), bbox_inches='tight', pad_inches=0)

if make_movie:
    fig,ax = plt.subplots()
    images = [
        [ax.imshow(
            layer, animated=True,origin = 'lower', vmin = 0, vmax = np.max(rain_series))]
        for layer in rain_out
    ]
    ax.set_title(f'Velocity = {velocity}')
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    # fig.colorbar()
    animation_3d = animation.ArtistAnimation(fig, images, interval=100, blit=True)
    animation_3d.save(r'C:\Users\antsor\Desktop\rain_terrain.gif', writer='imagemagick', fps=60)
    plt.show()
