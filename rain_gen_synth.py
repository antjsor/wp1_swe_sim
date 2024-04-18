# from rain_utils.generate_rain import time_series_2_terrain as ts2t

import data_utils.generate_rain as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mikeio
import os
import pandas as pd
import pickle

rain_path  =  r'M:\phd\wps\wp1\data\received_data\rain_series\rainseries\obs_events_marked_TEST.csv'
time_series_path = os.path.abspath(r'M:\phd\wps\wp1\code\python\rolo_rain_generation_mod\data')
mike_data_path  = os.path.abspath(r'M:\phd\wps\wp1\data\generated_data\rain_data\mike\16-08-23')
pickle_path       = os.path.abspath(r'M:\phd\wps\wp1\data\generated_data\rain_data\pickle')
fdfsref    = os.path.join(time_series_path,'dem_5m_wb_wrd_512.dfs2') #DEM used for getting a spatial reference for the rain files to create
make_movie = True
export2mike = False
# export2pandas = True
export2pickle = False
# mikeio.eum.EUMUnit.second


### Synthetic data
n_samples   = 1
dt          = 1
x           = np.linspace(0,15,num = 64)
# velocity = np.array([2,1]).flatten() # Translates as expected 15-08-2023
# velocity = np.array([-2,-1]).flatten()# Translates as expected 15-08-2023
# velocity = np.array([1,-2]).flatten()# Translates as expected 15-08-2023
# velocity = np.array([-1,2]).flatten()# Translates as expected 15-08-2023
# x = np.linspace(0,1022,num = 512)

X,Y      = np.meshgrid(x,x)
ds_ref     = mikeio.read(fdfsref)
grid       = ds_ref.geometry
ds_time    = ds_ref.time
terrain = np.zeros_like(X)

velocities = []
data       = dict()
for ii in range(n_samples):
    rain_series  = np.abs(np.cumsum((np.random.normal(0,1,size = 50))))    
    # rain_series  = np.abs(np.cumsum((np.random.normal(0,1,size = 30))))    

    # velocity = np.random.uniform(-2,2,size = 2)
    velocity = np.array([2,1]).flatten()

    velocities.append(velocity)

    rain_out    = gr.time_series_2_terrain(rain_series, [X,Y], velocity, dt=dt)
    rain_stack  = np.stack(rain_out)
    # time_steps  = rain_stack.shape[0] # Perhaps not necessary to do everytime
    
    if export2pickle:
        # data_frame[str(rain_id)]  = {'Rain':rain_stack, 'Time':rain_datetime, 'Velocity':velocity}
        data[str(ii)] = {'Rain':rain_stack, 'time_series' : rain_series, 'velocity':velocity}

plt.plot()
with open(os.path.join(pickle_path,f'rain_data_synth.pkl'), 'wb') as f:
    pickle.dump(data, f)

if make_movie:
    fig,ax = plt.subplots()
    images = [
        [ax.imshow(
            layer, animated=True,origin = 'lower', vmin = 0, vmax = np.max(rain_series),cmap = 'Blues')]
        for layer in rain_out
    ]
    # ax.set_title(f'Velocity = {velocity}')
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    # fig.colorbar()

    # plt.title(f'Velocity = {velocity}')
    animation_3d = animation.ArtistAnimation(fig, images, interval=100, blit=True)
    animation_3d.save(r'C:\Users\antsor\Desktop\synth_rain_terrain.gif', writer='imagemagick', fps=60)
    plt.show()

