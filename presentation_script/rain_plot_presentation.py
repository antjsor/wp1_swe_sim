import numpy as np
import matplotlib.pyplot as plt
import math
import os
from data_management.rain_tools import Load_Rain

# plt.rcParams['text.usetex'] = True
domain_size = 16
t_final     = 16
plot_path = os.path.abspath(r'M:\phd\supervisor_meetings\presentations\23_08_21\figs')

np.random.seed(1337)
dt          = 1 
time_steps  = math.ceil(t_final/dt)
dx          = 1 # Cell resolution x
dy          = 1 # Cell resolution y
#TODO Need to translate windspeed to cell diz

domain = np.zeros(domain_size)
# print(np.tile([0,0,0,1],(1,16)))
# lineX = np.tile(np.tile([0,0,0,1],(1,16)),(domain_size,1))
# # liney = np.tile(np.tile([0,0,0,1],(1,16)),(domain_size,1))
# print(lineX.shape)
# # print(lineX.shape)
# lines = lineX + lineX.T

# plt.imshow(lines)
# plt.show()

rain_series = abs(np.cumsum(np.random.randn(1,time_steps)))

x = np.linspace(0,domain_size,domain_size+1)
y = np.linspace(0,domain_size,domain_size+1)
X,Y = np.meshgrid(x,y)


velocity = np.array([4,2]).flatten()
time_delays = np.int64(np.abs(X)/velocity[0] + np.abs(Y)/velocity[1])
data_type = r'train' # train, val or test
# print(current_drive)

rain_path         =  r'C:\Users\antsor\Desktop\d3a_local\data\rainseries\obs_events_marked_' + data_type.upper() + r'.csv'
ml_data_path      = os.path.abspath(r'C:\phd_project\wp1\data_tmp')
batch_path        = os.path.abspath(r'M:\phd\wps\wp1\data\generated_data\mike_sim_files\flat\train\sim_batches')
batch_name        = 'selected_16.txt'
# with open(os.path.join(batch_path,batch_name), 'r') as file:
#     # This will read each line, strip it of whitespace, and convert to an integer
#     batch_ids = [str(line.strip()) for line in file if line.strip()]
do_plot     = False
import_velocities = False
do_print = False


## For real rain series
dt_rain = 60
start_dry_time = 10*dt_rain
vmax = 10

rain_dict, rain_ids, rain_list, rain_datetime_dict = Load_Rain(rain_path)
# rain_series = abs(np.cumsum(np.random.randn(1,time_steps)))
rain_series = rain_dict(rain_ids[2])
### Time delay plot
# fig,ax = plt.subplots()
# cax = ax.imshow(time_delays,origin = 'lower',cmap = 'inferno_r')
# ax.set_title('Time delays[s] for rain application')
# fig.colorbar(cax, ax = ax)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_xlabel('X[m]')
# ax.set_ylabel('Y[m]')
# plt.show()

# ax.set_xticks(2*np.arange(time_delays.shape[1]))
# ax.set_yticks(np.arange(time_delays.shape[0]))

# Show grid
# ax.grid(which='both', color='red', linestyle='-', linewidth=1)

# plt.show()
plt.plot(rain_series,'-o')
plt.xlabel('Time [s]')
plt.ylabel('Intensity')
plt.title('Rain intensity time-series')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(plot_path,f"rain_t_series.png"), bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(plot_path,f"rain_t_series.svg"), bbox_inches='tight', pad_inches=0)
plt.figure()

rain_lines = np.tile(rain_series, (time_steps,1))
plt.imshow(rain_lines,origin='lower')
plt.xlabel('Time [s]')
plt.ylabel('Direction orthogonal to velocity vector[m]')
plt.colorbar()

### 
fig,ax = plt.subplots()
cax = ax.imshow(np.flipud(np.fliplr(rain_series[time_delays])),origin = 'lower',cmap = 'Blues')
ax.set_title('Rain_intensity t = max(time_delays)')
fig.colorbar(cax, ax = ax)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('X[m]')
ax.set_ylabel('Y[m]')

# plt.xlabel('Time(m)')
# plt.ylabel('Orthogonal to wind direction [m]')
# plt.colorbar()
# plt.title('Rain intensity')
# plt.axis('off')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(plot_path,f"rain_2_terrain.png"), bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(plot_path,f"rain_2_terrain.svg"), bbox_inches='tight', pad_inches=0)
# wind_direction = np.random.uniform(0,2*np.pi)
# wind_vec_true       = np.array([np.cos(wind_direction),np.sin(wind_direction)])
wind_direction = 0
wind_vec_true       = np.array([np.cos(wind_direction),np.sin(wind_direction)])
print(wind_vec_true)
wind_vec = np.abs(wind_vec_true)

plt.show()


# t_final = len(rain_series)
# rain_out = np.zeros_like(domain)
# for tt, rain in enumerate(rain_series):
#     rain_out[tt*wind_vec[0]]
#     domain_rain = [(rain[int(tt*wind_vec[0]):int(tt*wind_vec[0]) + domain_size -1, int(tt*wind_vec[1]):int(tt*wind_vec[1]) + domain_size -1])
#         for tt in range(0,t_final)]

# if wind_vec_true[0]<0 and wind_vec_true[1]<0:
#     domain_rain = [np.fliplr(np.flipud((rain[int(tt*wind_vec[0]):int(tt*wind_vec[0]) + domain_size -1, int(tt*wind_vec[1]):int(tt*wind_vec[1]) + domain_size -1])))
#                 for tt in range(0,t_final)]
# elif  wind_vec_true[0]<0:
#     domain_rain = [np.fliplr((rain[int(tt*wind_vec[0]):int(tt*wind_vec[0]) + domain_size -1, int(tt*wind_vec[1]):int(tt*wind_vec[1]) + domain_size -1]))
#             for tt in range(0,t_final)]

# elif wind_vec_true[1]<0:
#     domain_rain = [np.flipud((rain[int(tt*wind_vec[0]):int(tt*wind_vec[0]) + domain_size -1, int(tt*wind_vec[1]):int(tt*wind_vec[1]) + domain_size -1]))
#             for tt in range(0,t_final)]
# else:
#     domain_rain = [(rain[int(tt*wind_vec[0]):int(tt*wind_vec[0]) + domain_size -1, int(tt*wind_vec[1]):int(tt*wind_vec[1]) + domain_size -1])
#         for tt in range(0,t_final)]

