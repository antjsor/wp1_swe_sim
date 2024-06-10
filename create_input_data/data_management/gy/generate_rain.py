import numpy as np
import pandas
from datetime import datetime
### TODO Make class that initializes all the rain all the way to MIKE data sets, and then when the method to export to dfs is called it does so for all initialized data

def Zero_boundary_rain(terrain,n_shift = 2):
    terrain[:n_shift, :] = 0
    terrain[-(n_shift):, :] = 0
    terrain[n_shift:-n_shift,:n_shift] = 0
    terrain[n_shift:-n_shift,-n_shift:] = 0
        
    return terrain

def Load_rain(path, start_time = None):

    rain_df   = pandas.read_csv(path, sep=';', header=None)
    id_column = rain_df.columns[-1]
    
    # rain_datetime = rain_df[rain_df.columns[0].values]

    rain_ids  = list(set(rain_df[id_column].values))
    # rain_datetime_dict = {rain_id : rain_df.loc[rain_df[2]==rain_id][0].values for rain_id in rain_ids}
    rain_datetime_dict = {rain_id : rain_df.loc[rain_df[2]==rain_id][0] for rain_id in rain_ids}
    
    rain_dict = {rain_id : rain_df.loc[rain_df[2]==rain_id][1].values for rain_id in rain_ids}
    rain_list = rain_dict.values()
    
    # Loop for changing starting time of simulation to a specified value
    if start_time:
        date_format = '%Y-%m-%d %H:%M:%S'
        for rain_id in rain_ids:
            delta_time = datetime.strptime(rain_datetime_dict[rain_id][rain_datetime_dict[rain_id].keys()[0]], date_format) - datetime.strptime(start_time, date_format)
            rain_datetime_dict[rain_id] =rain_datetime_dict[rain_id].apply(lambda x: datetime.strftime(datetime.strptime(x,date_format)-delta_time,date_format))
    
    return rain_dict, rain_ids, rain_list, rain_datetime_dict
        
def Translate_Rain_2D(time_series, domain, velocity_true, dt, n_pad = 0, start_dry_t = 0):
    velocity = np.abs(velocity_true)
    
    t_rain  = len(time_series)*dt - dt
    t_final = t_rain + start_dry_t
    
    time_delays = np.int64(np.abs(domain[0])/velocity[0] + np.abs(domain[1])/velocity[1])
    # time_delays = np.int64(np.maximum(np.abs(domain[0])/velocity[0], np.abs(domain[1])/velocity[1]))
    if velocity_true[0]<0 and velocity_true[1]<0:
        time_delays = np.flipud(np.fliplr(time_delays))
    elif velocity_true[0]<0:
        time_delays = np.fliplr(time_delays)
    elif velocity_true[1]<0:
        time_delays = np.flipud(time_delays)
        rain_out = []
    
    
    #Dry loop over inial dry phase
    if start_dry_t>0: # Probably not necessary since it should be empty for range if start_dry_t too small
        rain_out = [np.zeros_like(domain[0]) for _ in range(0, start_dry_t, dt)]
    
    # Wet loop over rain series
    for tt in range(0, t_rain+dt, dt):
        rain_tt       = np.zeros_like(domain[0])
        mask          = tt-time_delays > 0
        times         = tt-time_delays
        rain_tt[mask] = time_series[np.int64((times[mask])/dt)]
        rain_tt[:n_pad, :] = 0
        rain_tt[-(n_pad):, :] = 0
        rain_tt[n_pad:-n_pad,:n_pad] = 0
        rain_tt[n_pad:-n_pad,-n_pad:] = 0
    
        rain_out.append(rain_tt)
    return rain_out


    # for tt in range(0, t_final+dt, dt):
    #     rain_tt       = np.zeros_like(domain[0])
        
    #     # Fill out with rain from time series
    #     if tt <= end_dry_t:
    #         mask          = tt-time_delays > 0
    #         times         = tt-time_delays
    #         rain_tt[mask] = time_series[np.int64((times[mask])/dt)]
    #         rain_tt[:n_pad, :] = 0
    #         rain_tt[-(n_pad):, :] = 0
    #         rain_tt[n_pad:-n_pad,:n_pad] = 0
    #         rain_tt[n_pad:-n_pad,-n_pad:] = 0
        
    #     rain_out.append(rain_tt)
    #     # Else return zeros since we entered the dry steps

    # return rain_out



# ### TODO loop for if we can get MIKE to run with unequal time-steps
# def time_series_2_terrain(time_series, domain, velocity_true, dt, n_pad = 0, start_dry_t = 0):
#     velocity = np.abs(velocity_true)
    
#     t_rain  = len(time_series)*dt - dt
#     # t_final = t_rain + start_dry_t

#     t_final = t_rain
    
#     time_delays = np.int64(np.abs(domain[0])/velocity[0] + np.abs(domain[1])/velocity[1])
#     # time_delays = np.int64(np.maximum(np.abs(domain[0])/velocity[0], np.abs(domain[1])/velocity[1]))
#     if velocity_true[0]<0 and velocity_true[1]<0:
#         time_delays = np.flipud(np.fliplr(time_delays))
#     elif velocity_true[0]<0:
#         time_delays = np.fliplr(time_delays)
#     elif velocity_true[1]<0:
#         time_delays = np.flipud(time_delays)
#         rain_out = []
    
#     rain_out = []
    
#     for tt in range(0,t_final+dt, dt):
#         rain_tt       = np.zeros_like(domain[0])
#         mask          = tt-time_delays > 0
#         times         = tt-time_delays
#         rain_tt[mask] = time_series[np.int64((times[mask])/dt)]
        
#         ## TODO make cleaner, perhaps
#         rain_tt[:n_pad, :] = 0
#         rain_tt[-(n_pad):, :] = 0
#         rain_tt[n_pad:-n_pad,:n_pad] = 0
#         rain_tt[n_pad:-n_pad,-n_pad:] = 0
#         rain_out.append(rain_tt)
        
#     # for tt in range(0,t_final+dt, dt):
#     #     rain_tt       = np.zeros_like(domain[0])
#     #     if tt <= t_rain:
#     #         mask          = tt-time_delays > 0
#     #         times         = tt-time_delays
#     #         rain_tt[mask] = time_series[np.int64((times[mask])/dt)]
            
#     #         ## TODO make cleaner, perhaps
#     #         rain_tt[:n_pad, :] = 0
#     #         rain_tt[-(n_pad):, :] = 0
#     #         rain_tt[n_pad:-n_pad,:n_pad] = 0
#     #         rain_tt[n_pad:-n_pad,-n_pad:] = 0
#     #     rain_out.append(rain_tt)
    
#     if start_dry_t>0:
#         rain_out.append(np.zeros_like(domain[0]))
#     return rain_out
