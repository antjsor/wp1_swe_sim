import pandas
from datetime import datetime
# from truncate_end_threshold import Truncate_End_Threshold

# def Truncate_End_Threshold(data, keep=60, threshold=0.0):
#     reversed_data = data[::-1]
#     low_value_indices = np.where(reversed_data <= threshold)[0]

#     if len(low_value_indices) == 0 or low_value_indices[0] != 0:
#         return data, len(data)
#     # Identify the last continuous low-value block
#     last_block = np.split(low_value_indices, np.where(np.diff(low_value_indices) != 1)[0] + 1)[0]

#     if len(last_block) > keep:
#         keep_index = len(data) - len(last_block) + keep
#         return data[:keep_index], keep_index
#     else:
#         return data, len(data)

def Load_Rain(path, start_time = None):

    # rain_df   = pandas.read_csv(path, sep=';', header=None)
    rain_df = pandas.read_csv(path, sep=';', header=None)
    rain_df[rain_df.columns[-1]] = rain_df[rain_df.columns[-1]].apply(str)
    rain_ids = rain_df.iloc[:, -1].unique().tolist()
    rain_datetime_dict = {rain_id : rain_df.loc[rain_df[2]==rain_id][0] for rain_id in rain_ids}
    
    rain_dict = {rain_id : rain_df.loc[rain_df[2]==rain_id][1].values for rain_id in rain_ids}
    rain_list = rain_dict.values()
    # Loop for changing starting time of simulation to a specified value
    if start_time:
        date_format = '%Y-%m-%d %H:%M:%S'
        for rain_id in rain_ids:
            delta_time = datetime.strptime(rain_datetime_dict[rain_id][rain_datetime_dict[rain_id].keys()[0]], date_format) - datetime.strptime(start_time, date_format)
            rain_datetime_dict[rain_id] =rain_datetime_dict[rain_id].apply(lambda x: datetime.strftime(datetime.strptime(x,date_format)-delta_time,date_format))
    
    return rain_dict, rain_ids, rain_datetime_dict, rain_list



# def Load_Rain(path, start_time = None):

#     # rain_df   = pandas.read_csv(path, sep=';', header=None)
#     rain_df = pandas.read_csv(path, sep=';', header=None)
#     rain_df[rain_df.columns[-1]] = rain_df[rain_df.columns[-1]].apply(str)
#     rain_ids = rain_df.iloc[:, -1].unique().tolist()
#     rain_datetime_dict = {rain_id : rain_df.loc[rain_df[2]==rain_id][0] for rain_id in rain_ids}
    
#     rain_dict = {rain_id : rain_df.loc[rain_df[2]==rain_id][1].values for rain_id in rain_ids}
#     rain_list = rain_dict.values()
#     # Loop for changing starting time of simulation to a specified value
#     if start_time:
#         date_format = '%Y-%m-%d %H:%M:%S'
#         for rain_id in rain_ids:
#             delta_time = datetime.strptime(rain_datetime_dict[rain_id][rain_datetime_dict[rain_id].keys()[0]], date_format) - datetime.strptime(start_time, date_format)
#             rain_datetime_dict[rain_id] =rain_datetime_dict[rain_id].apply(lambda x: datetime.strftime(datetime.strptime(x,date_format)-delta_time,date_format))
    
#     return rain_dict, rain_ids, rain_list, rain_datetime_dict