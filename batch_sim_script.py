import os
from data_management.rain_tools import Load_Rain
from utils import Batch_Simulations
data_type = r'train' # train, val or test

rain_path         =  r'M:\phd\wps\wp1\data\raw\from_rolo\rainseries\obs_events_marked_' + data_type.upper() + r'.csv'
mike_sim_path     = os.path.abspath(r'M:\phd\wps\wp1\data\generated_data\mike_sim_files\flat' + f'\{data_type}\sim_batches')


print(mike_sim_path)
rain_dict, rain_ids,_,_ = Load_Rain(rain_path)

Batch_Simulations(rain_ids,25,mike_sim_path)

