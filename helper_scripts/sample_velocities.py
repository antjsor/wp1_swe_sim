import numpy as np
import random
np.random.seed(666)
data_type = r'train' # train, val or test
# print(current_drive)

rain_path         =  r'M:\phd\wps\wp1\data\raw\from_rolo\rainseries\obs_events_marked_' + data_type.upper() + r'.csv'
rain_dict, rain_ids, rain_list, rain_datetime_dict = gr.load_rain(rain_path)

velocities = [[random.uniform(0, 1) for _ in range(2)] for _ in range(203)]
abc2 = random.uniform(-10,10,[203,2])