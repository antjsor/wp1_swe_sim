import numpy as np

import os
import random
import json
from data_management.rain_tools import Load_Rain
np.random.seed(666)
data_type = r'train' # train, val or test
# print(current_drive)
vmax = 10
rain_path         =  r'M:\phd\wps\wp1\data\raw\from_rolo\rainseries\obs_events_marked_' + data_type.upper() + r'.csv'
mike_sim_path     = os.path.abspath(r'M:\phd\wps\wp1\data\generated_data\mike_sim_files\flat' + f'\{data_type}')
import pandas as pd


_, rain_ids, _, _ = Load_Rain(rain_path)
velocities = {rain_id: [random.uniform(-10, 10) for _ in range(2)] for rain_id in rain_ids}
df = pd.DataFrame(velocities).T
df.columns = ['vx', 'vy']
df.index.name = 'rain_id'  # Naming the index as 'rain_id'
directions = np.linspace(0,2*np.pi, 16,endpoint=False)
velocities = np.random.uniform(2,10,16)

batch_path        = os.path.abspath(r'M:\phd\wps\wp1\data\generated_data\mike_sim_files\flat\train\sim_batches')
batch_name        = 'max15_30min_avg.txt'
with open(os.path.join(batch_path,batch_name), 'r') as file:
    batch_ids = [str(line.strip()) for line in file if line.strip()]

random.shuffle(batch_ids)

for batch in batch_ids:
    print(batch)
    
velocities2 = {rain_id: [np.cos(direction)*velocity, np.sin(direction)*velocity]\
    for rain_id,velocity,direction in zip(batch_ids,velocities,directions)}

df2 = pd.DataFrame(velocities2).T
df2.columns = ['vx', 'vy']
df2.index.name = 'rain_id'  
df2.to_csv(os.path.join(mike_sim_path, '16_v_vectors.csv'))




