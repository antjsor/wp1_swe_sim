import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mikeio
import torch

make_movie = True
import h5py
from datetime import datetime
from datetime import timedelta
import pandas as pd

import os
import re

# dfs2 = mikeio.open(full_dfs2_path)
data = mikeio.open('Results_49678.dfs2').read()

# Create a hdf file name

# Save data as HDF
data_np = data.to_numpy()
# data_np = np.transpose(data_np,(1,0,2,3))
# with h5py.File(full_hdf_path, 'w') as hdf:

if make_movie:
    fig,ax = plt.subplots()
    images = [
        [ax.imshow(
            layer, animated=True,origin = 'lower', vmin = 0, vmax = np.max(data_np))]
        for layer in data_np[0]
    ]
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    # fig.colorbar()

    # plt.title(f'Velocity = {velocity}')
    animation_3d = animation.ArtistAnimation(fig, images, interval=100, blit=True)
    animation_3d.save(r'C:\Users\antsor\Desktop\49678_result.gif', writer='imagemagick', fps=60)
    plt.show()
