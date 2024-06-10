import numpy as np
def Zero_boundary_rain(terrain,n_shift = 2):
    terrain[:n_shift, :] = 0
    terrain[-(n_shift):, :] = 0
    terrain[n_shift:-n_shift,:n_shift] = 0
    terrain[n_shift:-n_shift,-n_shift:] = 0
        
    return terrain