def Shift_Boundary_4_MIKE(terrain,n_shift = 2, boundary_shift = 0.1):
    terrain[:n_shift, :] -= boundary_shift
    terrain[-(n_shift):, :] -= boundary_shift
    terrain[n_shift:-n_shift,:n_shift] -= boundary_shift
    terrain[n_shift:-n_shift,-n_shift:] -= boundary_shift
    
    return terrain

def Lin_Shift_Boundary_4_MIKE(terrain, n_shift=3, boundary_shift=0.1):
    height, width = terrain.shape

    # Linearly change the top and bottom boundaries
    for i in range(1, n_shift + 1):
        change = boundary_shift * (n_shift - (i - 1)) / n_shift
        terrain[i - 1, :] -= change
        terrain[-i, :] -= change

    # Linearly change the left and right boundaries
    for j in range(1, n_shift + 1):
        change = boundary_shift * (n_shift - (j - 1)) / n_shift
        terrain[:, j - 1] -= change
        terrain[:, -j] -= change

    return terrain
