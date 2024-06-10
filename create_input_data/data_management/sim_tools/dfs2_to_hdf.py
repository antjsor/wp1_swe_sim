import mikeio
import os
import h5py

def dfs2_to_hdf(dfs2_directory,hdf_directory):
    # List all dfs2 files in the given dfs2_directory
    dfs2_files = [f for f in os.listdir(dfs2_directory) if f.endswith('.dfs2')]

    # Create a new folder for the HDF files
    # hdf_directory = os.path.join(dfs2_directory, "hdf_files")
    if not os.path.exists(hdf_directory):
        os.makedirs(hdf_directory)

    # Loop through each dfs2 file
    for dfs2_filename in dfs2_files[0:2]:
        full_dfs2_path = os.path.join(dfs2_directory, dfs2_filename)

        # Load the dfs2 file
        dfs2 = mikeio.open(full_dfs2_path)
        data = dfs2.read()

        # Create a hdf file name
        hdf_filename = dfs2_filename.replace('.dfs2', '.hdf')
        full_hdf_path = os.path.join(hdf_directory, hdf_filename)

        # Save data as HDF
        with h5py.File(full_hdf_path, 'w') as hdf:
            # You can customize this part to organize the HDF file as you want
            for idx, item in enumerate(data): # 
                hdf.create_dataset(f'{idx}', data=data)#Skal fikses

        # Check if the HDF file has been created successfully
        if os.path.isfile(full_hdf_path):
            print(f"Successfully created the HDF file: {hdf_filename}")

            # Delete the original dfs2 file
            if False:
                os.remove(full_dfs2_path)
                print(f"Deleted the original DFS2 file: {dfs2_filename}")
        else:
            print(f"Failed to create the HDF file: {hdf_filename}")