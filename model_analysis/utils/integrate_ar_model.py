import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
class Integrate_AR_Model():
    def __init__(self, model_path, hdf5_path,test_key,start_time = 0,end_time = -1): # Movie these to the integration function
        self.model = torch.jit.load(model_path).eval()
        self.hdf5_path = hdf5_path
        self.test_key = test_key
        self.start_time = start_time
        self.end_time = end_time
        self.terrain,self.rain,self.swe = self.load_single_sim()
        

        self.integration_done = False
        self.state_names = ['Height', 'X-momentum', 'Y-momentum']
        self.model_id = self.extract_model_id(model_path)
        
    def extract_model_id(self,model_path):
        # Use a regular expression to extract the part between 'model_' and '.pth'
        pattern = r'PW.*?FD_\d+'
        match = re.search(pattern, model_path)
        if match:
            return match.group(0)
        return None
        
    def load_single_sim(self):
        with h5py.File(self.hdf5_path, 'r') as hdf:
            terrain = torch.from_numpy(hdf[self.test_key]['terrain'][:])
            rain = torch.from_numpy(hdf[self.test_key]['rain'][:]/(1000.0*1440.0))
            swe = torch.from_numpy(hdf[self.test_key]['swe'][:])
        return terrain,rain[self.start_time:self.end_time],swe[self.start_time:self.end_time]
    
    def integrate_in_time(self):
        self.sim_results = np.zeros_like(self.swe.numpy())
        self.sim_results[0] = self.swe[0]
        current_state = torch.cat([self.terrain.unsqueeze(dim = 0),self.rain[0].unsqueeze(dim = 0),self.swe[0]], dim = 0).unsqueeze(dim = 0)
        # Only integrate until the second to last time step since we need the last time step to predict the next time step
        for idx, rain_in in enumerate(self.rain[:-1]): # TODO fix index, this is a hacky work around.
            swe_pred = self.model(current_state).squeeze()
            current_state = torch.cat([self.terrain.unsqueeze(dim = 0),rain_in.unsqueeze(dim = 0),swe_pred], dim = 0).unsqueeze(dim=0)
            self.sim_results[idx+1] = swe_pred.detach().numpy()
        self.integration_done = True
        
    def compute_error(self,type = 'rmse'):
        if not self.integration_done:
            print('Time integration not done, running.')
            self.integrate_in_time()
        print(self.sim_results.shape,self.swe.numpy().shape)
        if type == 'rmse':
            error = (np.sqrt((self.sim_results - self.swe.numpy())**2)).mean((2,3))
        elif type=='mse':
            ((self.sim_results - self.swe.numpy())**2).mean((2,3))
        elif type == 'mae':
            np.abs(self.sim_results - self.swe.numpy()).mean((2,3))
        else:
            raise ValueError(f'Error type {type} not supported.')
        return error
   
    def make_movie(self, path, state=0, plot_type = 'dif'):
        if not self.integration_done:
            print('Time integration not done, running.')
            self.integrate_in_time()
            return

            # Create a masked array for initial display
        def mask_data(data, threshold=1e-15):
            masked_data = np.ma.masked_where(data < threshold, data)
            return masked_data
        num_time_steps = self.sim_results.shape[0]-1

        min_swe = np.min(self.swe.numpy(),axis = (0,2,3))
        max_swe = np.max(self.swe.numpy(),axis = (0,2,3))

        initial_sim = self.swe[0, state]
        initial_cnn = self.sim_results[0, state]
        initial_rain = self.rain[0]
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # Set titles
        axs[0].set_title('Simulated State')
        axs[1].set_title('CNN Predicted State')

        # Static terrain background for all subplots
        terrain_image = self.terrain.numpy()
        # Initialize the images for each subplot
        for ax in axs[:2]:
            ax.imshow(terrain_image, cmap='terrain', origin='lower', alpha=0.5)

        im_sim = axs[0].imshow(initial_sim, cmap='jet', alpha=0.7, origin='lower')
        im_cnn = axs[1].imshow(initial_cnn, cmap='jet', alpha=0.7, origin='lower')
        if plot_type == 'rain':
            min_max_rain = [np.min(self.rain.numpy()),np.max(self.rain.numpy())]

            im_rain = axs[2].imshow(initial_rain, cmap='Blues', alpha=0.7, origin='lower')
            axs[2].set_title('Rainfall Intensity')
            axs[2].set_title('Rainfall Intensity')
            cbar_label = 'Rainfall Intensity (mm/h)'
        elif plot_type == 'dif':
            min_max_dif = [np.min((self.swe.numpy()-self.sim_results)[:,state]),np.max((self.swe.numpy()-self.sim_results)[:,state])]
            difference_0 = self.swe[0, state]-self.sim_results[0, state]
            im_rain = axs[2].imshow(difference_0, cmap='jet_r', origin='lower')
            axs[2].set_title('CNN - Simulated Difference')
            axs[2].set_title('CNN - Simulated Difference')
            cbar_label = 'Difference (Units)'
            
        cbar_sim = fig.colorbar(im_sim, ax=axs[0])
        cbar_sim.set_label('Simulation Units')
        cbar_third = fig.colorbar(im_rain, ax=axs[2])
        cbar_third.set_label(cbar_label)
        cbar_cnn = fig.colorbar(im_cnn, ax=axs[1])
        cbar_cnn.set_label('CNN Prediction Units')
        def update(t):
            # Update the data for each image
            # Mask data dynamically for each frame
            # masked_sim = mask_data(self.swe[t, state])
            # masked_cnn = mask_data(self.sim_results[t, state])
            masked_sim = self.swe[t, state]
            masked_cnn = self.sim_results[t, state]
            if plot_type == 'rain':
                masked_rain = mask_data(self.rain[t])
                
            elif plot_type == 'dif':
                masked_rain =self.swe[t, state]-self.sim_results[t, state]

            # Update data for the images
            im_sim.set_data(masked_sim)
            im_cnn.set_data(masked_cnn)
            im_rain.set_data(masked_rain)
            im_sim.set_clim(vmin=min_swe[state], vmax=max_swe[state]*1.25)
            im_cnn.set_clim(vmin=min_swe[state], vmax=max_swe[state]*1.25)

            # Update titles to show the current time step
            axs[0].set_title(f'Simulated State at t={t}')
            axs[1].set_title(f'CNN Predicted State at t={t}')
            if plot_type == 'rain':
                axs[2].set_title(f'Rainfall Intensity at t={t}')
                im_rain.set_clim(vmin=min_max_rain[0], vmax=min_max_rain[1])

            elif plot_type == 'dif':
                axs[2].set_title(f'CNN - Simulated Difference at t={t}')
            im_rain.set_clim(vmin=min_max_dif[0], vmax=min_max_dif[1])


            return im_sim, im_cnn, im_rain

        # Create the animation using FuncAnimation
        ani = animation.FuncAnimation(fig, update, frames=num_time_steps, blit=True, repeat_delay=2500,interval=400)
        plt.tight_layout()
        fig.subplots_adjust(top=0.95, right=0.85)
        # Save the animation
        ani.save(path, writer='imagemagick', fps=60)
        plt.show()

        
    def results2hdf(self):
        pass