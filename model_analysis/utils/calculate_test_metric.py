import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import pandas as pd
class CNN_Test_Metric():
    def __init__(self,hdf5_path,start_time = 0,end_time = -1): # start time end time should be set in the time integrator.
        # self.model_paths = model_paths
        # self.test_ids = test_ids
        self.start_time = start_time
        self.hdf5_path = hdf5_path
        self.end_time = end_time
    def load_single_sim(self,test_key):
        with h5py.File(self.hdf5_path, 'r') as hdf:
            terrain = torch.from_numpy(hdf[test_key]['terrain'][:])
            rain = torch.from_numpy(hdf[test_key]['rain'][:]/(1000.0*1440.0))
            swe = torch.from_numpy(hdf[test_key]['swe'][:])
        return terrain,rain[self.start_time:self.end_time],swe[self.start_time:self.end_time]
    
    def integrate_in_time(self, model_path, test_id):
        model = torch.jit.load(model_path)
        model.eval()
        
        terrain,rain,swe = self.load_single_sim(test_id)
        sim_results = np.zeros_like(swe.numpy())
        sim_results[0] = swe[0]
        current_state= torch.cat([rain[0].unsqueeze(dim = 0),swe[0]], dim = 0).unsqueeze(dim = 0)
        
        x_stat = terrain.unsqueeze(dim = 0).unsqueeze(dim = 0)
        for idx, rain_in in enumerate(rain[:-1]): # TODO fix index, this is a hacky work around.
            swe_pred = model(x_stat, current_state).squeeze()
            current_state = torch.cat([rain_in.unsqueeze(dim = 0),swe_pred], dim = 0).unsqueeze(dim=0)
            sim_results[idx+1] = swe_pred.detach().numpy()
        return sim_results,swe.numpy()


    def calculate_metric(self,model_paths, test_ids,type = 'rmse'):
        @staticmethod
        def extract_model_id(model_path):
            # Use a regular expression to extract the part between 'model_' and '.pth'
            pattern = r'PW.*?FD_\d+'
            match = re.search(pattern, model_path)
            if match:
                return match.group(0)
            return None
        
        metric_dict = {extract_model_id(model_path): {test_id: {} for test_id in test_ids} for model_path in model_paths}
        for model_path in model_paths:
            model_key = extract_model_id(model_path)
            print(f'Calculating metric for model: {model_key}')
            
            for test_id in test_ids:
                print(test_id)
                sim_results, swe = self.integrate_in_time(model_path, test_id)
                # print(sim_results.shape,swe.shape)
                if type == 'rmse':
                    error = (np.sqrt((sim_results - swe)**2)).mean((2,3)).T
                elif type=='mse':
                    ((sim_results - swe)**2).mean((2,3)).T
                elif type == 'mae':
                    np.abs(sim_results - swe).mean((2,3)).T
                else:
                    raise ValueError(f'Error type {type} not supported.')
   
                metric_dict[model_key][test_id]['height'] = error[0]
                metric_dict[model_key][test_id]['xmom']  = error[1]
                metric_dict[model_key][test_id]['ymom']  = error[2]
                
        return metric_dict
      
        # data = []
        # for model_id, tests in metric_dict.items():
        #     print(model_id)
        #     print(tests)
        #     for test_id, metrics in tests.items():
        #         print(test_id)
        #         print(metrics)
        #         data.append((model_id, test_id, metrics['height'], metrics['xmom'], metrics['ymom']))

        # # Create a DataFrame from the list
        # df = pd.DataFrame(data, columns=['Model ID', 'Test ID', 'Height', 'XMom', 'YMom'])

        # # Set the hierarchical index
        # df.set_index(['Model ID', 'Test ID'], inplace=True)

            # Display the DataFrame to check
        # return metric_dict, df