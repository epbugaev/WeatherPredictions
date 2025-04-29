import os
import numpy as np
import torch
import pandas as pd
import h5netcdf
from torch.utils.data import Dataset, DataLoader


class WeatherBench128(Dataset):
    def __init__(self, 
                 start_time: str='2000-01-01 00:00:00', 
                 end_time: str='2000-01-05 23:00:00',
                 include_target: bool=False,
                 lead_time: int=6, 
                 interval: int=6,
                 muti_target_steps: int=1,
                 # New parameters for time sequences
                 start_time_x: int=0,
                 end_time_x: int=1,
                 start_time_y: int=0,
                 end_time_y: int=1, 
                 cut=None, 
                 num_preload=12):
        
        self.variables_list = [
        0, 1, 2, 4 ,
        6 ,7 ,8 ,9 ,10,11,12,13,14,15,16,17,18,
        19,20,21,22,23,24,25,26,27,28,29,30,31,
        45,46,47,48,49,50,51,52,53,54,55,56,57,
        58,59,60,61,62,63,64,65,66,67,68,69,70,
        71,72,73,74,75,76,77,78,79,80,81,82,83]
        self.data_folder = '/home/fratnikov/weather_bench/npy/1.40625deg/' #"/home/fa.buzaev/data_to_egor"
        self.start_time = start_time
        self.end_time = end_time
        self.include_target = include_target
        self.lead_time = lead_time
        self.interval = interval
        self.muti_target_steps = muti_target_steps
        
        # Store the new sequence parameters
        self.start_time_x = start_time_x
        self.end_time_x = end_time_x
        self.start_time_y = start_time_y
        self.end_time_y = end_time_y
        
        # +1 because the ranges are now inclusive
        self.x_sequence_length = end_time_x - start_time_x + 1
        self.y_sequence_length = end_time_y - start_time_y + 1
        
        if self.x_sequence_length <= 0:
            raise ValueError("end_time_x must be greater than or equal to start_time_x")
        if self.y_sequence_length <= 0:
            raise ValueError("end_time_y must be greater than or equal to start_time_y")
            
        self.init_time_list()
        self.init_file_list()
        self.get_mean_std()
        # Update the length calculation to account for inclusive ranges
        self.length = len(self.x_time_ilst) - max(self.end_time_x, self.end_time_y + self.muti_target_steps * self.lead_time // self.interval)
        
        if self.length <= 0:
            raise ValueError("Not enough time steps available for the requested sequence lengths")
        

        self.cut = cut
        self.preload = {}
        self.num_preload = num_preload
        
        self.hit = 0
        self.not_hit = 0


    def custom_np_load(self, file_path):
        input_folder = '/home/fratnikov/weather_bench/1.40625deg/'
        match_set = {
                '2m_temperature': 't2m', 
                '10m_u_component_of_wind': 'u10', 
                '10m_v_component_of_wind': 'v10', 
                'total_cloud_cover': 'tcc', 
                'total_precipitation': 'tp', 
                'toa_incident_solar_radiation': 'tisr', 
                'geopotential': 'z', 
                'temperature': 't',
                'specific_humidity': 'q', 
                'relative_humidity' : 'r', 
                'u_component_of_wind': 'u', 
                'v_component_of_wind': 'v', 
                'vorticity': 'vo', 
                'potential_vorticity': 'pv'
        }

        ids = [0, 1, 2, 3, 4, 5, 6, 19, 32, 45, 58, 71, 84, 97, 110]
        year = file_path.split("/")[-1][0:4]
        hour = int(file_path.split("/")[-1][5:9])

        if self.preload is not None and (year + '-' + str(hour)) in self.preload: # If already loaded, do not load again
            self.hit += 1
            return self.preload[year + '-' + str(hour)]
        else:
            self.not_hit += 1
            self.preload = {}

        match_set_files = {}
        for key in match_set: 
            path = input_folder + key + '/' + key + '_' + year + '_1.40625deg.nc'
            match_set_files[key] = h5netcdf.File(path, 'r')

        

        right_bound_hours = max(hour + 1, min(7861, hour + self.num_preload))

        res = np.zeros([right_bound_hours - hour, 110, 128, 256])

        for ind, key in enumerate(match_set): 
            start_id = ids[ind]
            end_id = ids[ind + 1]

            if end_id - start_id == 1:
                res[:, start_id, :, :] = match_set_files[key][match_set[key]][hour:right_bound_hours, :, :]
            else:
                res[:, start_id:end_id, :, :] = match_set_files[key][match_set[key]][hour:right_bound_hours, 0:13, :, :]

        for cur_hour in range(hour, right_bound_hours): 
            self.preload[year + '-' + str(cur_hour)] = res[cur_hour - hour, ...]

        for key in match_set: 
            match_set_files[key].close()

        return self.preload[year + '-' + str(hour)]
    

    def init_time_list(self):
        if self.include_target:
            target_end_time = pd.to_datetime(self.end_time)
            input_end_time = target_end_time - pd.Timedelta(hours=self.muti_target_steps*self.lead_time)
            input_end_time_str = input_end_time.strftime('%Y-%m-%d %H:%M:%S')
            self.x_time_ilst = pd.date_range(self.start_time, input_end_time_str, freq=str(self.interval)+'h')
        else:
            self.x_time_ilst = pd.date_range(self.start_time, self.end_time, freq=str(self.interval)+'h')


    def idx_in_year(self, time_stamp):
        year = time_stamp.year
        first_day = pd.to_datetime(f'{year}-01-01 00:00:00')
        idx = int((time_stamp - first_day).total_seconds() / 3600)
        return idx
    

    def init_file_list(self):
        # your_weatherbench_data_path/1979/1979-0000.npy
        self.x_file_list= [os.path.join(self.data_folder, #str(time_stamp.year),
                                        str(time_stamp.year)+'-{:04d}'.format(self.idx_in_year(time_stamp))+'.npy')
                                        for time_stamp in self.x_time_ilst]
        

    def get_mean_std(self):
        mean_std = np.load("/home/epbugaev/weather_bench/1.40625deg/mean_std.npy")
        # mean_std = np.ones([2, 110]) # Test
        self.the_mean = mean_std[0]
        self.the_std = mean_std[1]
        self.data_mean_tensor = torch.from_numpy(self.the_mean[self.variables_list]).float()
        self.data_std_tensor = torch.from_numpy(self.the_std[self.variables_list]).float()

   
    def normalization(self, sample):
        return (sample[self.variables_list] - self.the_mean[self.variables_list, None, None]) / self.the_std[self.variables_list, None, None]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Load a sequence of X inputs
        x_sequence = []
        for i in range(self.start_time_x, self.end_time_x + 1):
            current_index = index + i
            file_path = self.x_file_list[current_index]
            sample_x = self.custom_np_load(file_path)
            sample_x = self.normalization(sample_x)
            sample_x = torch.from_numpy(sample_x).float()

            if self.cut is not None:
                sample_x = sample_x[..., self.cut[0][0]:self.cut[0][1], self.cut[1][0]:self.cut[1][1]]

            x_sequence.append(sample_x)

        # Stack X sequence
        sample_x_sequence = torch.stack(x_sequence, dim=0)  # [T, C, H, W]
        
        # Load a sequence of Y targets
        y_sequences = []
        for steps in range(self.muti_target_steps):
            y_sequence = []
            for i in range(self.start_time_y, self.end_time_y + 1):
                # Calculate the corresponding time for this target
                x_time = self.x_time_ilst[index + i]
                y_time = x_time + pd.Timedelta(hours=(steps+1)*self.lead_time)
                y_file_path = os.path.join(self.data_folder,
                                          str(y_time.year)+'-{:04d}'.format(self.idx_in_year(y_time))+'.npy')
                sample_y = self.custom_np_load(y_file_path)
                sample_y = self.normalization(sample_y)
                sample_y = torch.from_numpy(sample_y).float()

                if self.cut is not None:
                    sample_y = sample_y[..., self.cut[0][0]:self.cut[0][1], self.cut[1][0]:self.cut[1][1]]
                    
                y_sequence.append(sample_y)
            
            # Stack this target sequence
            stacked_y_sequence = torch.stack(y_sequence, dim=0)  # [T, C, H, W]
            y_sequences.append(stacked_y_sequence)
        
        if self.muti_target_steps > 1:
            # Shape: [muti_target_steps, T, C, H, W]
            sample_y_all = torch.stack(y_sequences, dim=0)
        else:
            # Shape: [T, C, H, W]
            sample_y_all = y_sequences[0]
            
        # self.preload = {} # This is useful if you run out of RAM
        return sample_x_sequence, sample_y_all
