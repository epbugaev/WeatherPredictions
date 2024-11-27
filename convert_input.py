import h5netcdf
import numpy as np
import calendar
import tqdm 
from pathlib import Path

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

START_YEAR = 2000
END_YEAR = 2018

for year in range(START_YEAR, END_YEAR + 1):
    YEAR = str(year)

    input_folder = '/home/fratnikov/weather_bench/1.40625deg/'
    target_folder = '/home/fratnikov/weather_bench/npy/1.40625deg/' # '/home/fa.buzaev/data_to_egor/' + YEAR + '/' #'/home/epbugaev/weather_bench/1.40625deg/' + YEAR + '/'
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    step = 292 if not calendar.isleap(year) else 244
    hours = 8760 if not calendar.isleap(year) else 8784

    match_set_files = {}
    for key in match_set: 
        path = input_folder + key + '/' + key + '_' + YEAR + '_1.40625deg.nc'
        match_set_files[key] = h5netcdf.File(path, 'r')

    #step = 1 # TEMPORTAL

    for hour in tqdm.tqdm(range(0, hours, step)): 

        #if hour == 1:  # TEMPORAL
        #    hour = hours // 2 # TEMPORAL
        #elif hour > 1: # TEMPORAL
        #    break # TEMPORAL

        res = np.zeros([step, 110, 128, 256])

        for ind, key in enumerate(match_set): 
            start_id = ids[ind]
            end_id = ids[ind + 1]
                        
            if end_id - start_id == 1:
                res[:, start_id, :, :] = match_set_files[key][match_set[key]][hour:hour+step, :, :]
            else:
                res[:, start_id:end_id, :, :] = match_set_files[key][match_set[key]][hour:hour+step, start_id:end_id, :, :]

        for i in range(step):
            hour_str = str(hour + i)
            while len(hour_str) < 4:
                hour_str = '0' + hour_str

            Path(target_folder).mkdir(parents=True, exist_ok=True)
            np.save(target_folder + YEAR + '-' + hour_str + '.npy', res[i, :, :, :])
            
    for key in match_set: 
        match_set_files[key].close()
