import torch
import pandas as pd
import numpy as np
import os
from pyhdf.SD import SD, SDC
import pdb
from datetime import datetime

def clean_dataset(input_dir: str, output_dir: str):
    """transfer HDF file into pytorch tensor

    Args:
        input_dir (str): root directory for input hdf file data
        output_dir (str): output directory for generated torch tensor data
    """
    file_list = os.listdir(input_dir)
    
    if os.path.exists(output_dir) == False:
        raise FileNotFoundError("Output dir do not exist!")
    
    for file_name in file_list:
        granule_id = file_name.split('.')[2]
        path = os.path.join(input_dir, file_name)
        hdf = SD(path, SDC.READ)
        timestamp_list = hdf.attributes()['Orbit_time_stamp'].split()
        data_055 = get_AOD_data(hdf, '055')
        data_047 = get_AOD_data(hdf, '047')
        tensor_055 = torch.from_numpy(data_055)
        for i, timestamp in enumerate(timestamp_list):
            t_or_a = timestamp[-1]
            timestamp = datetime.strptime(timestamp[:-1], '%Y%j%H%M')
            time_str = timestamp.strftime('%Y%m%d%H%M')
            export_tensor(tensor_055[i].half(), os.path.join(output_dir, f'{time_str}_{t_or_a}_{granule_id}_055.pt'))
    return

def export_tensor(tensor, file_path):
    return torch.save(tensor, file_path)

def get_AOD_data(hdf_file, code = '055'):
    data_055 = hdf_file.select(f'Optical_Depth_{code}')[:]
    data_055 = data_055.astype(np.float32)
    data_055[data_055 == -28672] = np.nan
    data_055 = data_055 * 0.001
    return data_055

if __name__ == '__main__':
    clean_dataset('./maiac/CA_2018_2023', './maiac/torch/CA_2018_2023')