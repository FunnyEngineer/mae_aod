import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import matplotlib
from pyhdf.SD import SD, SDC
import pdb
import pprint
import os


explain_dict = {
    0: 'Undefined',
    1: 'Clear',
    2: 'Possibly Cloudy (detected by AOD filter)',
    3: 'Cloudy',
    4: 'Cloud Shadow',
    5: 'Hot Spot or fire',
    6: 'Water sediments', }


def slicer_vectorized(a):
    old_shp = a.shape
    n_arr = a.flatten().shape[0]
    b = a.flatten().view((str, 1)).reshape(n_arr, -1)[:, -3:]
    return np.fromstring(b.tostring(), dtype=(str, 3)).reshape(old_shp)


def cloud_mask_classify(a):
    new_arr = np.zeros_like(a, dtype=np.int16)
    reference_table = {
        '000': 0,  # undefined
        '001': 1,  # Clear
        '010': 2,  # Possibly Cloudy (detected by AOD filter)
        '011': 3,  # Cloudy
        '101': 4,  # cloud shadow
        '110': 5,  # Hot Spot or fire
        '111': 6, }  # Water sediments
    for str_bit in reference_table:
        new_arr[a == str_bit] = reference_table[str_bit]
    return new_arr


def plot_qa_table(img_qa, img_aod):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img_aod, interpolation='nearest')
    im = ax2.imshow(img_qa, interpolation='nearest', cmap='Paired')
    values = np.unique(img_qa.ravel())
    x = {}
    for value in values:
        x[value] = {"color": im.cmap(im.norm(value)),
                    "label": explain_dict[value]}
    patches = [mpatches.Patch(
        color=x[value]["color"],
        label=x[value]["label"])
        for value in values]
    ax2.legend(handles=patches, loc='lower right', 
               bbox_to_anchor=(1, -0.5), 
               borderaxespad=0.,
               fancybox=True)
    plt.tight_layout()
    plt.show()
    # plt.savefig('preview_AOD.png')

def get_055_data(hdf_file):
    data_055 = hdf_file.select('Optical_Depth_055')[:]
    data_055 = data_055.astype(np.float32)
    data_055[data_055 == -28672] = np.nan
    data_055 = data_055 * 0.001
    masked_055 = np.ma.array(data_055, mask=np.isnan(data_055))

    return masked_055

def get_qa_data(hdf_file):
    binary_repr_v = np.vectorize(np.binary_repr)
    data_QA = binary_repr_v(hdf_file.select('AOD_QA')[:], 16)
    class_QA = np.zeros_like(data_QA, dtype=np.int8)
    for i in range(data_QA.shape[0]):
        img_qa = slicer_vectorized(data_QA[i])
        img_qa = cloud_mask_classify(img_qa)
        class_QA[i] = img_qa
    
    return class_QA

def transfer_hdf(file_path):
    hdf = SD(file_path, SDC.READ)
    pprint.pprint(hdf.datasets())

    masked_055 = get_055_data(hdf)
    
    
    # QA processing
    qa = get_qa_data(hdf)

    for i in range(masked_055.shape[0]):
        plot_qa_table(qa[i], masked_055[i])
        
    # see aggre gation
    # agg_value = np.zeros_like(masked_055[0], dtype=np.int8)
    # for i in range(masked_055.shape[0]):
    #     agg_value[masked_055[i].mask == False] += 1
    
    # pdb.set_trace()
    return qa, masked_055

def test_overlad_rate(file_dir):

    file_name_list = os.listdir(file_dir)
    result_df = pd.DataFrame(columns=['File Name', 'Ratio'])
    for file_name in file_name_list:
        file_path = os.path.join(file_dir, file_name)

        hdf = SD(file_path, SDC.READ)
        masked_055 = get_055_data(hdf)

        for i in range(masked_055.shape[0]):
            masked_img = masked_055[i]
            ratio = (masked_img.mask == False).sum() / masked_img.size
            result_df.loc[len(result_df.index)] = [f'{file_name}_{i}', ratio]
    
    result_df.to_csv('cover_ratio.csv')
    pdb.set_trace()

def plot_cover_ratio():
    df = pd.read_csv('./cover_ratio.csv')
    fig = px.ecdf(df, x="Ratio")
    fig.show()
    
    pdb.set_trace()

if __name__ == '__main__':
    # transfer_hdf(
    #     '/home/niyogi_shared/ting_yu/data/maiac/MCD19A2.A2023048.h08v05.006.2023050091938.hdf')
    # test_overlad_rate('./maiac')
    plot_cover_ratio()