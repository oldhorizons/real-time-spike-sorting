import sklearn
import pickle
import numpy as np
import os

os.chdir('C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/sim_hybrid_ZFM_10sec')
with open('kilosort4/pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)

def load_data(data_dir):
    """
    loads data into format appropriate for processing.
    data for this comes in one of two formats: 
    recorded from the lab 
      (in which case sample_numbers and timestamps will exist, and ground truth will not exist)
    or downloaded from kilosort test data 'https://janelia.figshare.com/ndownloader/articles/25298815/versions/1' 
      (in which case sample_numbers will NOT exist, and ground truth WILL exist)
    Args:
        dir(string): the location of the data, including the name of the raw binary file
    Returns: 
        data([[float]]): the data, reshaped to appropriate size
    """
    if os.path.isfile(os.path.join(data_dir, 'continuous.dat')):
        num_channels = 384
        extension = ".dat"
        data = np.memmap(os.path.join(data_dir, 'continuous.dat'), mode='r', dtype='int16')
    else:
        num_channels = 385
        extension = ".bin"
        data = np.memmap(os.path.join(data_dir, 'continuous.bin'), mode='r', dtype='int16')
    data = data.reshape((len(data) // num_channels, num_channels))
    return data, extension

def load_gt(data_dir):
    """
    Args: 
        dir (str) the dir in which the ground truth is held
            NB if the ground truth is a kilosort output, provide the kilosort folder, NOT the folder with the raw binary file 
            (see file structure in README.md)
    Returns: 
        gt (dict): a dictionary of the ground truth
    """
    if os.path.isfile(os.path.join(data_dir, 'sim.imec0.ap_params.npz')):
        #ground truth is genuine ground truth, comes in .npz format
        with np.load(data_dir + "/sim.imec0.ap_params.npz") as f:
            gt = dict(f)
    else:
        # ground truth is kilosort output, comes in three separate .npy files
        gt = dict()
        gt['st'] = np.load(data_dir + "/spike_times.npy")
        gt['cl'] = np.load(data_dir + "/spike_clusters.npy")
        gt['wfs'] = np.load(data_dir + "/templates.npy")
    return gt

# def crop_data(data_dir, num_samples = 300000, gt_dir = None, offset = 0):
#     """
#     crops a dataset to the required length. If gt_dir is given, crops ground truth to the same length for validation purposes
#     and copies that gt to the new folder, per structure in README.md
#     Args: 
#         data_dir (str): the directory in which the raw binary file is found
#         num_samples (int=300000): the target length of the new data
#         gt_dir (str): the directory in which the ground truth is found
#         offset (int=0): number of samples at the beginning to cut off (for data where useful recording starts after sampling has begun)
#     Returns: 
#         None - saves data directly to file
#     """
#     # load data and ground truth
#     data, extension = load_data(data_dir)
#     if gt_dir != None:
#         gt = load_gt(gt_dir)
#     if num_samples+offset >= len(data):
#         raise IndexError(f"Target length exceeds available data. Data is {len(data)} samples, but offset({offset}) + target length ({num_samples}) is {offset+len} samples")
    
#     # create a new directory in the data's parent directory, where cropped data will go.
#     new_dir = get_new_dir(data_dir, num_samples)
#     if not os.path.isdir(new_dir):
#         os.mkdir(new_dir)
#     else:
#         print(f"CROPPING FOLDER: {new_dir} \n FOLDER ALREADY EXISTS. EXISTING DATA IN THAT FOLDER WILL BE OVERWRITTEN")

#     # crop the data and save it
#     d = data[:][offset:offset+num_samples]
#     write_save(d, new_dir + "/continuous" + extension)

#     # crop ground truth and save
#     if gt_dir != None:
#         crop_gt(new_dir, gt, num_samples, offset)
    
#     print(f"CROPPED DATA SAVED TO {new_dir}")

pca
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html