import numpy as np
import os
import matplotlib.pyplot as plt
import datetime

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

def get_new_dir(data_dir, l, freq = 30000):
    """
    Creates the filepath to the new cropped directory
    Args: 
        data_dir (str): current data directory
        l (int): target length of data once cropped
        freq (int=30000): sampling frequency of the data, in Hz
    Returns: 
        new_dir(str): new data directory
    """
    dir_list = data_dir.split('/')
    t = l//freq
    suffix = f"{(str(t//60)).zfill(2)}m{str(t%60).zfill(2)}s"
    dir_name = dir_list[-1] + f"_{suffix}"
    dir_list = dir_list[:-1]
    dir_list.append(dir_name)
    return '/'.join(dir_list)

def relabel_cropped_cl(cl, wfs = []):
    """
    Redoes cluster labels to ensure no label is skipped
    NB to future self - I haven't thought super hard about the wfs thing
    because you don't actually need it for the built-in validation function
    
    """
    ucl = np.unique(cl)
    n = len(ucl)
    wfs_remove = []
    j = 0
    for i in range(n):
        # so we don't have to keep iterating over the same labels
        j = i if j < i else j
        # seek next cluster label
        while j not in cl:
            wfs_remove.append(j)
            j += 1
        # replace that cluster label with next available cluster label
        # NB assumes clusters are 1-indexed
        cl[cl == j] = i + 1
    #remove all unused waveforms
    if len(wfs) > 0:
        np.delete(wfs, wfs_remove)
    return cl, wfs

def crop_gt(gt, num_samples, offset):
    """
    crops ground truth
    Args: 
        gt (dict): ground truth. Requires 'st': [int], 'cl': [int], 'wfs': [float] as keys
    Returns: 
        gt_new  (dict): as gt, but with times cropped and labels redone if necessary
    """
    st = gt['st'].astype('int64')
    cl = gt['cl'].astype('int64')
    cl0 = len(np.unique(cl))
    wfs = gt['wfs']
    #find index at which spike time > offset (start position):
    i_start = next(i for i,v in enumerate(st) if v > offset)
    #find the index at which spike time > length of cropped recording
    # NB + 1 here for readability; python list indexing is [inclusive: non-inclusive]
    i_end = next(i for i,v in enumerate(st) if (v > num_samples or i == len(st)-1)) + 1
    st = st[i_start:i_end]
    cl = cl[i_start:i_end]
    if len(np.unique(cl)) <= np.max(cl) or len(np.unique(cl)) < cl0:
        cl, wfs = relabel_cropped_cl(cl, wfs)
    
    return {'st': st, 'cl': cl, 'wfs': wfs}

def write_save(data, filename):
    with open(filename, 'wb') as f:
        data.tofile(f)
    f.close()

def crop_data(data_dir, num_samples = 300000, gt_dir = None, offset = 0):
    """
    crops a dataset to the required length. If gt_dir is given, crops ground truth to the same length for validation purposes
    and copies that gt to the new folder, per structure in README.md
    Args: 
        data_dir (str): the directory in which the raw binary file is found
        num_samples (int=300000): the target length of the new data
        gt_dir (str): the directory in which the ground truth is found
        offset (int=0): number of samples at the beginning to cut off (for data where useful recording starts after sampling has begun)
    Returns: 
        None - saves data directly to file
    """
    # load data and ground truth
    data, extension = load_data(data_dir)
    if gt_dir != None:
        gt = load_gt(gt_dir)
    if num_samples+offset >= len(data):
        raise IndexError(f"Target length exceeds available data. Data is {len(data)} samples, but offset({offset}) + target length ({num_samples}) is {offset+len} samples")
    
    # create a new directory in the data's parent directory, where cropped data will go.
    new_dir = get_new_dir(data_dir, num_samples)
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    else:
        print(f"CROPPING FOLDER: {new_dir} \n FOLDER ALREADY EXISTS. EXISTING DATA IN THAT FOLDER WILL BE OVERWRITTEN")

    # crop the data and save it
    d = data[:][offset:offset+num_samples]
    write_save(d, new_dir + "/continuous" + extension)

    # crop ground truth and save
    if gt_dir != None:
        new_gt = crop_gt(gt, num_samples, offset)
        np.savez(new_dir + "/sim.imec0.ap_params.npz", new_gt)
    
    print(f"CROPPED DATA SAVED TO {new_dir}")

## VISUALISATION FOR DEBUGGING PURPOSES. This one honestly doesn't work all that well
def plot_heatmap(data, title, num_subplots = 1, ratio = 15, filename = None, save_img = False):
    l = data.shape[0]
    orientation = 'vertical'
    if data.shape[0] > data.shape[1]:
        orientation = 'horizontal'
        l = data.shape[1]
    l *= ratio
    match orientation:
        case "horizontal":
            fig, axs = plt.subplots(nrows = num_subplots)
        case "vertical":
            fig, axs = plt.subplots(ncols = num_subplots)
            
    for i in range(0,num_subplots):
        match orientation:
            case "horizontal":
                cropped = data[l*i:(l*i)+l]
            case "vertical":
                cropped = data[:,l*i:(l*i)+l]
                
        axs[i].imshow(cropped, cmap = 'hot', interpolation='nearest')
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
    plt.suptitle(title)
    plt.subplots_adjust(wspace=0, hspace=0)
    if save_img:
        if filename is None:
            print("YOU FORGOT A FILENAME")
            filename = f"{str(datetime.now())}.png"
        plt.savefig("heatmaps/" + filename)
    else:
        plt.show()

#visualisation for debugging purposes. The simpler & sexier one
def show(data, length=2000, filename = None):
    if data.shape[0] < data.shape[1]:
        plt.imshow(data[:,0:length], cmap="hot")
    else:
        plt.imshow(data[0:length], cmap='hot') #interpolation = '
    plt.title(filename)
    plt.tight_layout()
    if filename is not None:
        plt.savefig("heatmaps/" + filename + ".png", dpi=500)
    else:
        plt.show()