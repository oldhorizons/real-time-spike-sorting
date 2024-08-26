import numpy as np
import os
import matplotlib.pyplot as plt
import datetime

#CONFIG
directory  = 'B:/SpikeData/JCPM4853/JCPM4853'
directory = 'E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM-01936_2021-01-24'
sampling_frequency = 30000

# data for this comes in one of two formats: 
# recorded from the lab 
#   (in which case sample_numbers and timestamps will exist, and ground truth will not exist)
# or downloaded from kilosort test data 'https://janelia.figshare.com/ndownloader/articles/25298815/versions/1' 
#   (in which case sample_numbers will NOT exist, and ground truth WILL exist)
if os.path.isfile(os.path.join(directory, 'sample_numbers.npy')):
    sample_numbers = np.load(os.path.join(directory, 'sample_numbers.npy'))
    timestamps = np.load(os.path.join(directory, 'timestamps.npy'))
    extension = 'dat'
    num_channels = 384
    data = np.memmap(os.path.join(directory, 'continuous.dat'), mode='r', dtype='int16')
    ground_truth = None
else:
    sample_numbers = None
    timestamps = None
    num_channels = 385
    extension = 'bin'
    #TODO check this works
    data = np.memmap(os.path.join(directory, 'continuous.bin'), mode='r', dtype='int16')
    ground_truth = None #TODO

data = data.reshape((len(data) // num_channels, num_channels))

def write_save(data, filename):
    with open(filename, 'wb') as f:
        data.tofile(f)
    f.close()

#save cropped data. length of new data will either be 1/crop rate (if l not given) or l (default 300000 - 10sec recording)
def save_cropped_data(data, timestamps=None, sample_numbers=None, offset=0, crop_rate = None, l = 300000, extension = None, ground_truth = None):
    if l == None:
        l = len(data)//crop_rate
    if l+offset >= len(data):
        raise IndexError("TARGET LENGTH LONGER THAN AVAILABLE DATA")
    dir = directory + f"_CROPPED_{l//30}ms"
    if not os.path.exists(dir):
        os.makedirs(dir)

    d = data[:][offset:offset+l]
    write_save(d, dir+"/continuous." + extension)

    #save cropped timestamps
    if timestamps != None:
        t = timestamps[offset:offset+l]
        np.save(dir+"/timestamps.npy", t)

    #save cropped sample numbers
    if sample_numbers != None:
        s = sample_numbers[offset:offset+l]
        np.save(dir+"/sample_numbers.npy", s)
    
    if ground_truth != None:
        pass
        #TODO or not??


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

save_cropped_data(data, timestamps=timestamps, sample_numbers=sample_numbers, extension=extension, ground_truth=ground_truth, l=18000000)