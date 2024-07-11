import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import mmap


#os.chdir('B:/SpikeData/JCPM4853/JCPM4853')
#directory = 'B:/SpikeData/JPCM04855/JPCM04855/2022-10-31_14-01-34/Record_Node_101/experiment1/recording1/continuous/JPCM04855/Neuropix-PXI-100.ProbeA-AP'
directory  = 'B:/SpikeData/JCPM4853/JCPM4853'
sample_numbers = np.load(os.path.join(directory, 'sample_numbers.npy'))
timestamps = np.load(os.path.join(directory, 'timestamps.npy'))

#ref: https://community.brain-map.org/t/how-to-download-raw-data-from-neuropixels-public-datasets/1923/3
num_channels = 384
sampling_frequency = 30000
num_timepoints = len(timestamps)
data = np.memmap(os.path.join(directory, 'continuous.dat'), mode='r', dtype='int16')
data = data.reshape((len(data) // num_channels, num_channels))

# plt.plot(timestamps[610000:710000], data[:,0][610000:710000])
# plt.show()

#DATA FROM ABOUT 610000 IS THE GOOD STUFF
#NP samples at 30kHz, meaning 1 second will be about 30,000 samples

# def memmap_save(data, chunk_size, output_file):
#     num_chunks = len(data) // chunk_size + 1
#     print(f"num chunks: {num_chunks}")
#     with open(output_file, 'wb') as f:
#         for i in range(num_chunks):
#             start = time.time()
#             chunk_start = i * chunk_size
#             chunk_end = min((i + 1) * chunk_size, len(data))
#             chunk = data[chunk_start:chunk_end]
#             shape = chunk.shape
#             chunk = chunk.tolist()
#             np.memmap(f, dtype=data.dtype, mode='w+', shape=shape)[...] = chunk
#             now = time.time()
#             print(f"saved chunk {i}/{num_chunks}. time: {now - start} seconds")

# def mmap_save(data, chunk_size, output_file):
#     num_chunks = len(data) // chunk_size + 1
#     print(f"num chunks: {num_chunks}")
#     with open(output_file, 'wb') as f:
#         for i in range(num_chunks):
#             start = time.time()
#             chunk_start = i * chunk_size
#             chunk_end = min((i + 1) * chunk_size, len(data))
#             portion_size = chunk_size * data.dtype.itemsize
#             chunk = data[chunk_start:chunk_end]
#             shape = chunk.shape
#             chunk = chunk.tolist()
#             with mmap.mmap(f.fileno(), length=portion_size, access=mmap.ACCESS_WRITE) as out_map:
#                 # Copy the portion of the memmap array to the output file
#                 out_map.write(chunk.tobytes())
#                 now = time.time()
#             print(f"saved chunk {i}/{num_chunks}. time: {now - start} seconds")


# def memmap_save_no_loop(data, output_file):
#     with open(output_file, 'wb') as f:
#         shape = data.shape
#         data = data.tolist()
#         np.memmap(f, dtype=data.dtype, mode='w+', shape=shape)[...] = data

# def mmap_save_no_loop(data, output_file):
#     with open(output_file, 'wb') as f:
#         shape = data.shape
#         length = len(data)*data.dtype.itemsize
#         data = data.tolist()
#         with mmap.mmap(f.fileno(), length=length, access=mmap.ACCESS_WRITE) as out_map:
#                 out_map.write(data.tobytes())

def write_save(data, filename):
    with open(filename, 'wb') as f:
        data.tofile(f)
    f.close()

def save_cropped(data, timestamps, sample_numbers, offset=0, crop_rate = 1000, l = None):
    if l == None:
        l = len(data)//crop_rate
    d = data[:][offset:offset+l]
    t = timestamps[offset:offset+l]
    s = sample_numbers[offset:offset+l]
    dir = directory + f"_CROPPED_{l//30}ms"
    if not os.path.exists(dir):
        os.makedirs(dir)
    write_save(d, dir+"/continuous.dat")
    np.save(dir+"/timestamps.npy", t)
    np.save(dir+"/sample_numbers.npy", s)

def plot(data, timestamps, single=True, crop_rate=10000):
    l = len(data)//crop_rate
    t=timestamps[610000:610000+l]
    d = data[:][610000:610000+l]
    if single:
        plt.plot(t, d[:,0], linewidth=0.5, color='black')
    else:
        for i in range(384):
            dat=[j+600*i for j in d[:,i]]
            plt.plot(t, dat, linewidth=0.5, color='black')
    plt.show()

def plot_heatmap(data, title):
    l = data.shape[0]
    if data.shape[0] > data.shape[1]:
        l = data.shape[1]
    else:
        data = np.transpose(data)
    # ratio of about 1:15 channels:samples per row is good
    l *= 15
    fig, axs = plt.subplots(5)
    for i in range(0,5):
        cropped = data[:][l*i:(l*i)+l]
        cropped = np.transpose(cropped)
        axs[i].plot(cropped)
    plt.title(title)
    plt.show()


save_cropped(data, timestamps, sample_numbers, offset= 610000, crop_rate=1000)