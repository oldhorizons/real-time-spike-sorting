# trying to data wrangle to get hybrid recordings into appropriate format for kilosort
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('C:/Users/miche/OneDrive/Documents/A-Uni/REIT4841/Data_Raw')
dat1 = np.memmap('JCPM4853_CROPPED_6922ms/continuous.dat')
dat2 = np.memmap('HYBRID_JANELIA/continuous.dat')
# dat1 = np.memmap('JCPM4853_CROPPED_166ms/continuous.dat')
# dat2 = np.memmap('HYBRID_JANELIA_CROPPED_166ms/continuous.dat')
dat1 = dat1.reshape((len(dat1) // 384, 384))
# dat2 = dat2.reshape((len(dat2) //16,16))


def write_save(data, filename):
    with open(filename, 'wb') as f:
        data.tofile(f)
    f.close()

def save_cropped_np(data, name, offset=0, crop_rate = 1000, l = None):
    if l == None:
        l = len(data)//crop_rate
    d = data[:][offset:offset+l]
    directory = name + f"_CROPPED_{l//30}ms"
    if not os.path.exists(directory):
        os.makedirs(directory)
    write_save(d, directory+"/continuous.dat")

def plot_heatmap(data, title):
    l = data.shape[0]
    if data.shape[0] > data.shape[1]:
        l = data.shape[1]
    else:
        data = np.transpose(data)
    # ratio of about 1:15 channels:samples per row is good
    l *= 30
    fig, axs = plt.subplots(5)
    for i in range(0,5):
        cropped = data[:][l*i:(l*i)+l]
        cropped = np.transpose(cropped)
        axs[i].imshow(cropped, cmap = 'hot')
    plt.title(title)
    plt.show()

def show(data):
    plt.imshow(data, cmap='hot') #interpolation = 'nearest'
    plt.show()

# show(dat2)

original = np.memmap('HYBRID_JANELIA/continuous.dat')
o1a = original.reshape(len(original) //16,16) #C-like index ordering
o1b = original.reshape(len(original) // 16, 16, order='F') #Fortran-like index ordering
o2a = original.reshape(16,len(original)//16)
o2b = original.reshape(16,len(original)//16, order = 'F')

plot_heatmap(dat1, "control")
plot_heatmap(o1a, "o1a")
plot_heatmap(o1b, "o1b")
plot_heatmap(o2a, "o2a")
plot_heatmap(o2b, "o2b")

# timestamps = np.load('timestamps.npy')
# sample_numbers = np.load('sample_numbers.npy')
# timestamps = timestamps[:len(dat2)]
# sample_numbers = sample_numbers[:len(dat2)]
# ts1 = np.load('JCPM4853_CROPPED_6922ms/timestamps.npy')
# sn1 = np.load('JCPM4853_CROPPED_6922ms/sample_numbers.npy')

# np.save('HYBRID_JANELIA/timestamps.npy', timestamps)
# np.save('HYBRID_JANELIA/sample_numbers.npy', sample_numbers)



