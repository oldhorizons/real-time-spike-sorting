# trying to data wrangle to get hybrid recordings into appropriate format for kilosort
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

os.chdir('C:/Users/miche/OneDrive/Documents/A-Uni/REIT4841/Data_Raw')
dat1 = np.memmap('JCPM4853_CROPPED_6922ms/continuous.dat')
dat2 = np.memmap('HYBRID_JANELIA/continuous.dat')
# dat1 = np.memmap('JCPM4853_CROPPED_166ms/continuous.dat')
# dat2 = np.memmap('HYBRID_JANELIA_CROPPED_166ms/continuous.dat')
dat1 = dat1.reshape((len(dat1) // 384, 384))
# dat2 = dat2.reshape((len(dat2) //16,16))

def plot_heatmap(data, title, num_subplots = 5, ratio = 15, filename = None, save_img = False):
    l = data.shape[0]
    if data.shape[0] > data.shape[1]:
        l = data.shape[1]
    else:
        # initial transpose because it's really hard to index this data otherwise
        # also kilosort takes data in this format
        #TODO though is this messing with anything? I'm assuming it just changes how np interprets indexes but I'm not 100%
        data = np.transpose(data)
    # ratio of about 1:15 channels:samples per row is good
    l *= ratio
    fig, axs = plt.subplots(num_subplots)
    for i in range(0,num_subplots):
        cropped = data[:][l*i:(l*i)+l]
        cropped = np.transpose(cropped)
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

def show(data, length, filename = None):
    if data.shape[0] < data.shape[1]:
        plt.imshow(data[:,0:length], cmap="hot")
    else:
        plt.imshow(data[0:length], cmap='hot') #interpolation = '
    if filename is not None:
        plt.savefig("heatmaps/" + filename)
    else:
        plt.show()

# show(dat2)

original = np.memmap('HYBRID_JANELIA/continuous.dat', dtype='uint32')
o1c = original.reshape(len(original) //16,16) #C-like index ordering
o1f = original.reshape(len(original) // 16, 16, order='F') #Fortran-like index ordering
o2c = original.reshape(16,len(original)//16)
o2f = original.reshape(16,len(original)//16, order = 'F')

(n1, ratio1) = (5, 20)
(n2, ratio2) = (10, 40)
save = False



print("finished control")
plot_heatmap(o1c, "1C: shape (samples, channels), C-like index ordering", n2, ratio2, f"1C_scC_n{n2}r{ratio2}.png", save)
print("finished 1C")
plot_heatmap(o1f, "1F: shape (samples, channels), Fortran-like index ordering", n2, ratio2, f"1F_scF_n{n2}r{ratio2}.png", save)
print("finished 1F")
plot_heatmap(o2c, "2C: shape (channels, samples), C-like index ordering", n2, ratio2, f"2C_csC_n{n2}r{ratio2}.png", save)
print("finished 2C")
plot_heatmap(o2f, "2F: shape (channels, samples), Fortran-like index ordering", n2, ratio2, f"2F_csF_n{n2}r{ratio2}.png", save)
print("finished 2F")




########################################################################

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

# timestamps = np.load('timestamps.npy')
# sample_numbers = np.load('sample_numbers.npy')
# timestamps = timestamps[:len(dat2)]
# sample_numbers = sample_numbers[:len(dat2)]
# ts1 = np.load('JCPM4853_CROPPED_6922ms/timestamps.npy')
# sn1 = np.load('JCPM4853_CROPPED_6922ms/sample_numbers.npy')

# np.save('HYBRID_JANELIA/timestamps.npy', timestamps)
# np.save('HYBRID_JANELIA/sample_numbers.npy', sample_numbers)



