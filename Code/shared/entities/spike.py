import numpy as np
import shared.config as config

#shared across all instances of the class
ops_data_dir = config.data_dir
ops = np.load(ops_data_dir + "/ops.npy", allow_pickle = True).item()
pca_axes = None

class Spike:
    def __init__(self, data, id=None, location=None):
        """
        location = channel in which the spike is largest
        data takes form numpy.ndarray[[]] where first dimension (inner array) is time, 
            and second dimension (ith item in inner array) is channel
        """
        self.data = data
        self.location = location
        self.id = id
        self.pca = None

    def calculate_pca(self):
        if self.pca == None:
            #TODO calculate PCA-adjusted version of the spike
            pass
    
    def get_pca(self):
        if self.pca == None:
            self.calculate_pca()
        return self.pca

    def get_location(self):
        if self.location == None:
            chan = np.argmax((self.data**2).sum(0), -1) #see kilosort/bench/clu_ypos for more on this TODO validate this works but it should???
            self.location = ops['yc'][chan]
        return self.location