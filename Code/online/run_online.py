from shared.entities.spike import Spike
from shared import config
import pickle

# C:\Users\miche\OneDrive\Documents\01 Uni\REIT4841\Data\sim_hybrid_ZFM_10sec

##TODO THIS IS TEMP - CONFIG
def load_config(filename):
    pass
data_dir = config.data_dir
location_tolerance = 10
with open(data_dir + "kilosort4/pca_mode.pkl", "rb") as f:
    pca_model = pickle.load(f)
pca_tolerance = 100 #TODO is there a way to get this from the data?


def load_tracked_templates(trained_templates, clusters_to_track):
    """
    TODO change this to use class
    """
    templates = {}
    for template, i in enumerate(trained_templates):
        if (i+1) in clusters_to_track:
            templates[i+1] = template
    return templates

def preprocess_spike(data_spike):
    """
    converts raw binary data into usable format
    TODO not sure what data format given will be yet, so can't really write this yet
    """
    # whiten (keep a running average of mean for each channel, calculate median across channels at time)
    # TODO?
    # high pass filter (kilosort uses scipy butterworth: b,a = butter(3, 300, fs = 30000, btype = 'high') 
    # TODO?
    #other notes - chucking them here because why not
    # assumption in kilosort is that each waveform is 61 samples long. sounds reasonable & also arbitrary so we'll do the same
    pass



def compare_spikes(data_spike, target_spike):
    """
    returns whether two spikes are sufficiently similar to be considered the same
    """
    #check location is within tolerance, else discard
    if abs(data_spike.get_location() - target_spike.get_location()) > location_tolerance:
        return False
    
    # apply PCA to reduce processing time
    # from kilosort/spikedetect/extract_wPCA_wTEMP
    # requires normalisation first: clips /= (clips**2).sum(1, keepdims=True)**.5
    # model = TruncatedSVD(n_components=ops['settings']['n_pcs']).fit(clips)
    # wPCA = torch.from_numpy(model.components_).to(device).float()
    # saved in ops['wPCA']

    # template match (VERY BASIC) (GONE WRONG) (NOT CLICKBAIT) - are they close enough bro....
    # euclidean distance between PC points with a tolerance


