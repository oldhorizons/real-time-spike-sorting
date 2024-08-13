import json
import numpy as np
import os
from kilosort.utils import download_probes
from kilosort import run_kilosort, DEFAULT_SETTINGS
import kachery_cloud as kcl

# NOTE TO SELF: go to repos/spikeforest/examples/list_all_recordings.py
# and repos/spikeforest/spikeforest/load_spikeforest_recordings.py
# this functionality relies on the kachery cloud protocol set up for spikeforest
uris = {
    'hybrid_janelia': 'sha1://43298d72b2d0860ae45fc9b0864137a976cb76e8?hybrid-janelia-spikeforest-recordings.json',
    'synth_monotrode': 'sha1://3b265eced5640c146d24a3d39719409cceccc45b?synth-monotrode-spikeforest-recordings.json',
    'paired_boyden': 'sha1://849e53560c9241c1206a82cfb8718880fc1c6038?paired-boyden-spikeforest-recordings.json',
    'paired_kampff': 'sha1://b8b571d001f9a531040e79165e8f492d758ec5e0?paired-kampff-spikeforest-recordings.json',
    'paired_english': 'sha1://dfb1fd134bfc209ece21fd5f8eefa992f49e8962?paired-english-spikeforest-recordings.json'
}

probe_names = [
    'neuropixPhase3A_kilosortChanMap.mat', #384 channels, linear
    'neuropixPhase3B1_kilosortChanMap.mat', #384 channels
    'neuropixPhase3B2_kilosortChanMap.mat', #384 channels
    'NP2_kilosortChanMap.mat',  #384 channels
    'Linear16x1_kilosortChanMap.mat', #16 channels
    ]

# base directory with ground truth datasets stored
base_dir = os.path.join('C:\\', 'Users', 'miche', 'OneDrive', 'Documents', 'A-Uni', 'REIT4841', 'Data_Raw') #'C:/Users/miche/OneDrive/Documents/A-Uni/REIT4841/Data_Raw'

#TODO hardcoded for now but make this dynamic - develop a map when downloading the data maybe?
test_data_folders = ['HYBRID_JANELIA']

#download the test data, if it doesn't already exist
def download_test_data():
    pass

def run_tests():
    #todo add a timer file?
    for folder in test_data_folders:
        print(f"running kilosort for: {folder}")
        path = os.path.join(base_dir, folder)
        #check data is in correct (.bin) format, if data doesn't exist, make it from existing files
        if not os.path.isfile(os.path.join(path,'continuous.dat')):
            #todo un-hardcode
            data = np.load(os.path.join(path, 'rec_16c_600s_11.npy'))
            with open(os.path.join(path, 'continuous.dat'), 'wb') as f:
                data.tofile(f)
            f.close()
        
        #run kilosort
        #download channel maps for probes
        download_probes()
        #configure drive location
        settings = DEFAULT_SETTINGS
        settings['data_dir'] = path
        #TODO hardcoded but figure out how to do it dynamically
        settings['n_chan_bin'] = 16
        ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = run_kilosort(settings=settings, probe_name='Linear16x1_kilosortChanMap.mat')
        
        
        """
        copy format from crop data to write as binary file, 
        figure out where num_channels is stored?? 
        then kilosort should just work question mark??
        theN LOOK AT OUTPUT DATA FORMAT AND AUTO VALIIDATION I LOVE YOU
        ALSO FORMAT YOUR CODE YOU GOOSE - automatically download? write documentation fuckin hell lmao

        """
            

def validate_results():
    pass

run_tests()