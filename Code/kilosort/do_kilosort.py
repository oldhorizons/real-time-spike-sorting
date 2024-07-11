from kilosort.utils import download_probes
from kilosort import run_kilosort, DEFAULT_SETTINGS

#CONFIG
setup = 1
if (setup == 1): #real data
    data_dir = 'C:/Users/miche/OneDrive/Documents/A-Uni/REIT4841/Data_Raw/JCPM4853_CROPPED_6922ms'
    n_channels = 384
    probe_name = 'neuropixPhase3B1_kilosortChanMap.mat'
elif (setup == 2): #validation data
    data_dir = 'C:/Users/miche/OneDrive/Documents/A-Uni/REIT4841/Data_Raw/JCPM4853_CROPPED_6922ms'
    n_channels = 16
    probe_name = 'Linear16x1_kilosortChanMap.mat'



# download channel maps for probes
download_probes()

#run kilosort
settings = DEFAULT_SETTINGS
# ( path to drive if mounted: /content/drive/MyDrive/ )
settings['data_dir'] = data_dir
settings['n_chan_bin'] = n_channels

ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = run_kilosort(settings=settings, probe_name=probe_name)