from kilosort.utils import download_probes
from kilosort import run_kilosort, DEFAULT_SETTINGS

#CONFIG
data_dir = 'B:/SpikeData/JCPM4853/CROPPED'
n_channels = 384

# download channel maps for probes
download_probes()

#run kilosort
settings = DEFAULT_SETTINGS
# ( path to drive if mounted: /content/drive/MyDrive/ )
settings['data_dir'] = data_dir
settings['n_chan_bin'] = n_channels

ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = run_kilosort(settings=settings, probe_name='neuropixPhase3B1_kilosortChanMap.mat')