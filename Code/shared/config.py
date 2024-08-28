##DIRECTORIES
#outputs for anything other than base data - images, analyses, etc.
outputs_dir = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data_Outputs" 
#any logs - for purposes of timekeeping, error tracking, etc. 
logger_dir = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Logs"
#data dir. see README for notes on data structure assumptions
data_dir = "E:/EPHYS_DATA"
#url to fetch ground truth datasets from
data_url = "https://janelia.figshare.com/articles/dataset/Simulations_from_kilosort4_paper/25298815/1"

kilosort_version = "4"

datasets = {
    "SIM_HYBRID_10S_ONEDRIVE": { #hybrid
        "data_dir": "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data_Raw/sim_hybrid_ZFM-01936_2021-01-24_CROPPED_10s",
        "n_channels": 385,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat" },
    "SIM_HYBRID_ZFM_FULL": { #hybrid
        "data_dir": "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM-01936_2021-01-24",
        "n_channels": 385,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat" },
    "SIM_HYBRID_ZFM_10S": { #hybrid
        "data_dir": "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM-01936_2021-01-24_CROPPED_10s",
        "n_channels": 385,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat" },
    "SIM_HYBRID_ZFM_1MIN": { #hybrid
        "data_dir": "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM-01936_2021-01-24_CROPPED_1min",
        "n_channels": 385,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat" },
    "SIM_HYBRID_ZFM_10MIN": { #hybrid
        "data_dir": "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM-01936_2021-01-24_CROPPED_10min",
        "n_channels": 385,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat" },
    "JCPM4853_CROPPED_1000ms": { #real data, 1000ms
        "data_dir": "E:/EPHYS_DATA/JCPM4853_CROPPED_1000ms", 
        "n_channels": 384,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat" },
    "JCPM4853_CROPPED_6922ms": { #real data, 6922ms
        "data_dir": "E:/EPHYS_DATA/JCPM4853_CROPPED_6922ms", 
        "n_channels": 384,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat" },
    }

probe_names = [
    'neuropixPhase3A_kilosortChanMap.mat', #384 channels, linear
    'neuropixPhase3B1_kilosortChanMap.mat', #384 channels
    'neuropixPhase3B2_kilosortChanMap.mat', #384 channels
    'NP2_kilosortChanMap.mat',  #384 channels
    'Linear16x1_kilosortChanMap.mat', #16 channels
    ]