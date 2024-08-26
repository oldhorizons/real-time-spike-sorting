#CONFIG FILE. SICK OF TYPING THIS STUFF OUT.

outputs_dir = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data_Outputs"
logger_dir = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Logs"
# data_dir = "E:/EPHYS_DATA" #can also be "B:/EPHYS_DATA" if needed but only on PC; 
data_dir = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data_Outputs"

data_url = "https://janelia.figshare.com/articles/dataset/Simulations_from_kilosort4_paper/25298815/1"

kilosort_version = "4"

probe_names = [
    'neuropixPhase3A_kilosortChanMap.mat', #384 channels, linear
    'neuropixPhase3B1_kilosortChanMap.mat', #384 channels
    'neuropixPhase3B2_kilosortChanMap.mat', #384 channels
    'NP2_kilosortChanMap.mat',  #384 channels
    'Linear16x1_kilosortChanMap.mat', #16 channels
    ]

datasets = {
    "SIM_HYBRID_10S_ONEDRIVE": {
        "data_dir": "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data_Raw/sim_hybrid_ZFM-01936_2021-01-24_CROPPED_10s",
        "n_channels": 385,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat",
        "type": "sim" },
    "SIM_HYBRID_ZFM_FULL": { #hybrid
        "data_dir": "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM-01936_2021-01-24",
        "n_channels": 385,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat",
        "type": "sim" },
    "SIM_HYBRID_ZFM_10S": { #hybrid
        "data_dir": "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM-01936_2021-01-24_CROPPED_10s",
        "n_channels": 385,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat",
        "type": "sim" },
    "SIM_HYBRID_ZFM_1MIN": { #hybrid
        "data_dir": "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM-01936_2021-01-24_CROPPED_1min",
        "n_channels": 385,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat",
        "type": "sim" },
    "SIM_HYBRID_ZFM_10MIN": { #hybrid
        "data_dir": "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM-01936_2021-01-24_CROPPED_10min",
        "n_channels": 385,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat",
        "type": "sim" },
    "JCPM4853_CROPPED_1000ms": { #real data, 1000ms
        "data_dir": "E:/EPHYS_DATA/JCPM4853_CROPPED_1000ms", #or "C:/Users/miche/OneDrive/Documents/A-Uni/REIT4841/Data_Raw/JCPM4853_CROPPED_1000ms",
        "n_channels": 384,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat",
        "type": "real" },
    "JCPM4853_CROPPED_6922ms": { #real data, 6922ms
        "data_dir": "E:/EPHYS_DATA/JCPM4853_CROPPED_6922ms", #"C:/Users/miche/OneDrive/Documents/A-Uni/REIT4841/Data_Raw/JCPM4853_CROPPED_6922ms",
        "n_channels": 384,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat",
        "type": "real" },
    }
