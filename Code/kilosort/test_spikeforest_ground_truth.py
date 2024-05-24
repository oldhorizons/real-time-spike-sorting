import json
import numpy as np
import 

# base directory with ground truth datasets stored
base_dir = 'C:\Users\miche\OneDrive\Documents\A-Uni\REIT4841\Data\HYBRID_JANELIA'

# list of ground truth directories. each directory should have the following: 
# name: recording_name - as the name of the folder
# - metadata: recording_name.json
#     (takes format (as e.g.):
#     {
#         'geom': [[43.0, 160.0], [11.0, 160.0], [57.0, 180.0], [27.0, 180.0]], 
#         'params': {'samplerate': 30000, 'scale_factor': 1, 'spike_sign': -1}, 
#         'raw': 'sha1://acf446faf9adc314699e592471441dd744d33aa9?label=raw.mda', 
#         'self_reference': 'sha1://774b97f88df8b4033e1a0331b538453ba57af03d?label=HYBRID_JANELIA/hybrid_drift_tetrode/rec_4c_600s_11.json'
#     })
# - true firings: recording_name.firings_true.json
# - data: recording_name.npy

ground_truth_datasets = []