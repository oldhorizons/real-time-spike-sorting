import os
import numpy as np

filename = "gt_ops.npy"
filename = "kilosort4/ops.npy"

os.chdir('C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data_Outputs/ZFM_SIM_full')
ops = np.load(filename, allow_pickle = True).item()
ops['nwaves'] = 6
# ops['yblk'] = [0., 1929.]
# ops['dshift']
np.save(filename, ops, allow_pickle = True)
