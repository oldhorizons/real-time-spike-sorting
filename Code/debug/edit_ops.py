import os
import numpy as np

os.chdir('C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data_Outputs/ZFM_SIM_full')
ops = np.load("gt_ops.npy", allow_pickle = True).item()
ops['nwaves'] = 6
np.save('gt_ops.npy', ops, allow_pickle = True)
