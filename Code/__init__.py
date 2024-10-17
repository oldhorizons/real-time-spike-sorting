import matplotlib.pyplot as plt
import offline.validate as v
import do_kilosort as k
import offline.crop_data as c
import os
import shared.config as config
import numpy as np


name = 'ODRV_SIM_HYBRID_ZFM_10S'
d = config.datasets[name]

data_dir = "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM"
gt_dir = data_dir

# dats = []

# for folder in os.listdir(data_dir):
#     if "30m" not in folder and "FULL" not in folder:
#         dats.append({
#                 "data_name": "ZFM_" + folder.split('_')[-1],
#                 "data_dir": data_dir + "/" + folder,
#                 "n_channels": 385,
#                 "probe_name": 'neuropixPhase3B1_kilosortChanMap.mat'
#         })

# dats.append({
#         "data_name": "ZFM_FULL",
#         "data_dir": data_dir + "/" + "sim_hybrid_ZFM_FULL",
#         "n_channels": 385,
#         "probe_name": 'neuropixPhase3B1_kilosortChanMap.mat'
# })


# new_dir = c.crop_data(data_dir = data_dir, gt_dir = gt_dir, num_samples = 30000*60*10, offset = 30000*60*15)

# for length in [30000*60*2, 30000*60*3, 30000*60*4]:
#     new_dir = c.crop_data(data_dir = data_dir, gt_dir = gt_dir, num_samples = length)
#     new_name = new_dir.split("_")[-1]
#     print(new_name)
#     dat = {
#         "data_name": new_name,
#         "data_dir": new_dir,
#         "n_channels": d['n_channels'],
#         "probe_name": d['probe_name']}
#     k.many_kilosort([dat])
#     v.run_ks_bench(new_dir)

# new_name = "SIM_HYBRID_ZFM_45M"
# new_dir = c.crop_data(data_dir = data_dir, gt_dir = gt_dir, num_samples=81000000)

# dat = {
#         "data_name": new_name,
#         "data_dir": new_dir,
#         "n_channels": d['n_channels'],
#         "probe_name": d['probe_name']
# }

# k.many_kilosort(dats)
# v.run_ks_bench(dat["data_dir"], pName=name)

# v.run_ks_bench(new_dir)

# for dir in os.listdir("E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24"):
#     print(dir)
#     v.run_ks_bench("E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/" + dir)




dat = {
        "data_name": "ZFM_VALIDATION_10MIN",
        "data_dir": "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/VALIDATION_10MIN_FROM_15",
        "n_channels": 385,
        "probe_name": 'neuropixPhase3B1_kilosortChanMap.mat'
}

k.many_kilosort([dat])