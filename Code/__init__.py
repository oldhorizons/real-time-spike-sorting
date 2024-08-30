import matplotlib.pyplot as plt
import offline.validate as v
import do_kilosort as k
import offline.crop_data as c
import os
import shared.config

dir = "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM"
gt_dir = "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM"

d = "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24"
dlist = os.listdir(d)
datas = []

for folder in dlist:
    dat = {
            "data_name": dlist.split('/')[-1],
            "dara_dir": os.path.join(d, folder),
            "n_channels": 385,
            "probe_name": "neuropixPhase3B1_kilosortChanMap.mat"
    }
    if not os.path.isdir(os.path.join(os.path.join(d, folder)), "kilosort4"):
        datas.append(dat)

k.many_kilosort(datas)

for folder in datas:
    v.run_ks_bench(os.path.join(d, folder))