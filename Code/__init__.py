import matplotlib.pyplot as plt
import offline.validate as v
import do_kilosort as k
import offline.crop_data as c
import os
import shared.config as config
import numpy as np

# data_dir = "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM"
# gt_dir = "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM"

name = 'ODRV_SIM_HYBRID_ZFM_10S'
d = config.datasets[name]

dat = {
        "data_name": name,
        "data_dir": d['data_dir'],
        "n_channels": d['n_channels'],
        "probe_name": d['probe_name']
}

k.many_kilosort([dat])
# v.run_ks_bench(dat["data_dir"], pName=name)