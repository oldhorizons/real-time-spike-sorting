import matplotlib.pyplot as plt
import offline.validate as v
import do_kilosort

dir = "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM-01936_2021-01-24_CROPPED_10min"
gt_dir = "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM-01936_2021-01-24"

outputs = v.run_ks_bench(gt_dir)

outputs2 = v.run_ks_bench(dir)
