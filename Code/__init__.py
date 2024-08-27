import matplotlib.pyplot as plt
import offline.check_outputs as co
import do_kilosort

# d1 = co.directory + '/ZFM_SIM_full'
# d2 = co.directory + '/ZFM_SIM_10min'
# d3 = co.directory + '/ZFM_SIM_1min'
# d4 = co.directory + '/ZFM_SIM_10sec'
# (st0, cl0, wfs0) = co.load_ground_truth(d1)
# (_, st1, cl1, wfs1) = co.load_kilosort_output(d1)
# (_, st2, cl2, wfs2) = co.load_kilosort_output(d2)
# (_, st3, cl3, wfs3) = co.load_kilosort_output(d3)
# (_, st4, cl4, wfs4) = co.load_kilosort_output(d4)

# m01 = co.align_spike_times(st0, st1)
# m01b = co.align_spike_times(st0, st1, tolerance = 2)
# m12 = co.align_spike_times(st1, st2)

# fig, axs = plt.subplots(5, 12, sharey=True)
# for i, axl in enumerate(axs):
#     for j, ax in enumerate(axl):
#         ax.plot(wfs1[0][12*i+j])

# crop_end = 0
# for spike in wfs1[0]:
#     for i in range(len(spike) - 1, 0, -1):
#         if spike[i] != 0.0:
#             if i > crop_end:
#                 crop_end = i
#             continue

outputs = co.inbuilt_gt()
