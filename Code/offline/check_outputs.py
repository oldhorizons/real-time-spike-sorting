import shared.config as config
import numpy as np
import pandas as pd
import shared.logger as log
import matplotlib.pyplot as plt
import kilosort.bench as ksb
import pickle

version = config.kilosort_version
directory = config.data_dir
logger_dir = config.logger_dir

def load_kilosort_output(dir, version = "4"):
    # have a look in kilosort/simulation.py to confirm these but I have a good feeling at LEAST about st/cl/ci
    dir += "/kilosort" + version + '/'
    st = np.load(dir + "spike_times.npy")
    cl = np.load(dir + "spike_clusters.npy")
    wfs = np.load(dir + "templates.npy")
    return st, cl, wfs

def load_ground_truth(dir):
    dir += '/'
    data = np.load(dir + "sim.imec0.ap_params.npz")
    #TODO - astype from kilosort/bench.py - necessary?
    st = data['st'].astype('int64')
    cl = data['cl'].astype('int64')
    wfs = data['wfs']
    # nb there's also cb and ci
    return st, cl, wfs

def load(dir, version = "4"):
    return [load_ground_truth(dir), load_kilosort_output(dir, version)]

# aligns spike times, with a tolerance of x samples (typically zero)
# returns a list of indices (i1, i2) where i1 is the index in st1 matching the spike at i2 in st2 (hopefully)
# pretty rudimentary - waveforms not accounted for
def align_spike_times(st1, st2, tolerance = 0):
    tolerance = abs(tolerance)
    [i1, i2] = [0, 0]
    matches = []
    while i1 < len(st1) and i2 < len(st2):
        diff = st1[i1] - st2[i2]
        if diff < -tolerance:
            i1 += 1
        elif diff > tolerance:
            i2 += 1
        else:
            matches.append((i1, i2))
            i2 += 1
            i1 += 1
    return matches

def compare_clusters(matches, cl1, cl2):
    cm12 = dict()
    cm21 = dict()
    for match in matches:
        i1 = cl1[match[0]]
        i2 = cl2[match[1]]
        cm12[i1] = cm12.get(i1, {})
        cm12[i1][i2] = cm12[i1].get(i2, 0) + 1
        cm21[i2] = cm21.get(i2, {})
        cm21[i2][i1] = cm21[i2].get(i1, 0) + 1
    return (cm12, cm21)

def compare_good_clusters(matches, cl1, cl2, q1, q2):
    pass

#NB assumes ST1 is the ground truth
def df_spike_times(st1, st2):
    arr = [[],[]]
    for t1 in st1:
        for t2 in st2:
            if t1 < t2:
                arr[0].append(str(t1))
                arr[1].append("")
            elif t2 > t1:
                arr[0].append("")
                arr[1].append(str(t2))
            else:
                arr[0].append(str(t1))
                arr[1].append(str(t2))
    npa = np.array(arr)
    df = pd.DataFrame({"GROUND": npa[0], "TEST": npa[1]})
    return df

def plot_spike_events(spiketimes):
    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots()
    x = [2*i for i in range(len(spiketimes))]
    ax.eventplot(spiketimes, orientation="horizontal", lineoffsets=x, linewidth=0.75)
    plt.show()


def main():
    d1 = directory + '/ZFM_SIM_full'
    d2 = directory + '/ZFM_SIM_10min'
    d3 = directory + '/ZFM_SIM_1min'
    d4 = directory + '/ZFM_SIM_10sec'
    (st0, cl0, wfs0) = load_ground_truth(d1)
    (st1, cl1, wfs1) = load_kilosort_output(d1)
    (st2, cl2, wfs2) = load_kilosort_output(d2)
    (st3, cl3, wfs3) = load_kilosort_output(d3)
    (st4, cl4, wfs4) = load_kilosort_output(d4)

    m01 = align_spike_times(st0, st1)
    m01b = align_spike_times(st0, st1, tolerance = 2)
    m12 = align_spike_times(st1, st2)

def inbuilt_gt():
    #TODO inputs: gt_filename, gt_ops, gt_path, ops, st, clu
    gt_path = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data_Outputs/ZFM_SIM_full/"
    gt_filename = "sim.imec0.ap_params.npz"
    ops = np.load(gt_path + "kilosort4/ops.npy", allow_pickle=True).item()
    ###TODO - IMPORTANT: nwaves is currently 552500 (number of spike times), but may be 100 (number of waveforms)
    gt_ops = np.load(gt_path + "gt_ops.npy", allow_pickle = True).item()
    st, clu, _ = load_kilosort_output(gt_path)

    #TODO gt_ops and ops might be different, not sure wtf filename is meant to be
    st_gt, clu_gt, yclu_gt, mu_gt, Wsub_gt, nsp = ksb.load_GT(gt_filename, gt_ops, gt_path + "sim.imec0.ap_params.npz", toff = 20, nmax = 600)
    #st_gt, clu_gt, yclu_gt, Wsub_gt = kilosort.convert_ks_output(gt_ops, gt_st, gt_clu, toff = 20)
    st_new, clu_new, yclu_new, Wsub_new = ksb.convert_ks_output(ops, st, clu, toff = 20)
    fmax, fmiss, fpos, best_ind, matched_all, top_inds = ksb.compare_recordings(st_gt, clu_gt, yclu_gt, st_new, clu_new, yclu_new)

    file = open(logger_dir + "/pickled_outputs", "wb")
    pickle.dump({"fmax": fmax, "fmiss": fmiss, "fpos": fpos, "best_ind": best_ind, "matched_all": matched_all, "top_inds": top_inds}, file)
    file.close()

    return {"fmax": fmax, "fmiss": fmiss, "fpos": fpos, "best_ind": best_ind, "matched_all": matched_all, "top_inds": top_inds}