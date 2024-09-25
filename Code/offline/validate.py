import shared.config as config
import numpy as np
import pandas as pd
import shared.logger as log
import matplotlib.pyplot as plt
import kilosort.bench as ksb
import time
import pickle

version = config.kilosort_version
directory = config.data_dir
logger_dir = config.logger_dir
pickle_dir = config.pickle_dir

def load_kilosort_output(data_dir, version = "4"):
    data_dir += "/kilosort" + version
    st = np.load(data_dir + "/spike_times.npy")
    cl = np.load(data_dir + "/spike_clusters.npy")
    wfs = np.load(data_dir + "/templates.npy")
    ops = np.load(data_dir + "/ops.npy", allow_pickle = True).item()
    return st, cl, wfs, ops

def load_ground_truth(gt_dir):
    # load ground truth file
    with np.load(gt_dir + "/sim.imec0.ap_params.npz") as gt:
        # extract spike times, cluster labels, and waveforms
        # nb the file also has cb (best channel) and ci
        st = gt['st'].astype('int64')
        cl = gt['cl'].astype('int64')
        wfs = gt['wfs']

    #load operating parameters, if they exist
    try:
        ops = np.load(gt_dir + "/gt_ops.npy", allow_pickle = True)
        ops = ops.item()
    except FileNotFoundError:
        ops = None
    return st, cl, wfs, ops

def save_ops(dir, ops, target="ksgt"):
    """
    saves ops with changes necessary to make ks benchmarking run. 
    Don't ask me why nwaves is 6. I don't know.
    """
    ops['nwaves'] = 6
    if "ks" in target:
        np.save(dir + "/kilosort4/ops.npy", ops, allow_pickle = True)
    if "gt" in target:
        np.save(dir + "/gt_ops.npy", ops, allow_pickle = True)
    return ops


def load_all(data_dir, gt_dir = None, version = "4", ):
    """
    loads all information necessary for validation
    gt_dir should only be provided if gt lives in a different folder from the ks output
    """
    if gt_dir == None:
        gt_dir = data_dir
    gt_st, gt_cl, gt_wfs, gt_ops = load_ground_truth(gt_dir)
    ks_st, ks_cl, ks_wfs, ks_ops = load_kilosort_output(data_dir, version)
    # gt needs ops to run
    if gt_ops == None:
        ops = save_ops(data_dir, ks_ops)
        gt_ops = ops
    return [gt_st, gt_cl, gt_wfs, gt_ops, ks_st, ks_cl, ks_wfs, ks_ops]

def run_ks_bench(data_dir, gt_dir = None, p = True, pName = ""):
    if gt_dir == None:
        gt_dir = data_dir
    # load in necessary variables
    _, _, _, gt_ops, ks_st, ks_cl, _, ks_ops = load_all(data_dir, gt_dir)
    data_location = data_dir + "/continuous.bin"

    # convert to comparable format
    st_gt, clu_gt, yclu_gt, mu_gt, Wsub_gt, nsp = ksb.load_GT(data_location, gt_ops, gt_dir + "/sim.imec0.ap_params.npz", toff = 20, nmax = 600)
    st_new, clu_new, yclu_new, Wsub_new = ksb.convert_ks_output(data_location, ks_ops, ks_st, ks_cl, toff = 20)

    # perform comparison
    fmax, fmiss, fpos, best_ind, matched_all, top_inds = ksb.compare_recordings(st_gt, clu_gt, yclu_gt, st_new, clu_new, yclu_new)

    #save results
    results = {"fmax": fmax, "fmiss": fmiss, "fpos": fpos, "best_ind": best_ind, "matched_all": matched_all, "top_inds": top_inds}
    if p:
        filename = f"/benchmark_{pName}_{int(time.time())}.pkl"
        # todo pickle_dir not data_dir
        file = open(data_dir + filename, "wb")
        pickle.dump(results, file)
        file.close()
    return results

def load_bench_results(benchfile):
    """
    loads benchmark results from a .pkl file
    TODO
    """
    pass


def get_vars():
    d1 = directory + '/ZFM_SIM_full'
    d2 = directory + '/ZFM_SIM_10min'
    d3 = directory + '/ZFM_SIM_1min'
    d4 = directory + '/ZFM_SIM_10sec'
    (st0, cl0, wfs0, ops0) = load_ground_truth(d1)
    (st1, cl1, wfs1, ops1) = load_kilosort_output(d1)
    (st2, cl2, wfs2, ops2) = load_kilosort_output(d2)
    (st3, cl3, wfs3, ops3) = load_kilosort_output(d3)
    (st4, cl4, wfs4, ops4) = load_kilosort_output(d4)