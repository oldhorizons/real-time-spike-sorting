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

def load_kilosort_output(dir, version = "4"):
    dir += "/kilosort" + version + '/'
    st = np.load(dir + "spike_times.npy")
    cl = np.load(dir + "spike_clusters.npy")
    wfs = np.load(dir + "templates.npy")
    ops = np.load(dir + "ops.npy", allow_pickle = True).item()
    return st, cl, wfs, ops

def load_ground_truth(dir):
    dir += '/'
    # load ground truth file
    gt = np.load(dir + "sim.imec0.ap_params.npz")
    # extract spike times, cluster labels, and waveforms
    # nb the gt dict also has cb and ci
    st = gt['st'].astype('int64')
    cl = gt['cl'].astype('int64')
    wfs = gt['wfs']
    #load operating parameters, if they exist
    try:
        ops = np.load(dir + "gt_ops.npy", allow_pickle = True)
        ops = ops.item()
    except FileNotFoundError:
        ops = None
    return st, cl, wfs, ops

def load_all(dir, gt_dir = None, version = "4", ):
    """
    loads all information necessary for validation
    gt_dir should only be provided if gt lives in a different folder from the ks output
    """
    if gt_dir == None:
        gt_dir = dir
    gt_st, gt_cl, gt_wfs, gt_ops = load_ground_truth(gt_dir)
    ks_st, ks_cl, ks_wfs, ks_ops = load_kilosort_output(dir, version)
    if gt_ops == None:
        # assume gt hasn't been translated over
        crop_output(dir, gt_dir = gt_dir)
        save_ops(dir, ks_ops)
    return [gt_st, gt_cl, gt_wfs, gt_ops, ks_st, ks_cl, ks_wfs, ks_ops]


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

def relabel_cropped_cl(cl, wfs = None):
    """
    NB to future self - I haven't thought super hard about the wfs thing
    because you don't actually need it for the built-in validation function
    """
    ucl = np.unique(cl)
    n = len(ucl)
    wfs_remove = []
    for i in range(n):
        # so we don't have to keep iterating over the same labels
        j = i if j < i else j
        # seek next cluster label
        while j not in cl:
            wfs_remove.append(j)
            j += 1
        # replace that cluster label with next available cluster label
        # NB assumes clusters are 1-indexed
        cl[cl == j] = i + 1
    #remove all unused waveforms
    if wfs != None:
        np.delete(wfs, wfs_remove)
    return cl, wfs


def crop_output(dir, gt = True, num_samples = None, gt_dir = None):
    """
    Crops the ground truth (or ks output) to match length of cropped data
    - for use in validation of performance with recordings cropped to different lengths
    (i.e. different amounts of training data available)
    """
    print("CROPPING GROUND TRUTH TO MATCH SORTER")
    if gt_dir == None:
        gt_dir = dir
    if num_samples == None:
        # figure out how many samples
        num_channels = 385 if gt else 384
        data = np.memmap(dir + 'continuous.bin', mode='r', dtype='int16')
        data = data.reshape((len(data) // num_channels, num_channels))
        num_samples = len(data)
    if (gt):
        gt = np.load(gt_dir + "/sim.imec0.ap_params.npz")
        st = gt['st'].astype('int64')
        cl = gt['cl'].astype('int64')
        wfs = gt['wfs']
    else:
        st, cl, wfs, _ = load_kilosort_output(gt_dir)

    #find the index at which spike time > length of cropped recording
    # NB + 1 here for readability; python list indexing is non-inclusive
    crop_index = next(i for i,v in enumerate(st) if (v > num_samples or i == len(st)-1)) + 1
    st = st[:crop_index]
    cl = cl[:crop_index]
    wfs = wfs[:crop_index]
    if len(np.unique(cl)) != np.max(cl):
        cl, wfs = relabel_cropped_cl(cl, wfs)

    #save new data
    if (gt):
        gt['st'] = st
        gt['cl'] = cl
        gt['wfs'] = wfs
        np.save(dir + "/sim.imec0.ap_params.npz", gt)
    else:
        np.save(dir + "spike_times.npy", st)
        np.save(dir + "spike_clusters.npy", cl)
        np.save(dir + "templates.npy", wfs)


def run_ks_bench(dir, gt_dir = None, pickle = True):
    if gt_dir == None:
        gt_dir = dir
    # load in necessary variables
    _, _, _, gt_ops, ks_st, ks_cl, _, ks_ops = load_all(dir, gt_dir)
    data_location = dir + "/continuous.bin"

    # convert to comparable format
    st_gt, clu_gt, yclu_gt, mu_gt, Wsub_gt, nsp = ksb.load_GT(data_location, gt_ops, gt_dir + "sim.imec0.ap_params.npz", toff = 20, nmax = 600)
    st_new, clu_new, yclu_new, Wsub_new = ksb.convert_ks_output(data_location, ks_ops, ks_st, ks_cl, toff = 20)

    # perform comparison
    fmax, fmiss, fpos, best_ind, matched_all, top_inds = ksb.compare_recordings(st_gt, clu_gt, yclu_gt, st_new, clu_new, yclu_new)

    #save results
    results = {"fmax": fmax, "fmiss": fmiss, "fpos": fpos, "best_ind": best_ind, "matched_all": matched_all, "top_inds": top_inds}
    if pickle:
        filename = f"benchmark_{int(time.time())}.pkl"
        file = open(logger_dir + "/pickled_outputs/" + filename, "wb")
        pickle.dump(results, file)
        file.close()
    return results


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