import config
import numpy as np
import os
import pandas as pd
import mslog

version = config.kilosort_version

['st', 'cl', 'ci', 'wfs', 'cb']

def load_kilosort_output(dir, version):
    # have a look in kilosort/simulation.py to confirm these but I have a good feeling at LEAST about st/cl/ci
    dir += "/kilosort" + version
    st = np.load(dir + "spike_times.npy")
    cl = np.load(dir + "spike_clusters.npy")
    wfs = np.load(dir + "templates.npy")

    m = pd.read_csv(os.path.join(dir, 'cluster_KSLabel.tsv'), sep='\t')   
    is_ref = m['KSLabel'].values=="good"
    nc = np.unique(cl, return_counts=True)[1]
    # 0.5 hz firing rate or higher
    cinds = np.nonzero(is_ref & (nc>ops["Nbatches"]))[0]

    ops = np.load(os.path.join(dir, 'ops.npy'), allow_pickle=True).item()
    templates = np.load(os.path.join(dir, 'templates.npy'))
    wf_cb = ((templates**2).sum(axis=-2)**0.5).argmax(axis=1)
    wf_cb = wf_cb[cinds]
    return st, cl, ci, wfs, cb


def load_ground_truth(dir):
    data = np.load("sim.imec0.ap_params.npz")
    #TODO - astype from kilosort/bench.py - necessary?
    st = data['st'].astype('int64')
    cl = data['cl'].astype('int64')
    ci = data['ci'] 
    wfs = data['wfs']
    cb = 
    return st, cl, ci, wfs, cb

    

# # load kilosort4 output
#     data_folder = os.path.split(filename_bg)[0]
#     ks4_folder = os.path.join(data_folder, 'kilosort4/')   
#     ops = np.load(os.path.join(ks4_folder, 'ops.npy'), allow_pickle=True).item()
#     st = np.load(os.path.join(ks4_folder, 'spike_times.npy'))
#     cl = np.load(os.path.join(ks4_folder, 'spike_clusters.npy'))
#     templates = np.load(os.path.join(ks4_folder, 'templates.npy'))
#     wf_cb = ((templates**2).sum(axis=-2)**0.5).argmax(axis=1)
#     m = pd.read_csv(os.path.join(ks4_folder, 'cluster_KSLabel.tsv'), sep='\t')   
#     is_ref = m['KSLabel'].values=="good"
#     nc = np.unique(cl, return_counts=True)[1]
#     # 0.5 hz firing rate or higher
#     cinds = np.nonzero(is_ref & (nc>ops["Nbatches"]))[0]
    
#     # remove any spikes before padding window for waveform computation
#     cls = cl[st > ntw]
#     sts = st[st > ntw]
#     iref_c = np.isin(cls, cinds)
#     cls = cls[iref_c].astype("int64")
#     sts = sts[iref_c].astype("int64")
#     n_neurons = len(cinds)
#     print('n_neurons = ', len(cinds))
#     wf_cb = wf_cb[cinds]

#     ncm = len(chan_map)
#     wfa = np.zeros((n_neurons, tw, ncm))
#     nst = np.zeros(n_neurons, "int")
#     tic = time.time()