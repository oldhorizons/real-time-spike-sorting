import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
"""
fmax ([0 < float < 1]): 1 - (fmiss + fpos)
fmiss ([0 < float < 1]): the proportion of ground truth spikes missed
fpos ([0 < float < 1]): the proportion of spikes incorrectly assigned to the gt cluster
best_ind ([int]): detected cluster labels that best match gt clusters, 1:1 mapping
matched_all ([int]): number of spikes in top 20 clusters that match the ground truth
    **NB - because nmatch uses pairwise, rather than holistic, comparison, this will often be >  
    the number of spikes in the ground truth cluster
top_inds ([int]): the cluster labels for the matched_all list
"""


baseDir = "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24"
# baseDir = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/"
outputsDir = ""

outputs = dict()
dirList = os.listdir(baseDir)

for d in dirList:
    if "ZFM" in d:
        print(d)
        with open(baseDir + "/" + d + "/benchmark.pkl", 'rb') as f:
            outputs[d] = pkl.load(f)

def load_gt(gt_dir=None):
    if gt_dir == None:
        gt_dir = 'C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/sim_hybrid_ZFM'
    with np.load(gt_dir + "/sim.imec0.ap_params.npz") as gt:
        st = gt['st'].astype('int64') #spike times
        cl = gt['cl'].astype('int64') #cluster labels
        wfs = gt['wfs'].astype('float64') #waveforms
        cb = gt['cb'] #best channel
        ci = gt['ci']
    return st,cl,wfs,cb,ci

def load_kilosort_output(data_dir):
    data_dir += "/kilosort4"
    st = np.load(data_dir + "/spike_times.npy")
    cl = np.load(data_dir + "/spike_clusters.npy")
    wfs = np.load(data_dir + "/templates.npy")
    ops = np.load(data_dir + "/ops.npy", allow_pickle = True).item()
    return st, cl, wfs, ops

st,cl,wfs,cb,ci = load_gt()

def plot_fmaxes():
    labels = []
    plots = []
    for key in outputs.keys():
        labels.append(key.split('_')[-1])
        plots.append(outputs[key]['fmax'])
    npPlots = np.array(plots)
    plt.boxplot(npPlots, label=labels)
    # plt.savefig("C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Outputs/visualisations/fmax_bwplot.png")
    plt.show()
    
