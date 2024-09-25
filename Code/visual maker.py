import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
from tslearn import metrics as tsm
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

def dtw_similarity(unit1, unit2):
    _, sim = tsm.dtw_path(unit1, unit2)
    return sim

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

def plot_fmaxes():
    labels = []
    plots = []
    for key in outputs.keys():
        if "0s" not in key:
            labels.append("45m00s")
        else:
            labels.append(key.split('_')[-1])
        plots.append(outputs[key]['fmax'])
    npPlots = np.array(plots)
    plt.boxplot(plots)
    plt.title("fMax vs Recording Length")
    plt.xticks(list(range(len(outputs.keys()) + 1))[1:], labels)
    plt.savefig("C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Outputs/visualisations/fmax_bwplot.png")
    plt.show()

def plot_similarity(method="Cosine"):
    labels = []
    sims = []
    for key in outputs.keys():
        keySims = []
        labels.append(key.split('_')[-1])
        for index, gt_unit in enumerate(wfs):
            # only checks cosine similarity of best_ind
            ks_wfs = np.load(baseDir + "/" + key + "/kilosort4/templates.npy")
            ks_unit = ks_wfs[outputs[key]["best-ind"][index]]
            match method:
                case "DTW":
                    keySims.append(dtw_similarity(ks_unit, gt_unit))
                case "dtw":
                    keySims.append(dtw_similarity(ks_unit, gt_unit))
                case _:
                    similarity = np.dot(ks_unit, gt_unit)/(np.linalg.norm(ks_unit) * np.linalg.norm(gt_unit))
                    keySims.append(similarity)
    sims.append(keySims)

    plt.boxplot(sims)
    plt.title(f"{method} similarity vs recording length")
    plt.xticks(list(range(len(outputs.keys()) + 1))[1:], labels)
    plt.savefig(f"C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Outputs/visualisations/{method}_sim_boxplot.png")
    plt.show()


def plot_all():
    plot_fmaxes()
    plot_similarity("Cosine")
    plot_similarity("DTW")


# baseDir = "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24"
baseDir = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data"
outputsDir = ""

outputs = dict()
dirList = os.listdir(baseDir)

for d in dirList:
    if "ZFM" in d:
        print(d)
        with open(baseDir + "/" + d + "/benchmark.pkl", 'rb') as f:
            outputs[d] = pkl.load(f)

st,cl,wfs,cb,ci = load_gt()
