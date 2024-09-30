import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
# from tslearn import metrics as tsm

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

# def dtw_similarity(unit1, unit2):
#     shift, sim = tsm.dtw_path(unit1, unit2)
#     return sim

def load_gt(gt_dir=None):
    if gt_dir == None:
        gt_dir = 'C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/sim_hybrid_ZFM_45m00s'
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

def align_templates_single_chan(ks_unit, gt_unit):
    gtMaxInds = np.unravel_index(np.argmax(gt_unit), gt_unit.shape)
    ksMaxInds = np.unravel_index(np.argmax(ks_unit), ks_unit.shape)
    gtMinInds = np.unravel_index(np.argmin(gt_unit), gt_unit.shape)
    ksMinInds = np.unravel_index(np.argmin(ks_unit), ks_unit.shape)
    if (-gt_unit[gtMinInds[0],gtMinInds[1]] - ks_unit[ksMinInds[0],ksMinInds[1]]) > gt_unit[gtMaxInds[0],gtMaxInds[1]] + ks_unit[ksMaxInds[0],ksMaxInds[1]]:
        # align by minima
        gtChan = gt_unit[gtMinInds[0],:]
        gtInd = gtMinInds[1]
        ksChan = ks_unit[:,ksMinInds[1]]
        ksInd = ksMinInds[0]
    else:
        # align by maxima
        gtChan = gt_unit[gtMaxInds[0],:]
        gtInd = gtMaxInds[1]
        ksChan = ks_unit[:,ksMaxInds[1]]
        ksInd = ksMaxInds[0]
    
    gtOffset = (gtInd - ksInd) if gtInd > ksInd else 0
    ksOffset = (ksInd - gtInd) if ksInd > gtInd else 0
    newGt = gtChan[gtOffset:]
    newKs = ksChan[ksOffset:]
    gtEnd = ksEnd = min(len(ksChan), len(gtChan))
    newGt = newGt[:gtEnd]
    newKs = newKs[:ksEnd]
    return newKs, newGt

def dimensional_max(item):
    return np.unravel_index(np.argmax(item), item.shape)

def dimensional_min(item):
    return np.unravel_index(np.argmin(item), item.shape)

def scatter_1d(wf):
    plt.plot(wf, [range(len(wf))])
    plt.show()

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
    plt.savefig("C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Outputs/visualisations/fma x_bwplot.png")
    plt.clf()

"""
speed takes values slow / fast / all
amps takes values small / large / all
"""
def plot_similarity(method, good_only=False, speed="all", amps="all"):
    labels = []
    sims = []
    speedThresh = np.median(numSpikes)
    ampThresh = np.median(ranges)
    for key in outputs.keys():
        keySims = []
        labels.append(key.split('_')[-1])
        for index, gt_unit in enumerate(wfs):
            # skip some if needed
            if (good_only == True and quality[index] == "mua"):
                continue
            if (speed == 'slow' and numSpikes[index] > speedThresh) or (speed == 'fast' and total_spikes[index] < speedThresh):
                continue
            if (amps == 'small' and ranges[index] > ampThresh) or (amps == 'large' and amplitudes[index] < ampThresh):
                continue
            # only checks cosine similarity of best_ind
            ks_wfs = np.load(baseDir + "/" + key + "/kilosort4/templates.npy")
            ks_unit = ks_wfs[outputs[key]["best_ind"][index]]
            ks_unit, gt_unit = align_templates_single_chan(ks_unit, gt_unit)
            match method:
                # case "DTW":
                #     keySims.append(dtw_similarity(ks_unit, gt_unit))
                #     continue
                # case "dtw":
                #     keySims.append(dtw_similarity(ks_unit, gt_unit))
                #     continue
                case "Cosine":
                    similarity = np.dot(ks_unit, gt_unit)/(np.linalg.norm(ks_unit) * np.linalg.norm(gt_unit))
                    keySims.append(similarity)
                    continue
                case _:
                    similarity = np.dot(ks_unit, gt_unit)/(np.linalg.norm(ks_unit) * np.linalg.norm(gt_unit))
                    keySims.append(similarity)
                    continue
        sims.append(keySims)
    plt.boxplot(sims)
    titleQualifier = []
    if good_only:
        titleQualifier.append("good")
    if speed != "all":
        titleQualifier.append(speed)
    if amps != "all":
        titleQualifier.append(amps)
    titleQualifier = "all units" if (not good_only and speed == amps =="all") else ", ".join(titleQualifier) + " units only"
    plt.title(f"{method} similarity vs recording length - {titleQualifier}")
    plt.xticks(list(range(len(outputs.keys()) + 1))[1:], labels)
    filenameQualifier = "_qall" if not good_only else "_qgood"
    filenameQualifier += f"_s{speed}_a{amps}"
    plt.savefig(f"C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Outputs/visualisations/{method}_sim_boxplot{filenameQualifier}.png")
    plt.clf()

def get_numspikes_spikeorder():
    numSpikes = [0] * 100
    for cluster in clFull:
        numSpikes[cluster] += 1
    sorted = numSpikes.copy()
    sorted.sort()
    spikeOrder = [numSpikes.index(i) for i in sorted]
    return numSpikes, spikeOrder

def get_ranges_rangeorder():
    ranges = []
    for waveform in wfsFull:
        maxIndex = dimensional_max(waveform)
        maxVal = waveform[maxIndex[0], maxIndex[1]]
        minIndex = dimensional_min(waveform)
        minVal = waveform[minIndex[0], minIndex[1]]
        ranges.append(maxVal-minVal)
    sorted = ranges.copy()
    sorted.sort()
    rangeOrder = [ranges.index(i) for i in sorted]
    return ranges, rangeOrder

def get_quality(filename = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/sim_hybrid_ZFM_45m00s/kilosort4/cluster_KSlabel.tsv"):
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                if int(line.split('\t')[0]) in outputs['sim_hybrid_ZFM_45m00s']['best_ind']:
                    labels.append(line.split('\t')[1].strip())
            except:
                continue
    return labels


def plot_all():
    print("PLOTTING FMAX")
    plot_fmaxes()
    print("PLOTTING DTW SIM")
    plot_similarity("DTW")
    print("PLOTTING COSINE SIM")
    plot_similarity("Cosine")
    for s in ["slow", "fast", "all"]:
        for a in ["small", "large", "all"]:
            for g in [True, False]:
                for m in ["Cosine", "DTW"]:
                    print(m + str(g) + a + s)
                    plot_similarity(m, g, s, a)


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

st,cl,wfs,cb,ci = load_gt('C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/VALIDATION_1minfrom15')

stFull,clFull,wfsFull,cbFull,ciFull = load_gt()

numSpikes, spikeOrder = get_numspikes_spikeorder()
ranges, rangeOrder = get_ranges_rangeorder()
quality = get_quality()



# plot_all()
