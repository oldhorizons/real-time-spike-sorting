import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import offline.crop_data as cd
import pandas as pd
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
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

globalTrackedUnits = [70, 84, 33, 65, 83]
globalUnitLabels = {'LGE': 'High Amplitude', 'SML': 'Low Amplitude', 'FST': 'High Activity', 'SLW': 'Low Activity', 'OVL': 'Overlapping'}
globalColours = {'LGE': 'tab:blue', 'SML': 'tab:orange', 'FST': 'tab:green', 'SLW': 'tab:red', 'OVL': 'tab:purple', 'thresh': 'tab:brown', 'noOvl': 'tab:pink', 'ovl': 'tab:gray', 'ALL': "tab:olive", 'GOOD': 'tab:cyan'}
# https://matplotlib.org/stable/gallery/color/named_colors.html

# def dtw_similarity(unit1, unit2):
#     shift, sim = tsm.dtw_path(unit1, unit2)
#     return sim

def load_gt(gt_dir=None):
    if gt_dir == None:
        gt_dir = 'C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/sim_hybrid_ZFM_45m00s'
    with np.load(gt_dir + "/sim.imec0.ap_params.npz") as gt:
        st = gt['st'].astype('int64') #spike times
        #handle the fact I forgot to correct for this earlier
        for i in range(len(st)):
            st[i] -= 27000000
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
    plt.plot(wf, list(range(len(wf))))
    plt.show()

def bwplot_fmaxes(outputs):
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
def bwplot_similarity(outputs, numSpikes, amplitudes, method, good_only=False, speed="all", amps="all"):
    labels = []
    sims = []
    speedThresh = np.median(numSpikes)
    ampThresh = np.median(amplitudes)
    for key in outputs.keys():
        keySims = []
        labels.append(key.split('_')[-1])
        for index, gt_unit in enumerate(wfs):
            # skip some if needed
            if (good_only == True and quality[index] == "mua"):
                continue
            if (speed == 'slow' and numSpikes[index] > speedThresh) or (speed == 'fast' and numSpikes[index] < speedThresh):
                continue
            if (amps == 'small' and amplitudes[index] > ampThresh) or (amps == 'large' and amplitudes[index] < ampThresh):
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

def linegraph_similarity(outputs, numSpikes, amplitudes, qualities):
    tickLabels = []
    tickValues = []
    lineLabels = ["all units", "good units only", "high activity", "low activity", "high amplitude", "low amplitude"]
    lineValues = {"all units": [], "good units only": [], "high activity": [], "low activity": [], "high amplitude": [], "low amplitude": []}
    lineColours = {"all units": globalColours["ALL"], 
                   "good units only": globalColours["GOOD"], 
                   "high activity": globalColours["FST"], 
                   "low activity": globalColours["SLW"], 
                   "high amplitude": globalColours["LGE"], 
                   "low amplitude": globalColours["SML"]}
    speedThresh = np.median(numSpikes)
    ampThresh = np.median(amplitudes)

    for key in outputs.keys():
        keySims = {"all units": [], "good units only": [], "high activity": [], "low activity": [], "high amplitude": [], "low amplitude": []}
        tickLabel = key.split('_')[-1]
        if tickLabel.startswith("00"):
            tickLabel = tickLabel[3:]
            value = int(tickLabel[:2])
        else:
            tickLabel = tickLabel[:3]
            value = int(tickLabel[:2]) * 60
        tickLabels.append(tickLabel)
        tickValues.append(value)

        for index, gt_unit in enumerate(wfs):
            ks_wfs = np.load(baseDir + "/" + key + "/kilosort4/templates.npy")
            ks_unit = ks_wfs[outputs[key]["best_ind"][index]]
            ks_unit, gt_unit = align_templates_single_chan(ks_unit, gt_unit)
            similarity = np.dot(ks_unit, gt_unit)/(np.linalg.norm(ks_unit) * np.linalg.norm(gt_unit))
            keySims["all units"].append(similarity)
            # is the unit good
            if qualities[index] != "mua":
                keySims["good units only"].append(similarity)
            # is the unit fast
            if numSpikes[index] > speedThresh:
                keySims["high activity"].append(similarity)
            else:
                keySims["low activity"].append(similarity)
            #is the unit high amplitude
            if amplitudes[index] > ampThresh:
                keySims["high amplitude"].append(similarity)
            else:
                keySims["low amplitude"].append(similarity)
            
        for key in lineValues.keys():
            lineValues[key].append(np.mean(keySims[key]))

    for key in lineValues.keys():
        plt.plot(tickValues, lineValues[key], label=key, color=lineColours[key])
    plt.xlabel("Training Time (s)")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.title("Empirical Assessment of Optimal Training Time")
    plt.show()
    # plt.savefig(f"C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Outputs/visualisations/cosine_similarity_averages.png")
    plt.clf()

def get_numspikes_spikeorder(clFull):
    numSpikes = [0] * 100
    for cluster in clFull:
        numSpikes[cluster] += 1
    sorted = numSpikes.copy()
    sorted.sort()
    spikeOrder = [numSpikes.index(i) for i in sorted]
    return numSpikes, spikeOrder

def get_ranges_rangeorder(wfsFull):
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

def get_quality(outputs, filename = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/sim_hybrid_ZFM_45m00s/kilosort4/cluster_KSlabel.tsv"):
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                if int(line.split('\t')[0]) in outputs['sim_hybrid_ZFM_45m00s']['best_ind']:
                    labels.append(line.split('\t')[1].strip())
            except:
                continue
    return labels

def get_ks_outputs(baseDir="C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data"):
    outputs = dict()
    dirList = os.listdir(baseDir)
    for d in dirList:
        if "ZFM" in d:
            with open(baseDir + "/" + d + "/benchmark.pkl", 'rb') as f:
                outputs[d] = pkl.load(f)
    return outputs

#qualifier could be threshpc, latency, accuracy, or ""
def get_bonsai_outputs(baseDir = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/Bonsai_Outputs", qualifier = ""):
    latency_outputs = dict()
    dirList = os.listdir(baseDir)

    for d in dirList:
        if qualifier in d:
            out1 = dict()
            csvList = os.listdir(baseDir + "/" + d)
            for c in csvList:
                arr = np.loadtxt(baseDir + "/" + d + "/" + c, delimiter = ",", dtype = float)

def add_df_blanks(generator, follower):
    newFollower = []
    j = 0
    for i, timestamp in enumerate(generator):
        if j < len(follower) and follower[j] >= timestamp and (i == len(generator-1) or follower[j] <= generator[i+1]):
            newFollower.append(follower[j])
            j += 1
        else:
            newFollower.append(None)
    return newFollower

def generate_master_df_latency(directory):
    dirList = os.listdir(directory)
    pds = dict()
    dirNames = ["01_source.csv",
               "02_select_channels.csv",
               "03_butterworth.csv",
               "04_convert_scale.csv",
               "05_spike_detector.csv",
               "06_compare_templates.csv",
               "06_match_level.csv"]
    columnNames = ["source", "select_channels", "butterworth", "convert_scale", "spike_detector", "compare_templates"]
    for name in dirList:
        pds[name] = pd.read_csv(directory + "/" + name, header=None)
    df = pd.DataFrame()
    df.insert(0, "source_timestamp", pds["01_source.csv"][0].array)
    df.insert(1, "select_channels", pds["02_select_channels.csv"][0].array)
    df.insert(2, "butterworth", pds["03_butterworth.csv"][0].array)
    df.insert(3, "convert_scale", pds["04_convert_scale.csv"][0].array)
    df.insert(4, "spike_detector", pds["05_spike_detector.csv"][0].array)
    if "06_compare_templates.csv" in dirList:
        t = add_df_blanks(pds["05_spike_detector.csv"][0].array, pds["06_compare_templates.csv"][0].array)
        df.insert(5, "compare_templates", t)
    df["chan_select_latency"] = df.apply(lambda x: x[1] - x[0], axis=1)
    df["butterworth_latency"] = df.apply(lambda x: x[2] - x[1], axis=1)
    df["convert_scale_latency"] = df.apply(lambda x: x[3] - x[2], axis=1)
    df["spike_detect_latency"] = df.apply(lambda x: x[4] - x[3], axis=1)
    if "06_compare_templates.csv" in dirList:
        df["compare_templates_latency"] = df.apply(lambda x: x[5] - x[4], axis=1)
    return df

def convert_match_lv(df, maxTimestamp):
    wfChans = {0: 21, 1: 77, 2: 10, 3: 173, 4: 22}
    titleToIndex = {"waveform_chan": 0, "gt_id": 1, "display_unit": 2, "template_chan": 3, "similarity": 4, "time_ms": 5}
    
    gtUnits = [70, 84, 77, 33, 65, 83]
    gtIdentifiers = ["LGE", "SML", "FST", "SLW", "OVL"]
    channels = [21, 77, 10, 173, 22]
    timestamps = [[] for i in range(5)]
    similarities = [[] for i in range(5)]
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # "waveform_chan", "gt_id", "display_unit", "template_chan", "similarity", "time_ms"
        if maxTimestamp != None and row[titleToIndex["time_ms"]] > maxTimestamp:
            break
        i = int(row[titleToIndex["waveform_chan"]])
        if i != row[titleToIndex["display_unit"]]:
            continue
        timestamps[i].append(row[titleToIndex["time_ms"]])
        similarities[i].append(row[titleToIndex["similarity"]])  

    return gtUnits, gtIdentifiers, channels, timestamps, similarities

def generate_master_df_accuracy(directory, gt_st, gt_cl, dataDir='C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/VALIDATION_10MIN_FROM_15', recordingLength=None):
    # waveformchan | gtId | displayUnit | template bestChan | similarityIndex | DateTime
    # in batches of 5 for each
    fileList =  os.listdir(directory)

    stampedSims = pd.read_csv(directory + "/06_match_level.csv", header=None)# ["waveform_chan", "gt_id", "display_unit", "template_chan", "similarity", "time_ms"])
    source = pd.read_csv(directory + "/01_source.csv", header=None)
    thresholds = [-27.0, -36.0, -26.0, -17.0, -73.0]
    firstCrosses = [[] for i in range(5)]
    ticks = [[] for i in range(5)]
    #align bin for first thing that crosses the threshold dataDir
    for i, thresh in enumerate(thresholds):
        dataChan = gt_data[:,i]
        firstCrosses[i] = np.argmax(dataChan < thresholds[i] + 1) // 30
    df = pd.DataFrame()
    df.insert(0, "source_timestamp", source[0].array)
    df.insert(1, "source_ticknum", list(range(0, len(source[0].array)*30, 30)))
    maxTimestamp = None
    if recordingLength != None:
        maxTimestamp = df["source_timestamp"][recordingLength]
    
    gtUnits, gtIdentifiers, channels, timestamps, similarities = convert_match_lv(stampedSims, maxTimestamp)
    #find the offset then find the closest one to that offset (some tolerance?)
    #STOP ONCE YOU GET TO RECORDING_LENGTH
    for i, timestampSet in enumerate(tqdm(timestamps)):
        offset = max(df['source_timestamp'][firstCrosses[i]] - timestampSet[0] - 0.1, 0)
        for j, ts in enumerate(timestampSet):
            idx = df[df["source_timestamp"].gt(ts+offset)].index[0] #return index of first timestamp >= timestamp
            idx = max(idx - 1, 0)
            ticks[i].append(df["source_ticknum"][idx])
            if recordingLength != None and j >= recordingLength: #stop once you get to target recording length
                break
            
    return gtUnits, gtIdentifiers, channels, ticks, similarities

def extract_gt_ticks(st, cl, tracked=[70, 84, 33, 65, 83]):
    ticks = [[] for i in range(len(tracked))]
    tracked = list(tracked)
    for i, tick in enumerate(st):
        if cl[i] in tracked:
            ticks[tracked.index(cl[i])].append(tick)
    return ticks

def gen_all_bonsai(gt_st, gt_cl, dataDir='C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/VALIDATION_10MIN_FROM_15', baseDir="C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/Bonsai_Outputs"):
    for directory in os.listdir(baseDir):
        print(directory)
        newDir = baseDir + "/" + directory
        if "latency" in directory:
            df = generate_master_df_latency(newDir)
            df.to_csv(newDir + "/07_full_latency.csv")   
        else:
            gtUnits, gtIdentifiers, channels, ticks, similarities = generate_master_df_accuracy(newDir, gt_st, gt_cl, dataDir, recordingLength = 60000)
            np.savez(newDir + "/07_full_accuracy.npz", 
                     dtype="object",
                     gtUnits=np.array(gtUnits, dtype="object"),
                     gtIdentifiers=np.array(gtIdentifiers, dtype="object"),
                     channels=np.array(channels, dtype="object"),
                     ticks=np.array(ticks, dtype="object"),
                     sims=np.array(similarities, dtype="object"))

def load_all_bonsai(baseDir = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/Bonsai_Outputs"):
    latencies = []
    latencyNames = []
    accuracies = []
    accuracyNames = []
    for directory in os.listdir(baseDir):
        if "latency" in directory:
            latencyNames.append(directory)
            latencies.append(pd.read_csv(f"{baseDir}/{directory}/07_full_latency.csv"))
        else:
            with open(f"{baseDir}/{directory}/07_full_accuracy.npz", "rb") as f:
                d = np.load(f, allow_pickle=True)
                dic = dict()
                for f in d.files:
                    dic[f] = d[f]
                accuracies.append(dic)
                accuracyNames.append(directory)
    return latencies, latencyNames, accuracies, accuracyNames

def plot_bonsai_latency(latencies, latencyNames):
    categories = ["thresh", "noOvl", "ovl"]
    categoryRenames = {"thresh": "Threshold Crossing", "noOvl": "Cosine (no channel overlap)", "ovl": "Cosine (total channel overlap)"}
    nums = [1, 2, 4, 16, 32, 64, 128] #removed 8 because it was a bit screwy aye
    headers = ['chan_select_latency', 'butterworth_latency', 'convert_scale_latency', 'spike_detect_latency', 'compare_templates_latency']
    dat = [[0 for i in range(len(nums))] for j in range(len(categories))]

    for i, name in enumerate(latencyNames):
        for j, num in enumerate(nums):
            for k, cat in enumerate(categories):
                if cat in name and "_"+str(num)+"units" in name:
                    for h in headers:
                        try:
                            desc = latencies[i].describe()[h]
                            # if desc['count'] != 60000.0:
                            #     print(f"{name} {h}: m{desc['mean']} c{desc['count']}")
                            dat[k][j] += desc["mean"]
                        except:
                            continue

    for i, cat in enumerate(categories):
        plt.plot(nums, dat[i], label=categoryRenames[cat], marker='o', color=globalColours[cat])
    plt.legend()
    plt.xlabel("# Units Tracked")
    plt.ylabel("Total Latency (ms)")
    plt.title("Latency Increase with Units Tracked")
    plt.show()
    # plt.savefig("C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Outputs/visualisations/latency_vs_units.png")
    plt.clf()

def get_detected_labels(gtTicks, detectedTicks, offset=0): #offset just in case of data misalignment earlier in the process
    if gtTicks[-1] > detectedTicks[-1]:
        gtTicks = gtTicks[:np.where(np.array(gtTicks) > detectedTicks[-1])[0][0]]
    labels = []
    fNegTicks = []
    tPos = 0
    fPos = 0
    fNeg = 0
    costMatrix = cdist(np.array([detectedTicks]).T, np.array([gtTicks]).T, 'euclidean')
    detectedInd, gtInd = linear_sum_assignment(costMatrix)
    j = 0
    for i, detectedTick in enumerate(detectedTicks):
        if j < len(detectedInd) and i == detectedInd[j]:
            if abs((detectedTick+offset) - gtTicks[gtInd[j]]) <= 65: #simulated source is 30 ticks wide, refractory period of a neuron is 90 ticks wide
                # true positive
                labels.append("tPos")
                tPos += 1
            else:
                fNegTicks.append(gtTicks[gtInd[j]])
                fNeg += 1
            j += 1
        else:
            labels.append("fPos")
            fPos += 1
    return labels, fNegTicks, tPos, fPos, fNeg

def get_optimal_sim_threshold(detectedTicks, gtTicks, similarities, labels):
    if gtTicks[-1] > detectedTicks[-1]:
        gtTicks = gtTicks[:np.where(np.array(gtTicks) > detectedTicks[-1])[0][0]]
    
    thresholds = list(range(-10, 10, 1))
    for i in range(len(thresholds)):
        thresholds[i] /= 10
    fPos = [0 for i in thresholds]
    tPos = [0 for i in thresholds]
    fNeg = [0 for i in thresholds]
    tNeg = [0 for i in thresholds]
    for i, threshold in enumerate(thresholds):
        for j in range(len(labels)):
            if labels[j] == 'tPos':
                if similarities[j] >= threshold:
                    tPos[i] += 1
                else:
                    fNeg[i] += 1
            if labels[j] == 'fPos':
                if similarities[j] >= threshold:
                    fPos[i] += 1
                else:
                    tNeg[i] += 1
    
    f1 = [0 for i in thresholds]
    for i in range(len(thresholds)):
        tp, fp, fn = tPos[i], fPos[i], fNeg[i]
        denom = tp + 0.5*(fp + fn)
        f1[i] = 0 if denom == 0 else tp / denom
    
    #find optimal threshold by F1 score
    maxInd = np.argmax(f1)
    return thresholds[maxInd], tPos[maxInd], fPos[maxInd], fNeg[maxInd]

def get_single_chan_f_vs_similarity_threshold(detectedTicks, gtTicks, similarities, labels, thresholds):
    #largely the same as get_optimal_sim_threshold EXCEPT NOT WHOAH
    if gtTicks[-1] > detectedTicks[-1]:
        gtTicks = gtTicks[:np.where(np.array(gtTicks) > detectedTicks[-1])[0][0]]
    
    fPos = [0 for i in thresholds]
    tPos = [0 for i in thresholds]
    fNeg = [0 for i in thresholds]
    tNeg = [0 for i in thresholds]
    for i, threshold in enumerate(thresholds):
        for j in range(len(labels)):
            if labels[j] == 'tPos':
                if similarities[j] >= threshold:
                    tPos[i] += 1
                else:
                    fNeg[i] += 1
            if labels[j] == 'fPos':
                if similarities[j] >= threshold:
                    fPos[i] += 1
                else:
                    tNeg[i] += 1
    
    f1 = [0 for i in thresholds]
    fpRatio = [0 for i in thresholds]
    ticksDetectedRatio = [0 for i in thresholds]
    for i in range(len(thresholds)):
        tp, fp, fn = tPos[i], fPos[i], fNeg[i]
        denom = tp + 0.5*(fp + fn)
        f1[i] = 0 if denom == 0 else tp / denom
        fpRatio[i] = 0 if (fp+tp) == 0 else fp / (fp + tp)
        ticksDetectedRatio[i] = 0 if (tp + fn) == 0 else tp / (tp + fn)
    
    #find optimal threshold by F1 score
    maxInd = np.argmax(f1)
    return f1, fpRatio, ticksDetectedRatio

def get_base_f1(labels, length):
    fp = labels.count("fPos")
    tp = labels.count("tPos")
    fn = labels.count("fNeg")
    denom = tp + 0.5*(fp + fn)
    f1 = 0 if denom == 0 else tp / denom
    return [f1 for i in range(length)]

def plot_all_f_vs_similarity_threshold(st, cl, threshAcc):
    trackedUnits = [70, 84, 33, 65, 83]
    unitLabels = {'LGE': 'High Amplitude', 'SML': 'Low Amplitude', 'FST': 'High Activity', 'SLW': 'Low Activity', 'OVL': 'Overlapping'}
    thresholds = list(range(-10, 40, 1))
    for i in range(len(thresholds)):
        thresholds[i] /= len(thresholds)
    gtTicks = extract_gt_ticks(st, cl, trackedUnits) #magic number but it's the only way to make it consistent. would be accuracySims[0]['gtUnits'] but that's got length 6
    for sim in threshAcc:
        if sim['name'] != "RT_accuracy_05m00s":
            continue
        for i, unit in enumerate(trackedUnits):
            if sim['gtIdentifiers'][i] == "OVL":
                continue
            f1, fpRatio, ticksDetectedRatio = get_single_chan_f_vs_similarity_threshold(sim['ticks'][i], gtTicks[i], sim['sims'][i], sim['labels'][i], thresholds)
            
            baseF1 = get_base_f1(sim['labels'][i], len(thresholds))
            # plt.plot(thresholds, fpRatio, 
            #          label=globalUnitLabels[sim['gtIdentifiers'][i]], 
            #          color=globalColours[sim['gtIdentifiers'][i]])
            # plt.plot(thresholds, ticksDetectedRatio, 
            #          color=globalColours[sim['gtIdentifiers'][i]], 
            #          linestyle='dashed')
            plt.plot(thresholds, f1, 
                     label=globalUnitLabels[sim['gtIdentifiers'][i]], 
                     color=globalColours[sim['gtIdentifiers'][i]])
            plt.plot(thresholds, baseF1, 
                     color=globalColours[sim['gtIdentifiers'][i]],
                     linestyle='dashed',
                     linewidth=0.5)
    
    plt.legend()
    plt.xlabel("Similarity Score Threshold")
    plt.ylabel("F1 Score")
    plt.title("Empirical Assessment of Cosine Similarity Threshold")
    plt.show()
    # plt.savefig("C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Outputs/visualisations/sim_threshold_measure.png")
    # plt.clf()


def get_thresh_accuracies(accuracySims, accuracyNames, st, cl, saveDir = None):
    trackedUnits = [70, 84, 33, 65, 83]
    gtTicks = extract_gt_ticks(st, cl, trackedUnits) #magic number but it's the only way to make it consistent. would be accuracySims[0]['gtUnits'] but that's got length 6
    for i, name in enumerate(accuracyNames):
        simulation = accuracySims[i]
        simulation['name'] = name
        simulation['tPos'] = []
        simulation['fPos'] = []
        simulation['fNeg'] =  []
        simulation['labels'] = []
        simulation['optimalThresh'] = []
        simulation['cosTPos'] = []
        simulation['cosFPos'] = []
        simulation['cosFNeg'] = []
        for j, chanGtTicks in enumerate(gtTicks):
            if simulation['gtUnits'][j] not in trackedUnits:
                continue
            labels, fNegTicks, tPos, fPos, fNeg = get_detected_labels(chanGtTicks, simulation['ticks'][j])
            simulation['tPos'].append(tPos)
            simulation['fPos'].append(fPos)
            simulation['fNeg'].append(fNeg)
            simulation['labels'].append(labels)
            if "threshpc" not in name: #will need to deal with sim scores
                similarityThreshold, newTPos, newFPos, newFNeg = get_optimal_sim_threshold(simulation['ticks'][j], chanGtTicks, simulation['sims'][j], labels)
                simulation['optimalThresh'].append(similarityThreshold)
                simulation['cosTPos'].append(newTPos)
                simulation['cosFPos'].append(newFPos)
                simulation['cosFNeg'].append(newFNeg + fNeg)
        accuracySims[i] = simulation
    if saveDir != None:
        with open(saveDir + "/all_thresh_accuracies.pkl", 'wb') as f:
            pkl.dump(accuracySims, f, protocol=pkl.HIGHEST_PROTOCOL)
    return accuracySims

def load_thresh_accuracies(saveDir):
    with open(saveDir + "/all_thresh_accuracies.pkl", 'rb') as f:
        accuracySims = pkl.load(f) #, protocol=pkl.HIGHEST_PROTOCOL?
    return accuracySims

def plot_threshold_crossing_accuracy(thresholdSims, unitNums, unitNames):
    f1s = [[] for u in unitNames]
    thresholds = []
    unitLabels = {'LGE': 'High Amplitude', 'SML': 'Low Amplitude', 'FST': 'High Activity', 'SLW': 'Low Activity', 'OVL': 'Overlapping'}
    #get the data out
    for t in thresholdSims:
        thresholds.append(t['value'])
        for i in range(len(unitNames)):
            tp, fp, fn = t['tPos'][i], t['fPos'][i], t['fNeg'][i]
            denom = tp + 0.5*(fp + fn)
            f1 = 0 if denom == 0 else tp / denom
            f1s[i].append(f1)
    
    #plot the data
    for i in range(len(unitNames)):
        if unitNames[i] == "OVL":
            continue
        plt.plot(thresholds, f1s[i], label=unitLabels[unitNames[i]], color=globalColours[unitNames[i]])
    plt.legend()
    plt.xlabel("Amplitude Threshold (Channel Percentile)")
    plt.ylabel("Unit F1 Score")
    plt.title("F1 Score vs Amplitude Threshold")
    # plt.savefig("C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Outputs/visualisations/threshold_crossing_accuracy.png")
    plt.show()
    plt.clf()

def plot_cos_accuracy(cosineSims, unitNums, unitNames): #TODO add optimal F1 scoring thing idk what it's called my brain is tired
    f1s = [[] for u in unitNames]
    thresholds = []
    unitLabels = {'LGE': 'High Amplitude', 'SML': 'Low Amplitude', 'FST': 'High Activity', 'SLW': 'Low Activity', 'OVL': 'Overlapping'}
    #get the data out
    for t in cosineSims:
        thresholds.append(t['value'])
        for i in range(len(unitNames)):
            tp, fp, fn = t['tPos'][i], t['fPos'][i], t['fNeg'][i]
            denom = tp + 0.5*(fp + fn)
            f1 = 0 if denom == 0 else tp / denom
            f1s[i].append(f1)

    #plot the data
    for i in range(len(unitNames)):
        if unitNames[i] == "OVL":
            continue
        plt.plot(thresholds, f1s[i], label=unitLabels[unitNames[i]], color=globalColours[unitNames[i]])
    plt.legend()
    plt.xlabel("Training Time (s)")
    plt.ylabel("Unit F1 Score")
    plt.title("F1 Score vs Training Time")
    plt.show()
    # plt.savefig("C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Outputs/visualisations/cosine_similarity_accuracy.png")
    plt.clf()

def plot_all_online_accuracy(threshAcc):
    thresholdInfo = []
    cosInfo = []
    units = threshAcc[0]['gtUnits']
    unitNames = threshAcc[0]['gtIdentifiers'] #todo check this is right lmao - there might be 6 in here and 6 in the one above
    for acc in threshAcc:
        if 'threshpc' in acc['name']:
            d = dict()
            d['label'] = acc['name'].split('_')[-1]
            d['value'] = float(d['label'])
            d['tPos'] = acc['tPos']
            d['fPos'] = acc['fPos']
            d['fNeg'] = acc['fNeg']
            thresholdInfo.append(d)
        elif "accuracy" in acc['name']:
            d = dict()
            label = acc['name'].split('_')[-1]
            label = label[3:] if label.startswith('00') else label[:3]
            d['label'] = label
            d['value'] = int(label[:2]) if label.endswith('s') else int(label[:2]) * 60
            d['tPos'] = acc['cosTPos']
            d['fPos'] = acc['cosFPos']
            d['fNeg'] = acc['cosFNeg']
            cosInfo.append(d)
        else:
            print(f"AND YOU MAY ASK YOURSELF \nHOW DID I GET HERE \n {acc['name']} \nAND YOU MAY TELL YOURSELF \nTHIS IS NOT MY BEAUTIFUL HOUSE \nAND YOU MAY TELL YOURSELF \nTHIS IS NOT MY BEAUTIFUL WIFE")
    plot_threshold_crossing_accuracy(thresholdInfo, units, unitNames)
    plot_cos_accuracy(cosInfo, units, unitNames)

def plot_all(ksOutputs, numSpikes, ranges, threshAcc, qualities):
    # print("PLOTTING FMAX")
    # bwplot_fmaxes(ksOutputs)
    # print("PLOTTING DTW SIM")
    # plot_similarity(ksOutputs, "DTW")
    # print("PLOTTING COSINE SIM")
    # plot_similarity(ksOutputs, "Cosine")
    linegraph_similarity(ksOutputs, numSpikes, ranges, qualities)
    # for s in ["slow", "fast", "all"]:
    #     for a in ["small", "large", "all"]:
    #         for g in [True, False]:
    #             for m in ["cosine"]:
    #                 print(m + str(g) + a + s)
                    # bwplot_similarity(ksOutputs, numSpikes, ranges, m, g, s, a)
    print("PLOTTING ACCURACY")
    plot_all_online_accuracy(threshAcc)

if __name__ == "__main__":
    baseDir = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data"
    dataDir = 'C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/VALIDATION_10MIN_FROM_15'
    outputsDir = "" #not used

    print("loading ks outputs...")
    ksOutputs = get_ks_outputs()

    print("loading ground truths...")
    st,cl,wfs,cb,ci = load_gt(dataDir)
    stFull,clFull,wfsFull,cbFull,ciFull = load_gt()

    print("loading errata...")
    numSpikes, spikeOrder = get_numspikes_spikeorder(clFull)
    ranges, rangeOrder = get_ranges_rangeorder(wfsFull)
    qualities = get_quality(ksOutputs)
    gt_data = cd.load_data(dataDir)[0]

    print("loading bonsai outputs...")
    latencies, latencyNames, accuracies, accuracyNames = load_all_bonsai(baseDir + "/Bonsai_Outputs")
    # threshAcc = get_thresh_accuracies(accuracies, accuracyNames, st, cl, saveDir = baseDir)
    threshAcc = load_thresh_accuracies(baseDir)
    
    print("plotting...")
    #kilosort
    # linegraph_similarity(ksOutputs, numSpikes, ranges, qualities) #TODO regenerate the data you're putting into this
    # bonsai
    # plot_bonsai_latency(latencies, latencyNames)
    plot_all_f_vs_similarity_threshold(st, cl, threshAcc)
    # plot_all_online_accuracy(threshAcc)

    # plot_all(ksOutputs, numSpikes, ranges, threshAcc, qualities)

