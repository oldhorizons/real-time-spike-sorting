import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import offline.crop_data as cd
import pandas as pd
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
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

def get_outputs(baseDir="C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data"):
    outputs = dict()
    dirList = os.listdir(baseDir)
    for d in dirList:
        if "ZFM" in d:
            print(d)
            with open(baseDir + "/" + d + "/benchmark.pkl", 'rb') as f:
                outputs[d] = pkl.load(f)

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

def extract_gt_ticks(st, cl, tracked=[70, 84, 77, 33, 65, 83]):
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
        print(directory)
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

def get_latency_graph(latencies, latencyNames):
    categories = ["noOvl", "thresh", "ovl"]
    categoryRenames = {"noOvl": "Cosine (distance tolerance=0)", "thresh": "Threshold Crossing", "ovl": "Cosine (distance tolerance=300)"}
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
                            if desc['count'] != 60000.0:
                                print(f"{name} {h}: m{desc['mean']} c{desc['count']}")
                            dat[k][j] += desc["mean"]
                        except:
                            continue

    for i, cat in enumerate(categories):
        plt.plot(nums, dat[i], label=categoryRenames[cat])
    plt.legend()
    plt.xlabel("Units Tracked")
    plt.ylabel("Total Latency (ms)")
    plt.title("Pipeline Latency vs Units Tracked")
    plt.savefig("C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Outputs/visualisations/latency_vs_units.png")
    plt.clf()

def get_detected_labels(gtTicks, detectedTicks):
    latestIndex = 0
    labels = []
    fNegs = []
    tPos = 0
    fPos = 0
    fNeg = 0
    cost_matrix = np.abs(gtTicks[:] - detectedTicks[:])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # # Return the optimal mappings and the cost
    # return row_ind, col_ind, cost_matrix[row_ind, col_ind]
    # for i, tick in enumerate(detectedTicks):
    #     if abs(gtTicks[latestIndex] - tick) >= 30:
    #         if gtTicks[latestIndex] > tick:
    #             fPos += 1
    #             labels.append("fPos")
    #         else:
    #             fNeg += 1
    #             fNegs.append(gtTicks[latestIndex])
    #     elif :
    #         # if this is the closest match
    #         tPos += 1
    #         labels.append("tPos")
    #     else:
    #         # find closest match and carry on like that
    #     pass

def get_thresh_accuracies(accuracies, accuracyNames, st, cl):
    ticks = extract_gt_ticks(st, cl, accuracies[0]['gtUnits'])
    for i, name in enumerate(accuracyNames):
        d = accuracies[i]
        d['name'] = name
        for j, channelTicks in enumerate(ticks):
            labels = get_detected_labels(channelTicks, accuracies[i]['ticks'][j])
            if "threshpc" in name: #no sim scores required
                d["tp"]
                d["fp"]
                d["fn"]
                
            else: #will need to deal with sim scores

                d["optimalSimThreshold"]
                d["tp"]
                d["fp"]
                d["fn"]

        #check accuracies has actually been updated
        continue
        


# baseDir = "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24"
baseDir = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data"
dataDir = 'C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/VALIDATION_10MIN_FROM_15'
outputsDir = ""


outputs = get_outputs()
st,cl,wfs,cb,ci = load_gt(dataDir)

stFull,clFull,wfsFull,cbFull,ciFull = load_gt()

numSpikes, spikeOrder = get_numspikes_spikeorder()
ranges, rangeOrder = get_ranges_rangeorder()
quality = get_quality()
gt_data = cd.load_data(dataDir)[0]

latencies, latencyNames, accuracies, accuracyNames = load_all_bonsai()
threshAcc = get_thresh_accuracies(accuracies, accuracyNames, st, cl)
# get_latency_graph(latencies, latencyNames)

# gen_all_bonsai(st, cl, gt_data)


# dir2 = os.listdir("C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/Bonsai_Outputs")[0]
# f = open(f"{baseDir}/Bonsai_Outputs/{dir2}/07_full_accuracy.npz", 'rb')
# a = np.load(f, allow_pickle=True)
# names = a.files


# plot_all()
