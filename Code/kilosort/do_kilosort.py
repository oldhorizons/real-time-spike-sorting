from kilosort.utils import download_probes
from kilosort import run_kilosort, DEFAULT_SETTINGS
import logging
import time

# CONFIG
dataName = None
dataName = "SIM_HYBRID_ZFM"
setups = {
    "SIM_HYBRID_ZFM": { #alternate hybrid
        "data_dir": "E:/EPHYS_DATA/sim_hybrid_ZFM-01936_2021-01-24/sim_hybrid_ZFM-01936_2021-01-24",
        "n_channels": 385,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat" },
    "JCPM4853_CROPPED_1000ms": { #real data, 1000ms
        "data_dir": "C:/Users/miche/OneDrive/Documents/A-Uni/REIT4841/Data_Raw/JCPM4853_CROPPED_6922ms",
        "n_channels": 384,
        "probe_name": "neuropixPhase3B1_kilosortChanMap.mat" },
    "HYBRID_JANELIA_CROPPED_1000ms_C": { #hybrid janelia - C
        "data_dir": "C:/Users/miche/OneDrive/Documents/A-Uni/REIT4841/Data_Raw/HYBRID_JANELIA_CROPPED_1000ms",
        "n_channels": 16,
        "probe_name": "Linear16x1_kilosortChanMap.mat" },
    # "HYBRID_JANELIA_CROPPED_1000ms_F": { #hybrid janelia - encoded differently
    #     "data_dir": "C:/Users/miche/OneDrive/Documents/A-Uni/REIT4841/Data_Raw/HYBRID_JANELIA_CROPPED_1000ms",
    #     "n_channels": 16,
    #     "probe_name": "Linear16x1_kilosortChanMap.mat" },
    # "PAIRED_BOYDEN_CROPPED_1000ms": {
    #     "data_dir": "C:/Users/miche/OneDrive/Documents/A-Uni/REIT4841/Data_Raw/PAIRED_BOYDEN_CROPPED_1000ms",
    #     "n_channels": 32,
    #     "probe_name": "Linear16x1_kilosortChanMap.mat" }, #TODO THIS IS WRONG - FIGURE OUT WHAT THE NAME IS. Future Michelle - there are only mappings for 16 and 384 channels
    }

def do_kilosort(data_dir, n_channels, probe_name): 
    download_probes()
    settings = DEFAULT_SETTINGS
    # ( path to drive if mounted: /content/drive/MyDrive/ )
    settings['data_dir'] = data_dir
    settings['n_chan_bin'] = n_channels
    ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = run_kilosort(settings=settings, probe_name=probe_name)

def setup_logger(filename):
    # https://stackoverflow.com/questions/55169364/python-how-to-write-error-in-the-console-in-txt-file
    logger = logging.getLogger('my_application')
    logger.setLevel(logging.INFO) # you can set this to be DEBUG, INFO, ERROR
    # Assign a file-handler to that instance
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO) # again, you can set this differently
    # Format your logs (optional)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter) # This will set the format to the file handler
    # Add the handler to your logging instance
    logger.addHandler(fh)
    return logger
    # try:
    #     raise ValueError("Some error occurred")
    # except ValueError as e:
    #     logger.exception(e) # Will send the errors to the file

    
if __name__ == "__main__":
    if dataName == None:
        # run through all dict keys
        t = int(time.time())
        with open(f"C:/Users/miche/OneDrive/Documents/A-Uni/REIT4841/Data_Raw/progresslog{t}.txt", "w+") as progresslog:
            for key in setups.keys():
                print(f"RUNNING KILOSORT FOR {key}")
                d = setups[key]
                data_dir = d["data_dir"]
                n_channels = d["n_channels"]
                probe_name = d["probe_name"]
                try: 
                    do_kilosort(data_dir, n_channels, probe_name)
                    progresslog.write(f'SUCCESS: {key}\n')
                except Exception as e:
                    progresslog.write(f'FAILURE: {key}\n')
                    progresslog.write(repr(e))
                    progresslog.write("\n")
            progresslog.close()
    else: 
        #run through single dataset
        d = setups[dataName]
        data_dir = d["data_dir"]
        n_channels = d["n_channels"]
        probe_name = d["probe_name"]
        do_kilosort(data_dir, n_channels, probe_name)

