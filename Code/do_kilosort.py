from kilosort.utils import download_probes
from kilosort import run_kilosort, DEFAULT_SETTINGS
import time
import shared.config as config
import shared.logger as logger
import pickle

# CONFIG
dataName = None
dataName = "SIM_HYBRID_10S_ONEDRIVE"
logger_dir = config.logger_dir
# data_base_dir = config.data_dir
setups = config.datasets


def do_kilosort(data_dir, n_channels, probe_name, save_outputs = False): 
    download_probes()
    settings = DEFAULT_SETTINGS
    # ( path to drive if mounted: /content/drive/MyDrive/ )
    settings['data_dir'] = data_dir
    settings['n_chan_bin'] = n_channels
    ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = run_kilosort(settings=settings, probe_name=probe_name)
    if save_outputs:
        file = open(logger_dir + "/pickled_outputs", "wb")
        pickle.dump({"ops": ops, "st": st, "clu": clu, "tF": tF, "Wall": Wall, "similar_templates": similar_templates, "is_ref": is_ref, "est_contam_rate": est_contam_rate, "kept_spikes": kept_spikes}, file)
        file.close()


if __name__ == "__main__":
    t = int(time.time())
    log = logger.set_up_logger(logger_dir + f"/logger_do_kilosort_{t}.txt", 'RUN_KILOSORT')
    if dataName == None or type(dataName) == list:
        # run through all dict keys
        for key in setups.keys():
            if dataName != None:
                if key not in dataName:
                    continue
            print(f"RUNNING KILOSORT FOR {key}")
            d = setups[key]
            data_dir = d["data_dir"]
            n_channels = d["n_channels"]
            probe_name = d["probe_name"]
            try: 
                log.info(f"START: {key}")
                do_kilosort(data_dir, n_channels, probe_name, True)
                log.info(f"SUCCESS: {key}")
            except Exception as e:
                log.exception(e)
    else: 
        #run through single dataset
        d = setups[dataName]
        data_dir = d["data_dir"]
        n_channels = d["n_channels"]
        probe_name = d["probe_name"]
        log.info(f"START: {dataName}")
        do_kilosort(data_dir, n_channels, probe_name, True)
        log.info(f"SUCCESS: {dataName}")
 