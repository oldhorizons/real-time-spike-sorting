from kilosort.utils import download_probes
from kilosort import run_kilosort, DEFAULT_SETTINGS
import time
import shared.config as config
import shared.logger as logger
import pickle

# CONFIG
dataName = None
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


def many_kilosort(dataDirs):
    """
    does many kilosorts and logs the results
    
    """
    t = int(time.time())
    log = logger.set_up_logger(logger_dir + f"/logger_do_kilosort_{t}.txt", 'RUN_KILOSORT')
    for d in dataDirs:
        data_name = d["data_name"]
        data_dir = d["data_dir"]
        n_channels = d["n_channels"]
        probe_name = d["probe_name"]
        print(f"RUNNING KILOSORT FOR {data_name}")
        try: 
            log.info(f"START: {data_name}")
            do_kilosort(data_dir, n_channels, probe_name, True)
            log.info(f"SUCCESS: {data_name}")
        except Exception as e:
            log.exception(e)
 