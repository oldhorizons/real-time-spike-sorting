from kilosort.utils import download_probes
from kilosort import run_kilosort, DEFAULT_SETTINGS
import time
import config
import mslog

# CONFIG
dataName = None
dataName = "SIM_HYBRID_ZFM"
logger_dir = config.outputs_dir
data_base_dir = config.data_dir
setups = config.datasets


def do_kilosort(data_dir, n_channels, probe_name): 
    download_probes()
    settings = DEFAULT_SETTINGS
    # ( path to drive if mounted: /content/drive/MyDrive/ )
    settings['data_dir'] = data_dir
    settings['n_chan_bin'] = n_channels
    ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = run_kilosort(settings=settings, probe_name=probe_name)


if __name__ == "__main__":
    t = int(time.time())
    logger = mslog.set_up_logger(logger_dir + f"/logger_do_kilosort_{t}.txt", 'RUN_KILOSORT')
    if dataName == None:
        # run through all dict keys
        for key in setups.keys():
            print(f"RUNNING KILOSORT FOR {key}")
            d = setups[key]
            data_dir = d["data_dir"]
            n_channels = d["n_channels"]
            probe_name = d["probe_name"]
            try: 
                logger.info(f"START: {key}")
                do_kilosort(data_dir, n_channels, probe_name)
                logger.info(f"SUCCESS: {key}")
            except Exception as e:
                logger.exception(e)
    else: 
        #run through single dataset
        d = setups[dataName]
        data_dir = d["data_dir"]
        n_channels = d["n_channels"]
        probe_name = d["probe_name"]
        logger.info(f"START: {dataName}")
        do_kilosort(data_dir, n_channels, probe_name)
        logger.info(f"SUCCESS: {dataName}")
