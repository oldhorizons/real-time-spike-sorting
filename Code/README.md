## Pipeline

### The Basics

This pipeline uses a fork of [kilosort](https://github.com/MouseLand/Kilosort) - fork available [here](https://github.com/oldhorizons/Kilosort). Major changes:
* adding docstrings to basically any function I touch
* bug fixes to the benchmarking system so it works as intended

### Data Acquisition

Synthesised data is available [here](https://janelia.figshare.com/articles/dataset/Simulations_from_kilosort4_paper/25298815/1), and when downloaded is compressed with the structure found in compressed_synthesised_data. To decompress, use [mtscomp](https://github.com/int-brain-lab/mtscomp). While it is theoretically possible to synthesise your own data using the kilosort library, I'm not doing this. 
You can also run data collected from neuropixels probes through any part of this pipeline. Validation functions are able to take these outputs as 'ground truths' if you want to find comparative (rather than absolute) performance metrics.

### Configuration

The config file lives under shared/config.py, and includes information about data locations, probe names, and honestly anything else I could find that might be useful and hard to find later.

### Offline Sorting

Offline sorting is available under do_kilosort.py, which wraps run_kilosort from the kilosort library in some useful functions like logging with timestamps, automatically going through multiple runs in sequence, and the handy-dandy config file

### Validation

Run it from offline/validate.py. The main function is run_ks_bench, which does pretty much everything, including crop ground truth data if you need it cropped (NB THIS HAS NOT YET BEEN TESTED)

### Running Everything

The program is run from main.py in the code folder. Temporary programs are run from \_\_init\_\_.py, which ensures access to the whole library


## Config and Data Structure Assumptions

### Data Structure

    data_dir (as in config.py)
    compressed_synthesised_data (from data_url, decompress using )
        sim.imec0.ap.cbin
        sim.imec0.ap.ch
        sim.imec0.ap.meta
        sim.imec0.ap_params.npz
    decompressed_synthesised_data
        sim.imec0.ap.meta
        sim.imec0.ap_params.npz
        continuous.bin
    unsynthesised_data
        continuous.dat
        (optional) sample_numbers.npy
        (optional) timestamps.npy
    fully_processed_dataset
        kilosort4
            output.npy
            output.npz
            output.etc
        kilosort4.X (as kilosort4)
        continuous.bin or continuous.dat (the actual binary file)
        (optional) sim.imec0.ap_params.npz (ground truth, if known)
        (optional) gt_ops.npy (ground truth options, generated in validate.py)
            
        
