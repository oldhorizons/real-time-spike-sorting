# Might be Good to Know:

## Config and Data Structure Assumptions

### Data Structure

    data_dir (as in config.py)
    compressed_synthesised_data (from data_url, decompress using [mtscomp](https://github.com/int-brain-lab/mtscomp))
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
            
        
