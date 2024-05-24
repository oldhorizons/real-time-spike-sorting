import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """ from https://stackoverflow.com/a/53877507 """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

## FULL DATASET (download locally then decompress)
# compressed using mtscomp (https://github.com/int-brain-lab/mtscomp)
# URL = 'https://ibl.flatironinstitute.org/public/mainenlab/Subjects/ZFM-02370/2021-04-28/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.e510da60-53e0-4e00-b369-3ea16c45623a.cbin'

## CROPPED DATASET
URL = 'http://www.kilosort.org/downloads/ZFM-02370_mini.imec0.ap.bin'

download_url(URL, URL.split('/')[-1])