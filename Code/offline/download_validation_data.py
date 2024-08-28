import urllib.request
import shared.config as config
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

## CROPPED DATASET
URL = 'http://www.kilosort.org/downloads/ZFM-02370_mini.imec0.ap.bin'
# all synthetic data
# URL = 'https://janelia.figshare.com/ndownloader/articles/25298815/versions/1'


download_url(URL, URL.split('/')[-1])


"""
DECOMPRESSING THE DATA FILE
usage: mtsdecomp [-h] [-o [OUT]] [--overwrite] [-nc] [-v] cdata [cmeta]

Decompress a raw binary file.

positional arguments:
  cdata                 path to the input compressed binary file (.cbin)
  cmeta                 path to the input compression metadata JSON file (.ch)

optional arguments:
  -h, --help            show this help message and exit
  -o [OUT], --out [OUT] path to the output decompressed file (.bin)
"""