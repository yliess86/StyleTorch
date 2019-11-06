"""Download

The file contains methods to download training data from the coco [1] dataset.

[1] http://images.cocodataset.org
"""
import os

def download_coco( dst: str ) -> None:
    """Download Coco

    Parameters
    ----------
    dst: str
         Destination folder for the Coco dataset
    """
    os.system( f'''
        cd { dst }
        wget http://images.cocodataset.org/zips/train2017.zip && \
        unzip train2017.zip && \
        rm train2017.zip
    ''' )
