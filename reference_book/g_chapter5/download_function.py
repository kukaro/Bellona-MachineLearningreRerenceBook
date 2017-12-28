import shutil
import os
from urllib.request import urlopen
from reference_book.g_chapter5.disk_cashing_decorator import ensure_directory


def download(url, directory, filename=None):
    """
    Download a file and return its filename on the local file system. If the file is already there, it will not be downloaded again. The filename is derived from the url if not provided. Return thre filepath.
    :param url:
    :param directory:
    :param filename:
    :return: filepath
    """
    if not filename:
        _, filename = os.path.split(url)
        directory = os.path.expanduser(directory)
        ensure_directory(directory)
        filepath = os.path.join(directory, filename)

        if os.path.isfile(filepath):
            return filepath
        print('Download', filepath)
        with urlopen(url) as response, open(filepath, 'wb') as file_:
            shutil.copyfileobj(response, file_)
        return filepath
