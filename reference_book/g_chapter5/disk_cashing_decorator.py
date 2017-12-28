import errno
import os


def ensure_directory(directory):
    """
    Create the directories along the provided directory path that do not exist.
    :param directory:
    :return:
    """
    directory = os.path.expanduser(directory)
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
