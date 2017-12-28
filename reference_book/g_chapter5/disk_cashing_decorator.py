import errno
import os
import functools
import pickle


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


def disk_cache(basename, directory, method=False):
    """
    Function decorator for caching pickleable return values on disk. Uses a hash computed from the function arguments for invalidation. If 'method', skip the first argument, usually being self or cls. The cache filepath is 'directory/basename-hash pickle'.
    :param basename:
    :param directory:
    :param method:
    :return:
    """
    directory = os.path.expanduser(directory)
    ensure_directory(directory)

    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            key = (tuple(args), tuple(kwargs.items()))
            # Don't use self or cls for the invalidation hash.
            if method and key:
                key = key[1:]
            '''
            https://wikidocs.net/83
            파일관련 정보
            '''
            filename = '{}-{}.pickle'.format(basename, hash(key))
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as handle:
                    return pickle.load(handle)
            result = func(*args, **kwargs)
            with open(filepath, 'wb') as handle:
                pickle.dump(result, handle)
            return result

        return wrapped

    return wrapper
