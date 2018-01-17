"""
Populations dynamics utilities

ALCES, 2018
"""

import h5py


def file_or_array(file_or_array):
    """
    Parse an input as an h5py file instance/dataset or array
    :param file_or_array: (object) path to file and dataset directory or numpy array
    :return: array-like object
    """
    return h5py.File
