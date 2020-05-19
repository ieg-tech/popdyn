"""
Used as a dask replacement for debugging purposes
"""


from numpy import *
from numpy import where as npwhere
from numpy import random as nprandom
from numpy import min as npmin
from numpy import max as npmax
from numpy import pad as nppad
from numpy import concatenate as npconcat
from numpy import atleast_3d as np3d
from numpy import atleast_1d as np1d
from numpy import dstack as npdstack
from dask.array.core import normalize_chunks


CHUNKS = [5000, 5000]


class Array(ndarray):
    def __init__(self, shape):
        super(Array, self).__init__(shape)

    def map_blocks(self, method, *args, **kwargs):
        args = (self,) + args
        return map_blocks(method, *args, **kwargs)

    def compute(self):
        return self

    def rechunk(self, chunks=None):
        return self

    @property
    def chunks(self):
        return normalize_chunks(CHUNKS, shape=self.shape)


def from_array(a, chunks=None):
    data = asarray(a)
    nda = Array(data.shape)
    nda[...] = data[...]
    return nda


def from_delayed(a, shape=None, dtype=None):
    return from_array(a.compute())


def map_blocks(method, *args, **kwargs):
    return method(*args).astype(kwargs.get('dtype', 'float32'))


def store(sources, targets, **kwargs):
    for _source, target in zip(sources, targets):
        target[:] = _source


def where(mask, yes, no):
    return from_array(npwhere(mask, yes, no))


def zeros(shape, dtype='float32', chunks=None):
    a = Array(shape).astype(dtype)
    a[:] = 0
    return a


def zeros_like(a, dtype='float32', chunks=None):
    a = Array(a.shape).astype(dtype)
    a[:] = 0
    return a


def ones(shape, dtype='float32', chunks=None):
    a = Array(shape).astype(dtype)
    a[:] = 1
    return a


def min(*args, **kwargs):
    return from_array(npmin(*args))


def max(*args, **kwargs):
    return from_array(npmax(*args))


def pad(*args, **kwargs):
    return from_array(nppad(*args))


def concatenate(*args, **kwargs):
    return from_array(npconcat(*args))


def atleast_3d(*args, **kwargs):
    return from_array(np3d(*args))


def atleast_1d(*args, **kwargs):
    return from_array(np1d(*args))


def dstack(*args, **kwargs):
    return from_array(npdstack(*args))


class random(object):
    @staticmethod
    def normal(*args, **kwargs):
        return from_array(nprandom.normal(*args))

    @staticmethod
    def random(*args, **kwargs):
        return from_array(nprandom.random(*args))
