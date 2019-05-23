from numpy import *


class Array(ndarray):
    def __init__(self, shape):
        super(Array, self).__init__(shape)

    def map_blocks(self, method, *args, **kwargs):
        args = (self.a,) + args
        return map_blocks(method, *args, **kwargs)

    def compute(self):
        return asarray(self)


def from_array(a, chunks=None):
    data = a[:]
    shape = a.shape
    nda = Array(shape)
    nda[:] = data
    return nda


def map_blocks(method, *args, **kwargs):
    return method(*args).astype(kwargs.get('dtype', 'float32'))


def store(sources, targets, **kwargs):
    dsts = []


def zeros(shape, dtype='float32', chunks=None):
    a = Array(shape).astype(dtype)
    a[:] = 0
    return a
