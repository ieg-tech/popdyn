from numpy import *


class Array(ndarray):
    def __init__(self, a):
        super(Array, self).__init__(a)

    def map_blocks(self, method, *args, **kwargs):
        args = (self.a,) + args
        return map_blocks(method, *args, **kwargs)

    def compute(self):
        return asarray(self)


def from_array(a, chunks):
    return Array(a)


def map_blocks(method, *args, **kwargs):
    return method(*args).astype(kwargs.get('dtype', 'float32'))
