from collections import defaultdict
import numpy as np
import dask.array as da
# import dafake as da
from numba import jit
# from dask import sharedict, core
# from dask.delayed import Delayed
# from dask.base import tokenize
# import uuid

def rec_dd():
    """Recursively update defaultdicts to avoid key errors"""
    return defaultdict(rec_dd)


def dstack(dsts):
    """dask.array.dstack with one array is slow"""
    if len(dsts) > 1:
        return da.dstack(dsts)
    else:
        return da.atleast_3d(dsts[0])


def dsum(dsts):
    """Apply a sum reduction along of all arrays"""
    if len(dsts) == 1:
        return dsts[0]

    out = da.zeros_like(dsts[0])
    for d in dsts:
        out += d

    return out


def dmean(dsts):
    """Apply a mean reduction along of all arrays"""
    if len(dsts) == 1:
        return dsts[0]

    out = da.zeros_like(dsts[0])
    for d in dsts:
        out += d

    return out / len(dsts)


def da_zeros(shape, chunks):
    """Create a dask array of broadcasted zeros"""
    return da.zeros(shape, dtype=np.float32, chunks=chunks)


@jit(nopython=True, nogil=True)
def apply_where(mask, yes, no, yes_scalar, no_scalar, yes_is_scalar, no_is_scalar, dtype):
    shape = mask.shape
    output = np.empty(shape, dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if mask[i, j]:
                if yes_is_scalar:
                    output[i, j] = yes_scalar
                else:
                    output[i, j] = yes[i, j]
            else:
                if no_is_scalar:
                    output[i, j] = no_scalar
                else:
                    output[i, j] = no[i, j]
    return output


def da_where(mask, yes, no):
    """Substitute for da.where for performance enhancement"""
    yes_scalar = 1
    yes_is_scalar = False
    if not isinstance(yes, da.Array):
        yes_is_scalar = True
        yes_scalar = yes
        yes = np.empty(mask.shape, np.bool)
    no_scalar = 1
    no_is_scalar = False
    if not isinstance(no, da.Array):
        no_is_scalar = True
        no_scalar = no
        no = np.empty(mask.shape, np.bool)
    return da.map_blocks(apply_where, mask, yes, no, yes_scalar, no_scalar,
                         yes_is_scalar, no_is_scalar, np.float32, dtype='float32')


def store(sources, targets):
    """
    Adapted from dask.array.store
    :param sources: sources dask arrays
    :param targets: target data store locations
    :return: None
    """
    # For debugging
    # -------------
    for source, target in zip(sources, targets):
        da.store(source, target, compute=True)
    return
    # -------------

    da.store(sources, targets, compute=True)

    # ---------- OLD ----------- #

    # For debugging
    # -------------
    # for source, target in zip(sources, targets):
    #     target[:] = source.compute()
    # return
    # -------------

    # Optimize all sources together
    # sources_dsk = sharedict.merge(*[e.__dask_graph__() for e in sources])
    # sources_dsk = da.core.Array.__dask_optimize__(
    #     sources_dsk,
    #     list(core.flatten([e.__dask_keys__() for e in sources]))
    # )
    #
    # # Optimize all targets together
    # targets2 = targets
    # targets_keys = []
    # targets_dsk = []
    #
    # targets_dsk = sharedict.merge(*targets_dsk)
    # targets_dsk = Delayed.__dask_optimize__(targets_dsk, targets_keys)
    #
    # store_keys = []
    # store_dsk = []
    # for tgt, src, reg in zip(targets2, sources, [None] * len(sources)):
    #     src = da.core.Array(sources_dsk, src.name, src.chunks, src.dtype)
    #
    #     each_store_dsk = da.core.insert_to_ooc(
    #         src, tgt, lock=True, region=reg,
    #         return_stored=False, load_stored=False
    #     )
    #
    #     # Ensure the store keys are unique with a uuid, as duplicate graphs will result in missed stores
    #     each_store_dsk = {(key[0] + '-' + str(uuid.uuid4()),) + key[1:]: val
    #                       for key, val in each_store_dsk.items()}
    #
    #     store_keys.extend(each_store_dsk.keys())
    #     store_dsk.append(each_store_dsk)
    #
    # store_dsk = sharedict.merge(*store_dsk)
    # store_dsk = sharedict.merge(store_dsk, targets_dsk, sources_dsk)
    #
    # name = 'store-' + tokenize(*store_keys)
    # dsk = sharedict.merge({name: store_keys}, store_dsk)
    # result = Delayed(name, dsk)
    #
    # result.compute()
