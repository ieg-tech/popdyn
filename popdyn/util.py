from collections import defaultdict
import numpy as np
import dask.array as da
from dask import sharedict, core
from dask.delayed import Delayed
from dask.base import tokenize
import uuid

def rec_dd():
    """Recursively update defaultdicts to avoid key errors"""
    return defaultdict(rec_dd)


def dstack(dsts):
    """dask.array.dstack with one array is slow"""
    if len(dsts) > 1:
        return da.dstack(dsts)
    else:
        return da.atleast_3d(dsts[0])


def da_zeros(shape, chunks):
    """Create a dask array of broadcasted zeros"""
    return da.from_array(np.broadcast_to(np.float32(0), shape), chunks)


def store(sources, targets):
    """
    Adapted from dask.array.store
    :param sources: sources dask arrays
    :param targets: target data store locations
    :return: None
    """
    # Optimize all sources together
    sources_dsk = sharedict.merge(*[e.__dask_graph__() for e in sources])
    sources_dsk = da.core.Array.__dask_optimize__(
        sources_dsk,
        list(core.flatten([e.__dask_keys__() for e in sources]))
    )

    # Optimize all targets together
    targets2 = targets
    targets_keys = []
    targets_dsk = []

    targets_dsk = sharedict.merge(*targets_dsk)
    targets_dsk = Delayed.__dask_optimize__(targets_dsk, targets_keys)

    store_keys = []
    store_dsk = []
    for tgt, src, reg in zip(targets2, sources, [None] * len(sources)):
        src = da.core.Array(sources_dsk, src.name, src.chunks, src.dtype)

        each_store_dsk = da.core.insert_to_ooc(
            src, tgt, lock=True, region=reg,
            return_stored=False, load_stored=False
        )

        # Ensure the store keys are unique with a uuid, as duplicate graphs will result in missed stores
        unique_store_id = str(uuid.uuid4())
        each_store_dsk = {(key[0] + unique_store_id,) + key[1:]: val
                          for key, val in each_store_dsk.items()}

        store_keys.extend(each_store_dsk.keys())
        store_dsk.append(each_store_dsk)

    store_dsk = sharedict.merge(*store_dsk)
    store_dsk = sharedict.merge(store_dsk, targets_dsk, sources_dsk)

    name = 'store-' + tokenize(*store_keys)
    dsk = sharedict.merge({name: store_keys}, store_dsk)
    result = Delayed(name, dsk)

    result.compute()