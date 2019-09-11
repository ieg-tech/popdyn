import csv
from collections import defaultdict
from string import punctuation
import numpy as np
import dask.array as da
# import dafake as da

# from dask import sharedict, core
# from dask.delayed import Delayed
# from dask.base import tokenize
# import uuid

def rec_dd():
    """Recursively update defaultdicts to avoid key errors"""
    return defaultdict(rec_dd)


def name_key(name):
    """Map a given name to a stripped alphanumeric hash"""
    # Remove white space and make lower-case
    name = name.strip().replace(' ', '').lower()

    try:
        # String
        return name.translate(None, punctuation)
    except:
        # Unicode
        return name.translate(dict.fromkeys(punctuation))


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


def da_where(mask, yes, no):
    """Substitute for da.where for performance enhancement"""

    return da.where(mask, yes, no)


def store(sources, targets):
    """
    Adapted from dask.array.store
    :param sources: sources dask arrays
    :param targets: target data store locations
    :return: None
    """
    # For debugging
    # -------------
    # for source, target in zip(sources, targets):
    #     da.store(source, target, compute=True)
    # return
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


"""
Supplementary module to solve Chronic Wasting Disease simulations

Devin Cairns, 2019
"""
class CWDError(Exception):
    pass


def read_cwd_input(f):
    """
    Read a .csv with a matrix of direct transmission correlations

    :param str f: Input file path
    :return: dict of age group-sex direct transmission correlations
    """
    with open(f) as csvfile:
        data = [row for row in csv.reader(csvfile)]

    # Search for the direct transmission data
    cols = None
    for rows, row in enumerate(data):
        try:
            cols = [_j for _j, e in enumerate(row) if e == 'FROM'][0]
            break
        except IndexError:
            pass

    if cols is None:
        raise CWDError('Unable to parse Direct Transmission data in the CWD input spreadsheet')

    group_row = rows + 1
    sex_row = rows + 2
    rows += 3

    group_col = cols - 2
    sex_col = cols - 1

    # Parse direct transmission data
    direct_transmission = rec_dd()
    for j in range(cols, len(data[group_row])):
        from_gp = name_key(data[group_row][j])
        if len(from_gp) == 0:
            break
        from_sex = name_key(data[sex_row][j])
        for i in range(rows, len(data)):
            to_gp = name_key(data[i][group_col])
            if len(to_gp) == 0:
                break
            to_sex = name_key(data[i][sex_col])
            try:
                value = float(data[i][j])
            except ValueError:
                raise CWDError('Unable to read the value from {}-{} to {}-{}'.format(from_gp, from_sex, to_gp, to_sex))
            direct_transmission[from_gp][from_sex][to_gp][to_sex] = value

    # Search for environmental transmission data
    cols = None
    for rows, row in enumerate(data):
        try:
            cols = [_j for _j, e in enumerate(row) if e == 'C'][0]
            break
        except IndexError:
            pass

    if cols is None:
        raise CWDError('Unable to parse Env. Transmission data in the CWD input spreadsheet')

    group_row = rows - 2
    sex_row = rows - 1
    cols += 1

    # Parse environmental transmission data
    C = defaultdict(dict)
    for j in range(cols, len(data[group_row])):
        gp = name_key(data[group_row][j])
        if len(gp) == 0:
            break
        sex = name_key(data[sex_row][j])
        try:
            value = float(data[rows][j])
        except ValueError:
            raise CWDError('Unable to read the env. transmission value for {}-{}'.format(gp, sex))
        C[gp][sex] = value

    # Change into regular dictionaries
    def update(key, val, d):
        if isinstance(val, defaultdict):
            d[key] = {}
            for _key, _val in val.items():
                update(_key, _val, d[key])
        else:
            d[key] = val

    out_direct_transmission = {}
    for key, val in direct_transmission.items():
        update(key, val, out_direct_transmission)

    out_C = {}
    for key, val in C.items():
        update(key, val, out_C)

    return out_direct_transmission, out_C

