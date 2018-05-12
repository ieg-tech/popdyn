"""
Manipulate datasets used in the popdyn simulation domain. Implemented types are:
 -Lookup tables, which modify values based on bivariate relationships
 -Random number generators using various sampling distributions

Devin Cairns 2018
"""
import numpy as np
import dask.array as da
from numba import jit


RANDOM_METHODS = {'normal': np.random.normal,
                  'uniform': np.random.uniform,
                  'chi-square': np.random.chisquare,
                  'weibull': np.random.weibull}


class DynamicError(Exception):
    pass


@jit(nopython=True, nogil=True)
def derive_from_lookup(a, lookup):
    """Perform linear interpolation on the input array using the lookup parameters"""
    shape = a.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            e = a[i, j]
            for row in range(lookup.shape[0]):
                if lookup[row, 0] <= e < lookup[row, 1]:
                    a[i, j] = (e * lookup[row, 2]) + lookup[row, 3]
                    break
                # Upper and lower boundaries
                elif e < lookup[0, 0]:
                    a[i, j] = (lookup[0, 0] * lookup[0, 2]) + lookup[0, 3]
                    break
                elif e >= lookup[-1, 1]:
                    a[i, j] = (lookup[-1, 1] * lookup[-1, 2]) + lookup[-1, 3]
                    if np.isnan(a[i, j]):
                        a[i, j] = 0
                    break
    return a


def apply_random(a, method, **kwargs):
    """
    Generate random values for each domain element using the existing values
    and a random number generator distribution
    :param a: Input arrays
    :param method: random method chosen from globally implemented methods
    :param args: arguments for the random function
    :return: randomly generated values
    """
    # Array should implicitly come from dask

    return RANDOM_METHODS[method](a, **kwargs)


def collect_lookup(input_table):
    """
    Parse an input lookup dictionary.
    :param iterable input_table: Lookup table in the form: [(x1, y1), (x2, y2)...(xn, yn)]
    :return:
    """
    try:
        a = np.asarray(input_table).T
    except:
        raise DynamicError('The input lookup table of type "{}" cannot be read in its current format'.format(
            type(input_table).__name__)
        )

    # Calculate linear regression parameters for each segment of the lookup
    X = zip(a[0, :-1], a[0, 1:])
    Y = zip(a[1, :-1], a[1, 1:])
    return np.array(
        [np.concatenate([x, np.linalg.solve([[x[0], 1.], [x[1], 1.]], [y[0], y[1]])]) for x, y in zip(X, Y)]
    )


def collect(data, **kwargs):
    """
    Route input data through appropriate methods to dynamically derive parameters
    :param data: data for dynamic allocation
    :param kwargs:
        random_method: a random distribution to pick from the globally defined methods
        random_args: arguments to accompany the random method
        lookup_data: array to be filtered through a lookup table
        lookup_table: input lookup table
    :return: dask array object
    """
    if data is None:
        # There must be lookup data
        try:
            lookup_data, lookup_table = kwargs['lookup_data'], kwargs['lookup_table']
        except KeyError:
            raise DynamicError('The kwargs lookup_data and lookup_table are required if no data are provided')

        data = lookup_data.map_blocks(derive_from_lookup, lookup_table, dtype='float32')
    else:
        if not isinstance(data, da.Array):
            raise DynamicError('Input data must be a dask array')

    # Perturb using random values generated from the input distribution
    random_method = kwargs.get('random_method', None)
    random_args = kwargs.get('random_args', None)
    if random_method is not None:
        if random_args is None:
            raise DynamicError('If a random method is supplied, random arguments may not be NoneType')
        random_args.update({'dtype': 'float32'})
        data = data.map_blocks(apply_random, random_method, **random_args)

    return data
