"""
Manipulate datasets used in the popdyn simulation domain. Implemented types are:
 -Lookup tables, which modify values based on bivariate relationships
 -Random number generators using various sampling distributions

Devin Cairns 2018
"""
import numpy as np
from util import file_or_array


RANDOM_METHODS = {'normal': np.random.normal,
                  'uniform': np.random.uniform,
                  'chi-square': np.random.chisquare,
                  'weibull': np.random.weibull}


class DynamicError(Exception):
    pass


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
    return [np.linalg.solve([[x[0], 1.], [x[1], 1.]], [y[0], y[1]]) for x, y in zip(X, Y)]


def random_array(array, method, **kwargs):
    """
    Collect a dataset that requires dynamic values (i.e. random or
    selected based on distribution)
    """
    method = method.lower()
    if method not in RANDOM_METHODS.keys():
        raise ValueError('Unsupported method "{}". '
                         'Choose from one of:\n{}'.format(RANDOM_METHODS, '\n'.join(RANDOM_METHODS.keys())))

    a = file_or_array(array)

    return RANDOM_METHODS[method](a, **kwargs)

def snap_time(f):
    pass

def return_dynamic(f):
    # If key points directly to a dataset, send it back
    if isinstance(f[key], h5py.Dataset):
        return f[key][:]
    keys = f[key].keys()
    if all(['normalloc' in keys, 'normalscale' in keys]):
        # Collect random value using normal distribution
        loc, scale = (f[key + '/normalloc'][:].ravel(),
                      f[key + '/normalscale'][:].ravel())
        if np.any(scale > 0):
            _mean = loc.mean()
            _rnd = np.random.normal(_mean, scale)  # Use a single mean to get random value as a delta
            a = (loc + _rnd - _mean).reshape(self.shape)
        else:
            a = loc.reshape(self.shape)
        a[a < 0] = 0
    elif all(['uniformupper' in keys, 'uniformlower' in keys]):
        # Random selection within uniform range
        low, high = (f[key + '/uniformlower'][:].ravel(),
                     f[key + '/uniformupper'][:].ravel())
        a = np.random.uniform(low, high).reshape(self.shape)
    else:
        raise PopdynError('Unknown data collection method')
    return a

# if file_instance is not None:
#     return return_dynamic(file_instance)
#
# with self.file as f:
#     return return_dynamic(f)

def collect_from_lookup(self, key, a, file_instance=None):
    """
    Return a modified array based on lookup tables from a GID
    """

    def lookup(f):
        x, y = f[key].attrs['lookup'][0, :], f[key].attrs['lookup'][1, :]
        return np.pad(y, 1, 'edge')[1:][np.digitize(a, x)]

    if file_instance is not None:
        return lookup(file_instance)
    else:
        with self.file as f:
            return lookup(f)
