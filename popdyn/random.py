"""
Generate stochasticity in population dynamics simulations

ALCES 2018
"""
import numpy
from util import file_or_array


RANDOM_METHODS = {'normal': numpy.random.normal,
                  'uniform': numpy.random.uniform}

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

    def return_dynamic(f):
        # If key points directly to a dataset, send it back
        if isinstance(f[key], h5py.Dataset):
            return f[key][:]
        keys = f[key].keys()
        if all(['normalloc' in keys, 'normalscale' in keys]):
            # Collect random value using normal distribution
            loc, scale = (f[key + '/normalloc'][:].ravel(),
                          f[key + '/normalscale'][:].ravel())
            if numpy.any(scale > 0):
                _mean = loc.mean()
                _rnd = numpy.random.normal(_mean, scale)  # Use a single mean to get random value as a delta
                a = (loc + _rnd - _mean).reshape(self.shape)
            else:
                a = loc.reshape(self.shape)
            a[a < 0] = 0
        elif all(['uniformupper' in keys, 'uniformlower' in keys]):
            # Random selection within uniform range
            low, high = (f[key + '/uniformlower'][:].ravel(),
                         f[key + '/uniformupper'][:].ravel())
            a = numpy.random.uniform(low, high).reshape(self.shape)
        else:
            raise PopdynError('Unknown data collection method')
        return a

    if file_instance is not None:
        return return_dynamic(file_instance)

    with self.file as f:
        return return_dynamic(f)
