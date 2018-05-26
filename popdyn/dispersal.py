"""
Relocate mass using a variety of dispersal algorithms

Devin Cairns, 2018
"""

import numpy as np
import dask.array as da
from numba import jit
from scipy import ndimage


class DispersalError(Exception):
    pass


def apply(a, total, capacity, method, args, **kwargs):
    """Apply a dispersal method on the input dask array"""
    if not isinstance(a, da.Array):
        raise DispersalError('The input array must be a dask array')

    if method not in METHODS:
        raise DispersalError('The input method "{}" is mot supported'.format(method))

    return METHODS[method](a, total, capacity, *args, **kwargs)


def calculate_kernel(distance, csx, csy, outer_ring=False):
    """Calculate kernel index offsets using the distance and cell size"""
    # Calculate the kernel matrix size
    m = np.uint64(np.round(distance / csy))
    n = np.uint64(np.round(distance / csx))

    if m == 0 or n == 0:
        # Unable to traverse to adjacent elements
        return

    # Use a distance transform to select the active grid locations
    kernel = np.ones(shape=(int(m * 2), int(n * 2)), dtype='bool')
    kernel[m, n] = 0
    kernel = ndimage.distance_transform_edt(kernel, (csy, csx))
    kernel = (kernel <= distance)

    if outer_ring:
        kernel = ~(ndimage.binary_erosion(kernel, np.ones((3, 3), 'bool'))) & kernel

    # Create an offset matrix
    i, j = np.asarray(np.where(kernel))
    i -= m
    j -= n

    return np.asarray([i, j]).T, m, n


def pd_trim_internal():
    """
    Borrowed from da.trim_internal, refactored to add the edges of ghosted chunks together
    :return: dask array via map_blocks
    """
    olist = []
    for i, bd in enumerate(x.chunks):
        ilist = []
        for d in bd:
            ilist.append(d - axes.get(i, 0) * 2)
        olist.append(tuple(ilist))

    chunks = tuple(olist)

    return map_blocks(partial(chunk.trim, axes=axes), x, chunks=chunks,
                      dtype=x.dtype)


"""
Density flux, also known as inter-habitat dispersal. Calculates a mean density over a neighbourhood and
reallocates populations within the neighbourhood in attempt to flatten the gradient.
=======================================================================================================
"""

def density_flux(population, total_population, carrying_capacity, distance, csx, csy, **kwargs):
    # Check the inputs
    if any([not isinstance(a, da.Array) for a in [population, total_population, carrying_capacity]]):
        raise DispersalError('Inputs must be a dask arrays')

    if distance == 0:
        # Don't do anything
        return population

    mask = kwargs.get('mask', None)
    if mask is None:
        mask = da.ones(shape=population.shape, dtype='float32',
                       chunks=tuple(c[0] if c else 0 for c in population.chunks))

    # Normalize the mask
    mask_min = da.min(mask)
    _range = da.max(mask) - mask_min
    mask = da.where(_range > 0, (mask - mask_min) / _range, 1.)

    # Calculate the kernel indices and shape
    kernel = calculate_kernel(distance, csx, csy)
    if kernel is None:
        return population
    kernel, m, n = kernel
    # Dask does not like numpy types in depth
    m = int(m)
    n = int(m)
    depth = {0: m, 1: n, 2: 0}  # Padding
    boundary = {0: 0, 1: 0, 2:0}  # Constant padding value

    # Dstack and ghost the input arrays for the jitted function
    a = da.ghost.ghost(
        da.dstack([population, total_population, carrying_capacity, mask]),
        depth, boundary
    )
    chunks = tuple(c[0] if c else 0 for c in a.chunks)[:2]
    a = a.rechunk((chunks + (4,)))  # Need all of the last dimension passed at once

    # Perform the dispersal
    # args: population, total_population, carrying_capacity, kernel
    output = a.map_blocks(density_flux_task, kernel, m, n, dtype='float32',
                          chunks=(chunks + (1,)))
    # TODO: Fix trim internal, as overlapping blocks will not correctly propagate populations
    return da.ghost.trim_internal(output, {0: m, 1: n, 2: 0}).squeeze().astype('float32')


def masked_density_flux(population, total_population, carrying_capacity, distance, csx, csy, **kwargs):
    """wraps density flux, adds a mask"""
    # Check that there is a mask
    if kwargs.get('mask', None) is None:
        raise DispersalError('Masked Density Flux requires a mask, which is not available')
    return density_flux(population, total_population, carrying_capacity, distance, csx, csy, **kwargs)


@jit(nopython=True, nogil=True)
def density_flux_task(a, kernel, i_pad, j_pad):
    """
    Reallocate mass based on density gradients over the kernel
    :param a: 3d array with the following expected dimensionality
        last axis 0: mass to be redistributed (a subset of total mass)
        last axis 1: total mass
        last axis 2: total capacity (calculations are masked where capacity is 0)
        last axis 3: normalized mask
    :param kernel: Kernel index offset in the shape (m, n)
    :param i_pad: padding in the y-direction
    :param j_pad: padding in the x-direction
    :return: ndarray of redistributed mass
    """
    m, n, _ = a.shape
    k = kernel.shape[0]
    out = np.zeros((m, n, 1), np.float32)

    for i in range(i_pad, m - i_pad):
        for j in range(j_pad, n - j_pad):
            # Carry over mass to output
            out[i, j, 0] += a[i, j, 0]

            if a[i, j, 2] == 0 or a[i, j, 0] == 0:
                continue

            # Calculate a mean density
            _mean = 0.
            modals = 0.
            for k_i in range(k):
                if a[i + kernel[k_i, 0], j + kernel[k_i, 1], 2] != 0:
                    _mean += a[i + kernel[k_i, 0], j + kernel[k_i, 1], 1] / \
                             a[i + kernel[k_i, 0], j + kernel[k_i, 1], 2]
                    modals += 1.
            _mean /= modals

            # Evaluate gradient and skip if it is negative
            grad = a[i, j, 1] / a[i, j, 2] - _mean
            if grad <= 0:
                continue

            loss = (a[i, j, 0] / a[i, j, 1]) * (a[i, j, 2] * min(1., grad))

            # Find candidate locations based on their gradient
            _sum = 0.
            values = []
            locations = []
            for k_i in range(k):
                if a[i + kernel[k_i, 0], j + kernel[k_i, 1], 2] != 0:
                    grad = (_mean - a[i + kernel[k_i, 0], j + kernel[k_i, 1], 1] /
                            a[i + kernel[k_i, 0], j + kernel[k_i, 1], 2])
                    if grad > 0:
                        locations.append(k_i)
                        if a[i + kernel[k_i, 0], j + kernel[k_i, 1], 1] == 0:
                            destination_proportion = 1.
                        else:
                            destination_proportion = (a[i + kernel[k_i, 0], j + kernel[k_i, 1], 0] /
                                                      a[i + kernel[k_i, 0], j + kernel[k_i, 1], 1])
                        N = (destination_proportion *
                             a[i + kernel[k_i, 0], j + kernel[k_i, 1], 2] *
                             min(1., grad) *
                             min(1., a[i + kernel[k_i, 0], j + kernel[k_i, 1], 3]))  # mask
                        _sum += N
                        values.append(N)

            # Loss may not exceed candidates
            loss = min(loss, _sum)

            if len(locations) > 0 and _sum != 0:
                # Disperse the source mass to candidate locations linearly
                for l_i, k_i in enumerate(locations):
                    N = loss * (values[l_i] / _sum)
                    out[i + kernel[k_i, 0], j + kernel[k_i, 1], 0] += N
                    out[i, j, 0] -= N

    return out


"""
Distance Propagation, also known as maximum distance dispersal. Searches locations for the
minimum density at a specified distance and moves populations in attempt to flatten the gradient.
=======================================================================================================
"""

def distance_propagation(population, total_population, carrying_capacity, distance, csx, csy, **kwargs):
    # Check the inputs
    if any([not isinstance(a, da.Array) for a in [population, total_population, carrying_capacity]]):
        raise DispersalError('Inputs must be a dask arrays')

    if distance == 0:
        # Don't do anything
        return population

    # Calculate the kernel indices and shape
    kernel = calculate_kernel(distance, csx, csy)
    if kernel is None:
        return population
    kernel, m, n = kernel

    # Dask does not like numpy types in depth
    m = int(m)
    n = int(m)
    depth = {0: m, 1: n, 2: 0}  # Padding
    boundary = {0: 0, 1: 0, 2:0}  # Constant padding value

    # Dstack and ghost the input arrays for the jitted function
    a = da.ghost.ghost(
        da.dstack([population, total_population, carrying_capacity]),
        depth, boundary
    )
    chunks = tuple(c[0] if c else 0 for c in a.chunks)[:2]
    a = a.rechunk((chunks + (3,)))  # Need all of the last dimension passed at once

    # Perform the dispersal
    # args: population, total_population, carrying_capacity, kernel
    output = a.map_blocks(distance_propagation_task, kernel, m, n, dtype='float32',
                          chunks=(chunks + (1,)))
    # TODO: Fix trim internal, as overlapping blocks will not correctly propagate populations
    return da.ghost.trim_internal(output, {0: m, 1: n, 2: 0}).squeeze().astype('float32')


@jit(nopython=True, nogil=True)
def distance_propagation_task(a, kernel, i_pad, j_pad):
    """
    Reallocate mass to the best habitat at a specified distance
    :param a: 3d array with the following expected dimensionality
        last axis 0: mass to be redistributed (a subset of total mass)
        last axis 1: total mass
        last axis 2: total capacity (calculations are masked where capacity is 0)
        last axis 3: normalized mask
    :param kernel: Kernel index offset in the shape (m, n)
    :param i_pad: padding in the y-direction
    :param j_pad: padding in the x-direction
    :return: ndarray of redistributed mass
    """
    m, n, _ = a.shape
    k = kernel.shape[0]
    out = np.zeros((m, n, 1), np.float32)

    for i in range(i_pad, m - i_pad):
        for j in range(j_pad, n - j_pad):
            # Transfer mass to output
            out[i, j] += a[i, j, 0]

            # Skip if no mass or carrying capacity
            if a[i, j, 2] == 0 or a[i, j, 0] == 0:
                continue

            # Find the minimum density over the kernel
            _min = 0
            i_min, j_min = None, None
            eligible = []
            for k_i in range(k):
                if a[i + kernel[k_i, 0], j + kernel[k_i, 1], 2] != 0:
                    eligible.append(k_i)
                    d = a[i + kernel[k_i, 0], j + kernel[k_i, 1], 1] / \
                        a[i + kernel[k_i, 0], j + kernel[k_i, 1], 2]
                    if d <= _min:
                        _min = d
                        i_min, j_min = kernel[k_i, 0], kernel[k_i, 1]

            if _min == 0 and len(eligible) > 0:
                choice = eligible[np.random.randint(len(eligible))]
                i_min, j_min = kernel[choice, 0], kernel[choice, 1]

            # Calculate the gradient and propagate mass to the candidate
            if i_min is not None and j_min is not None:
                d_mean = ((a[i, j, 1] / a[i, j, 2]) + (a[i + i_min, j + j_min, 1] / a[i + i_min, j + j_min, 2])) / 2
                delta = d_mean - (a[i + i_min, j + j_min, 1] / a[i + i_min, j + j_min, 2])
                if delta > 0:
                    if a[i + i_min, j + j_min, 1] == 0:
                        destination_proportion = 1.
                    else:
                        destination_proportion = a[i + i_min, j + j_min, 0] / a[i + i_min, j + j_min, 1]
                    available = a[i, j, 1] / a[i, j, 2] - d_mean
                    if available > 0:
                        flux = min(
                            min(1., a[i, j, 0] / a[i, j, 1]) * (available * a[i, j, 2]),
                            (destination_proportion * (delta * a[i + i_min, j + j_min, 2]))
                        )

                        out[i + i_min, j + j_min] += flux
                        out[i, j] -= flux

    return out


def density_network(args):
    pass


def fixed_network(args):
    pass


def minimum_viable_population(population, min_pop, area, csx, csy, filter_std=3):
    """
    Eliminate clusters of low populations using minimum population and area thresholds
    :param population: Input population (dask array expected)
    :param min_pop: Minimum population of cluster
    :param area: Minimum cluster area
    :param filter_std: Standard deviation of gaussian filter to find clusters
    :return: Mask where population eliminated
    """
    # If the area is zero, just filter the population values directly
    if area == 0:
        return da.where(population < min_pop, 0, population)

    # Normalize population using gaussian kernel
    # ------------------------------------------
    # Calculate the padding using sigma (borrowed from scipy.ndimage.gaussian_filter1d)
    m = int(4. * filter_std + 0.5)

    depth = {0: m, 1: m}  # Padding
    boundary = {0: 'reflect', 1: 'reflect'}  # Constant padding value
    a = da.ghost.ghost(population, depth, boundary)

    regions = a.map_blocks(ndimage.gaussian_filter, filter_std, dtype=np.float32)
    regions = da.ghost.trim_internal(regions, depth)

    # Cluster populations and label using percentile based on n
    breakpoint = min_pop / (area / (csx * csy))
    breakpoint = (regions.min() + (
        (((breakpoint / population.mean()) + (breakpoint / population.max())) / 2) *
        (regions.max() - regions.min())))

    # Label the output and collect sums
    # TODO: This is incomplete from here onwards (but works)- the dask tree is computed to gather the labels
    # How does one ghost for a label operation??
    labels, num = ndimage.label(regions < breakpoint, np.ones(shape=(3, 3)))
    areas = ndimage.sum(np.ones(shape=labels.shape) * (csx * csy), labels, np.arange(num) + 1)
    pops = ndimage.sum(population, labels, np.arange(num) + 1)
    takeLabels = (np.arange(num) + 1)[(pops < min_pop) & (areas >= area)]
    indices = np.argsort(labels.ravel())
    bins = np.bincount(labels.ravel())
    indices = np.split(indices.ravel(), np.cumsum(bins[bins > 0][:-1]))
    indices = dict(zip(np.unique(labels.ravel()), indices))
    output = np.ones(shape=labels.ravel().shape, dtype='bool')
    for lab in takeLabels:
        output[indices[lab]] = 0

    return population * da.from_array(output.reshape(labels.shape), chunks=population.chunks)


def convolve(a, kernel):
    import numexpr as ne

    kernel = np.atleast_2d(kernel)

    if kernel.size == 1:
        return a * np.squeeze(kernel)

    # Create a padded array
    padding = (map(int, ((kernel.shape[0] - 1.) / 2, np.ceil((kernel.shape[0] - 1.) / 2))),
               map(int, ((kernel.shape[1] - 1.) / 2, np.ceil((kernel.shape[1] - 1.) / 2))))

    if 'float' not in a.dtype.name:
        output = a.astype('float32')
    else:
        output = a

    a_padded = np.pad(output, padding, 'constant')

    output.fill(0)

    # Perform convolution
    views = get_window_views(a_padded, kernel.shape)  # Views into a over the kernel
    local_dict = window_local_dict(views)  # Views turned into a pointer dictionary for numexpr
    # ne.evaluate only allows 32 arrays in one expression.  Need to chunk it up.
    keys = ['a{}_{}'.format(i, j) for i in range(len(views)) for j in range(len(views[0]))]  # preserve order
    keychunks = range(0, len(local_dict) + 31, 31)
    keychunks = zip(keychunks[:-1],
                    keychunks[1:-1] + [len(keys)])
    kernel = kernel.ravel()
    for ch in keychunks:
        new_local = {k: local_dict[k] for k in keys[ch[0]: ch[1]]}
        expression = '+'.join(['{}*{}'.format(prod_1, prod_2)
                               for prod_1, prod_2 in zip(new_local.keys(), kernel[ch[0]: ch[1]])])
        output += ne.evaluate(expression, local_dict=new_local)

    return output


def get_window_views(a, size):
    i_offset = (size[0] - 1) * -1
    j_offset = (size[1] - 1) * -1
    output = []
    for i in range(i_offset, 1):
        output.append([])
        _i = abs(i_offset) + i
        if i == 0:
            i = None
        for j in range(j_offset, 1):
            _j = abs(j_offset) + j
            if j == 0:
                j = None
            output[-1].append(a[_i:i, _j:j])
    return output


def window_local_dict(views, prefix='a'):
    return {'%s%s_%s' % (prefix, i, j): views[i][j]
            for i in range(len(views))
            for j in range(len(views[i]))}


METHODS = {'density-based dispersion': density_flux,
           'distance propagation': distance_propagation,
           'masked density-based dispersion': masked_density_flux}
