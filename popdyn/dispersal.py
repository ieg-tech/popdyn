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


def apply(a, method, args):
    """Apply a dispersal method on the input dask array"""
    if not isinstance(a, da.Array):
        raise DispersalError('The input array must be a dask array')

    if method not in METHODS:
        raise DispersalError('The input method "{}" is mot supported'.format(method))

    return METHODS[method](a, *args)


def calculate_kernel(distance, csx, csy):
    """Calculate kernel index offsets using the distance and cell size"""
    # Calculate the kernel matrix size
    m = np.uint64(np.round(distance / csy))
    n = np.uint64(np.round(distance / csx))

    # Use a distance transform to select the active grid locations
    kernel = np.ones(shape=(int(m * 2), int(n * 2)), dtype='bool')
    kernel[m, n] = 0
    kernel = ndimage.distance_transform_edt(kernel, (csy, csx)) <= distance

    # Create an offset matrix
    i, j = np.asarray(np.where(kernel))
    i -= m
    j -= n

    return np.asarray([i, j]).T, m, n


def density_flux(population, total_population, carrying_capacity, distance, csx, csy):
    # Check the inputs
    if any([not isinstance(a, da.Array) for a in [population, total_population, carrying_capacity]]):
        raise DispersalError('Inputs must be a dask arrays')

    # Calculate the kernel indices and shape
    kernel, m, n = calculate_kernel(distance, csx, csy)
    depth = {0: m, 1: n}  # Padding
    boundary = {0: 0, 1: 0}  # Constant padding value

    # Dstack and ghost the input arrays for the jitted function
    a = da.ghost.ghost(
        da.dstack([population, total_population, carrying_capacity]),
        depth, boundary
    )
    chunks = tuple(c[0] if c else 0 for c in a.chunks)
    a.rechunk((chunks[:2] + (3,)))  # Need all of the last dimension passed at once

    # Perform the dispersal
    # args: population, total_population, carrying_capacity, kernel
    output = a.map_blocks(density_flux_task, kernel, m, n)
    return da.ghost.trim_internal(output, {0: m, 1: n})


@jit(nopython=True, nogil=True)
def density_flux_task(a, kernel, i_pad, j_pad):
    """
    Reallocate mass based on density gradients over the kernel
    :param a: 3d array with the following expected dimensionality
        axis (2, 0): mass to be redistributed (a subset of total mass)
        axis (2, 1): total mass
        axis (2, 2): total capacity (calculations are masked where capacity is 0)
    :param kernel: Kernel index offset in the shape (m, n)
    :param i_pad: padding in the y-direction
    :param j_pad: padding in the x-direction
    :return: ndarray of redistributed mass
    """
    m, n, _ = a.shape
    k = kernel.shape[0]
    out = np.empty((m, n), np.float32)

    for i in range(i_pad, m - i_pad):
        for j in range(j_pad, n - j_pad):
            if a[i, j, 2] == 0:
                continue

            # Calculate a mean density
            _mean = modals = 0
            for k_i in range(k):
                if a[i + kernel[k_i, 0], j + kernel[k_i, 1], 2] != 0:
                    _mean += a[i + kernel[k_i, 0], j + kernel[k_i, 1], 1] / \
                             a[i + kernel[k_i, 0], j + kernel[k_i, 1], 2]
                    modals += 1
            _mean /= modals

            # Evaluate gradient and skip if it is negative
            loss = a[i, j, 0] * (a[i, j, 0] / (a[i, j, 1] * (a[i, j, 1] / a[i, j, 2] - _mean)))
            if loss <= 0:
                continue

            # Find candidate locations based on their gradient
            _sum = 0
            for k_i in range(k):
                if a[i + kernel[k_i, 0], j + kernel[k_i, 1], 2] != 0:
                    grad = (a[i + kernel[k_i, 0], j + kernel[k_i, 1], 1] / \
                            a[i + kernel[k_i, 0], j + kernel[k_i, 1], 2] - _mean)
                    if grad < 0:
                        grad

    return out


def distance_propagation(args):
    pass


def density_network(args):
    pass


def fixed_network(args):
    pass


def dispersive_flux(self, species, t, population, k, distance, file_instance,
                    relocate_distance=0, relocate_proportion=0.15):
    """
    Calculate migration based on a population and k, and update populations of all ages
    :param population: population array
    :param k: carrying capacity array
    :param distance: maximum possible distance travelled/time step
    :returns: Proportion of population change
    """
    # Calculate the window size based on distance
    i, j = (int(distance / self.csy) * 2) + 1, (int(distance / self.csx) * 2) + 1
    window = (i, j)

    # Create a fraction kernel and scale using distance
    dist = np.ones(shape=window, dtype='bool')
    dist[(window[0] - 1) / 2, (window[1] - 1) / 2] = 0
    dist = ndimage.distance_transform_edt(dist, (self.csy, self.csx))
    dm = dist <= distance  # dm is a mask of the distance kernel below the threshold
    dist[~dm] = 0

    # If relocation is desired, apply prior to dispersive flux calculation
    if relocate_distance > 0:
        if relocate_proportion > 1:
            relocate_proportion = 1.
        # Calculate the window size based on distance
        i, j = ((int(relocate_distance / self.csy) * 2) + 1,
                (int(relocate_distance / self.csx) * 2) + 1)
        reloc_window = (i, j)
        # Create an outer ring kernel using distance
        reloc_dist = np.ones(shape=reloc_window, dtype='bool')
        reloc_dist[(reloc_window[0] - 1) / 2, (reloc_window[1] - 1) / 2] = 0
        reloc_dist = ndimage.distance_transform_edt(reloc_dist, (self.csy, self.csx))
        reloc_dm = ((reloc_dist <= relocate_distance) &
                    (reloc_dist > relocate_distance - min(self.csx, self.csy)))
        # Calculate density
        m = k > 0
        dens = ne.evaluate('where(m,1-(population/k),0)')
        dens_m = dens == 1
        dens[dens_m] += k[dens_m]
        # Create outer ring iterator
        iter_i, iter_j = np.where(reloc_dm)
        iter_i -= (reloc_window[0] - 1) / 2
        iter_j -= (reloc_window[1] - 1) / 2
        p_i, p_j = np.where(population > 0)
        # Find best habitat using both density and k
        max_i, max_j = p_i.copy(), p_j.copy()
        max_values = np.zeros(shape=p_i.shape, dtype='float32')
        for i, j in zip(iter_i, iter_j):
            i, j = p_i + i, p_j + j
            i[i < 0] = 0
            i[i > population.shape[0] - 1] = population.shape[0] - 1
            j[j < 0] = 0
            j[j > population.shape[1] - 1] = population.shape[1] - 1
            cc_set = dens[i, j]
            cc_mask = cc_set > max_values
            max_i[cc_mask] = i[cc_mask]
            max_j[cc_mask] = j[cc_mask]
            max_values[cc_mask] = cc_set[cc_mask]
        # Duplicate target indices may exist.  Sort using max_ and add duplicates
        index_sort = np.argsort(np.rec.fromarrays([max_i, max_j]), order=('f0', 'f1'))
        p_i = p_i[index_sort]
        p_j = p_j[index_sort]
        max_i = max_i[index_sort]
        max_j = max_j[index_sort]
        # Find locations of duplicates
        duplicates = np.concatenate([[0], (np.diff(max_i) == 0) & (np.diff(max_j) == 0)])
        population_set = population[p_i, p_j] * relocate_proportion
        if duplicates.sum() != 0:
            dup_pop = population_set.copy()
            duplicates_exist = True
            # Label duplicates to prepare for bincount
            bins, _ = ndimage.label(duplicates.astype('uint32'), [1, 1, 1])
            # Add labels to first occurrence of duplicate value
            first_replace = bins[1:]
            first_m = first_replace != 0
            bins[:-1][first_m] = first_replace[first_m]
            # Recreate duplicates as index array
            bin_mask = bins > 0
            duplicates = bins[bin_mask] - 1
            # Cumulative population where duplicates exist
            pop_add = np.bincount(bins, dup_pop)[1:]
            # Replace values, as np only uses one index based on ordering
            dup_pop[bin_mask] = pop_add[duplicates]
        else:
            duplicates_exist = False
            dup_pop = population_set
        population[max_i, max_j] += dup_pop
        population[p_i, p_j] -= population_set

    # Calculate density and average density while dm is still a mask
    m = k > 0
    dens = ne.evaluate('where(m,population/k,0)')
    modals = ndimage.convolve(m.astype('uint32'), dm, mode='constant')
    m_dens = ndimage.convolve(dens, dm, mode='constant')
    m_dens = ne.evaluate('where(modals>0,m_dens/modals,0)')
    m_dens[m_dens > 1] = 1
    # Calculate population density potential
    flux = ne.evaluate('(m_dens-dens)*k')
    flux_mask = flux > 0

    # Resume with kernel
    dm[(window[0] - 1) / 2, (window[1] - 1) / 2] = 0
    if np.unique(dist[dm]).size > 1:
        _m, _b = np.linalg.solve([[dist.max(), 1.], [dist[dm].min(), 1.]], [1 / 3., 1.])
    else:
        _m, _b = 2 / 3., 0
    dist[dm] = (dist[dm] * _m) + _b
    dist[dm] /= dist[dm].sum()

    # Separate negative (losing) and positive (gaining) regions
    pos = flux.copy()
    pos[pos < 0] = 0
    neg = flux * -1
    neg[neg < 0] = 0

    # Redistribute mass, while avoiding recreating negative flux
    pos_m = (m & (flux > 0) & (neg == 0)).astype('float32')
    # Population supernova
    _pos = pos - ndimage.convolve(neg, dist, mode='constant')
    sub_alloc = _pos < 0  # Overallocated
    overalloc = np.abs(_pos[sub_alloc])
    pos_m[sub_alloc] = 1 - (overalloc / (pos[sub_alloc] + overalloc))  # Proportion of flux potential
    pos = _pos
    neg -= ndimage.convolve(pos_m, dist, mode='constant') * neg  # Calculate the total losses
    pos[sub_alloc] = 0

    # Change flux to an absolute change of population
    flux = ne.evaluate('flux-((neg*-1)+pos)')
    pop_mask = population > 0

    # Perform migration on each age
    dm[(window[0] - 1) / 2, (window[1] - 1) / 2] = 1
    for sex in ['male', 'female']:
        for key in file_instance['%s/%s/%s' % (species, t, sex)].keys():
            try:
                key = int(key)
            except ValueError:
                continue
            ds = file_instance['%s/%s/%s/%s' % (species, t, sex, key)][:].astype('float32')
            # If mass is relocated, it must be fractionated as well
            if relocate_distance > 0:
                # Apply relocation to age population
                ds_set = ds[p_i, p_j] * relocate_proportion
                if duplicates_exist:
                    ds_dup = ds_set.copy()
                    pop_add = np.bincount(bins, ds_set)[1:]
                    ds_dup[bin_mask] = pop_add[duplicates]
                    ds[max_i, max_j] += ds_dup
                else:
                    ds[max_i, max_j] += ds_set
                ds[p_i, p_j] -= ds_set
            prop = ne.evaluate('where(pop_mask,ds/population,0)')
            dil = ndimage.convolve(prop, dm, mode='constant')  # Expanded proportion in dispersion domain
            count = ndimage.convolve((prop > 0).astype('uint16'), dm, mode='constant')  # For mean of proportions
            dil = ne.evaluate('where(count>0,dil/count,0)')
            # Apply flux, given fraction of population
            ds = ne.evaluate('ds+(flux*dil)')

            ds[ds < 0] = 0  # Truncation as a result of math

            # Save dispersed population
            self._create_dataset('migrations/%s/%s' % (sex, key), ds, 'replace', file_instance)

def minimum_viable_population(self, population, min_pop, area, filter_std=3):
    """
    Eliminate clusters of low populations using minimum population and area thresholds
    :param population: Input population
    :param min_pop: Minimum population of cluster
    :param area: Minimum cluster area
    :param filter_std: Standard deviation of gaussian filter to find clusters
    :return: Mask where population eliminated
    """
    # Normalize population using gaussian kernel
    regions = ndimage.gaussian_filter(population, filter_std)
    # Cluster populations and label using percentile based on n
    breakpoint = min_pop / (area / (self.csx * self.csy))
    breakpoint = (regions.min() + (
        (((breakpoint / population.mean()) + (breakpoint / population.max())) / 2) *
        (regions.max() - regions.min())))
    labels, num = ndimage.label(regions < breakpoint, np.ones(shape=(3, 3)))
    areas = ndimage.sum(np.ones(shape=labels.shape) * (self.csx * self.csy), labels, np.arange(num) + 1)
    pops = ndimage.sum(population, labels, np.arange(num) + 1)
    takeLabels = (np.arange(num) + 1)[(pops < min_pop) & (areas >= area)]
    indices = np.argsort(labels.ravel())
    bins = np.bincount(labels.ravel())
    indices = np.split(indices.ravel(), np.cumsum(bins[bins > 0][:-1]))
    indices = dict(zip(np.unique(labels.ravel()), indices))
    output = np.ones(shape=labels.ravel().shape, dtype='bool')
    for lab in takeLabels:
        output[indices[lab]] = 0
    return output.reshape(labels.shape)


def convolve(a, kernel):
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
           'density-based network': density_network,
           'fixed network': fixed_network}
