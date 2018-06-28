"""
Relocate mass using a variety of dispersal algorithms

Devin Cairns, 2018
"""

import numpy as np
import dask.array as da
from dask import delayed
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


"""
Density flux, also known as inter-habitat dispersal. Calculates a mean density over a neighbourhood and
reallocates populations within the neighbourhood in attempt to flatten the gradient.
=======================================================================================================
"""

def density_flux(population, total_population, carrying_capacity, distance, csx, csy, **kwargs):
    """
    'density-based dispersion'

    Dispersal is calculated using the following sequence of methods:

    Portions of populations at each element (node, or grid cell) in the study area array (raster) are moved to
    surrounding elements (a neighbourhood) within a radius that is defined by the input distance (:math:`d`), as
    presented in the conceptual figure below.

        .. image:: images/density_flux_neighbourhood.png
            :align: center

    .. attention:: No dispersal will occur if the provided distance is less than the distance between elements (grid cells) in the model domain, as none will be included in the neighbourhood

    The mean density (:math:`\\rho`) of all elements in the neighbourhood is calculated as:

    .. math::
       \\rho=\\frac{\\sum_{i=1}^{n} \\frac{pop_T(i)}{k_T(i)}}{n}

    where,

    :math:`pop_T` is the total population (of the entire species) at each element (:math:`i`); and\n
    :math:`k_T` is the total carrying capacity for the species

    The density gradient at each element (:math:`\\Delta`) with respect to the mean is calculated as:

    .. math::
        \\Delta(i)=\\frac{pop_T(i)}{k_T(i)}-\\rho

    If the centroid element is above the mean :math:`[\\Delta(i_0) > 0]`, it is able to release a portion of its
    population to elements in the neighbourhood. The eligible population to be received by surrounding elements is equal
    to the sum of populations at elements with negative density gradients, the :math:`candidates`:

    .. math::
        candidates=\\sum_{i=1}^{n} \\Delta(i)[\\Delta(i) < 0]k_T(i)

    The minimum of either the population above the mean at the centroid element - :math:`source=\\Delta(i_0)*k_T(i_0)`,
    or the :math:`candidates` are used to determine the total population that is dispersed from the centroid element to
    the other elements in the neighbourhood:

    .. math::
        dispersal=min\{source, candidates\}

    The population at the centroid element becomes:

    .. math::
        pop_a(i_0)=pop_a(i_0)-\\frac{pop_a(i_0)}{pop_T(i_0)}dispersal

    where,

    :math:`pop_a` is the age (stage) group population, which is a sub-population of the total.

    The populations of the candidate elements in the neighbourhood become (a net gain due to negative gradients):

    .. math::
        pop_a(i)=pop_a(i)-\\frac{\\Delta(i)[\\Delta(i) < 0]k_T(i)}{candidates}dispersal\\frac{pop_a(i)}{pop_T(i)}

    :param da.Array population: Sub-population to redistribute (subset of the ``total_population``)
    :param da.Array total_population: Total population
    :param da.Array carrying_capacity: Total Carrying Capacity (k)
    :param float distance: Maximum dispersal distance
    :param float csx: Cell size of the domain in the x-direction
    :param float csy: Cell size of the domain in the y-direction

    .. Attention:: Ensure the cell sizes are in the same units as the specified direction

    :Keyword Arguments:
        **mask** (*array*) --
            A weighting mask that scales dispersal based on the normalized mask value (default: None)
    :return: Redistributed population
    """
    if any([not isinstance(a, da.Array) for a in [population, total_population, carrying_capacity]]):
        raise DispersalError('Inputs must be a dask arrays')

    if distance == 0:
        # Don't do anything
        return population

    chunks = tuple(c[0] if c else 0 for c in population.chunks)[:2]

    mask = kwargs.get('mask', None)
    if mask is None:
        mask = da.ones(shape=population.shape, dtype='float32', chunks=chunks)

    # Normalize the mask
    mask_min = da.min(mask)
    _range = da.max(mask) - mask_min
    mask = da.where(_range > 0, (mask - mask_min) / _range, 1.)

    # Calculate the kernel indices and shape
    kernel = calculate_kernel(distance, csx, csy)
    if kernel is None:
        # Not enough distance to cover a grid cell
        return population
    kernel, m, n = kernel
    m = int(m)
    n = int(n)

    a = da.pad(da.dstack([population, total_population, carrying_capacity, mask]), ((m, m), (n, n), (0, 0)),
               'constant', constant_values=0)
    _m = -m
    if m == 0:
        _m = None
    _n = -n
    if n == 0:
        _n = None
    output = delayed(density_flux_task)(a, kernel, m, n)[m:_m, n:_n, 0]
    output = da.from_delayed(output, population.shape, np.float32)

    return output.rechunk(chunks)


def masked_density_flux(population, total_population, carrying_capacity, distance, csx, csy, **kwargs):
    """
    'masked density-based dispersion'

    See :func:`density_flux`. The dispersal logic is identical to that of ``density_flux``, however a mask is specified
    as a keyword argument to scale the dispersal. The :math:`mask` elements :math:`i` are first normalized to ensure
    values are not less than 0 and do not exceed 1:

    .. math::
        mask(i)=\\frac{mask(i)-min\{mask\}}{max\{mask\}-min\{mask\}}

    When the :math:`candidates` are calculated (as outlined in :func:`density_flux`) they are first scaled by the mask value:

    .. math::
        candidates=\\sum_{i=1}^{n} \\Delta(i)[\\Delta(i) < 0]k_T(i)mask(i)

    and are scaled by the mask when transferring populations from the centroid element:

    .. math::
        pop_a(i)=pop_a(i)-\\frac{\\Delta(i)[\\Delta(i) < 0]k_T(i)mask(i)}{candidates}dispersal\\frac{pop_a(i)}{pop_T(i)}

    :param array population: Sub-population to redistribute (subset of the ``total_population``)
    :param array total_population: Total population
    :param array carrying_capacity: Total Carrying Capacity (k)
    :param float distance: Maximum dispersal distance
    :param float csx: Cell size of the domain in the x-direction
    :param float csy: Cell size of the domain in the y-direction

    .. Attention:: Ensure the cell sizes are in the same units as the specified direction

    :Keyword Arguments:
        **mask** (*array*) --
            A weighting mask that scales dispersal based on the normalized mask value (default: None)
    :return: Redistributed population
    """
    # Check that there is a mask
    if kwargs.get('mask', None) is None:
        raise DispersalError('Masked Density Flux requires a mask, which is not available')
    return density_flux(population, total_population, carrying_capacity, distance, csx, csy, **kwargs)


@jit(nopython=True, nogil=True, cache=True)
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
    """
    'distance propagation'

    Distance propagation is used to redistribute populations to distal locations based on density gradients. Portions of
    populations at each element (node, or grid cell) in the study area array (raster) are moved to a target element
    at a radius that is defined by the input distance (:math:`d`), as presented in the conceptual
    figure below.

    .. image:: images/distance_propagation_neighbourhood.png
        :align: center

    .. attention:: No dispersal will occur if the provided distance is less than the distance between elements (grid cells) in the model domain, as none will be included in the neighbourhood

    The density (:math:`\\rho`) of all distal elements (:math:`i`) is calculated as:

    .. math::
       \\rho(i)=\\frac{pop_T(i)}{k_T(i)}

    where,

    :math:`pop_T` is the total population (of the entire species) at each element (:math:`i`); and\n
    :math:`k_T` is the total carrying capacity for the species

    The distal element with the minimum density is chosen as a candidate for population dispersal from the centroid
    element. If the density of distal elements is homogeneous, one element is picked at random. The density gradient
    :math:`\\Delta` is then calculated using the centroid element :math:`i_0` and the chosen distal element :math:`i_1`:

    .. math::
        \\rho=\\frac{pop_T(i_0)/k_T(i_0)+pop_T(i_1)/k_T(i_1)}{2}

    .. math::
        \\Delta(i)=\\frac{pop_T(i)}{k_T(i)}-\\rho

    If the centroid element is above the mean :math:`[\\Delta(i_0) > 0]`, and the distal element is below the mean
    :math:`[\\Delta(i_1) < 0]`, dispersal may take place. The total population dispersed is calculated by taking the
    minimum of the population constrained by the gradient:

    .. math::
        dispersal=min\{|\\Delta(i_0)k_T(i_0)|, |\\Delta(i_1)k_T(i_1)|\}

    The population at the centroid element becomes:

    .. math::
        pop_a(i_0)=pop_a(i_0)-dispersal

    where,

    :math:`pop_a` is the age (stage) group population, which is a sub-population of the total.

    The population at the distal element becomes (a net gain due to a negative gradient):

    .. math::
        pop_a(i_1)=pop_a(i_1)-dispersal

    :param da.Array population: Sub-population to redistribute (subset of the ``total_population``)
    :param da.Array total_population: Total population
    :param da.Array carrying_capacity: Total Carrying Capacity (n)
    :param float distance: Maximum dispersal distance
    :param float csx: Cell size of the domain in the x-direction
    :param float csy: Cell size of the domain in the y-direction

    .. Attention:: Ensure the cell sizes are in the same units as the specified direction

    :return: Redistributed population
    """
    # Check the inputs
    if any([not isinstance(a, da.Array) for a in [population, total_population, carrying_capacity]]):
        raise DispersalError('Inputs must be a dask arrays')

    if distance == 0:
        # Don't do anything
        return population

    chunks = tuple(c[0] if c else 0 for c in population.chunks)[:2]

    # Calculate the kernel indices and shape
    kernel = calculate_kernel(distance, csx, csy, True)
    if kernel is None:
        return population
    kernel, m, n = kernel
    m = int(m)
    n = int(n)

    # Dask does not like numpy types in depth
    a = da.pad(da.dstack([population, total_population, carrying_capacity]), ((m, m), (n, n), (0, 0)),
               'constant', constant_values=0)

    # Perform the dispersal
    # args: population, total_population, carrying_capacity, kernel
    _m = -m
    if m == 0:
        _m = None
    _n = -n
    if n == 0:
        _n = None
    output = delayed(distance_propagation_task)(a, kernel, m, n)[m:_m, n:_n, 0]
    output = da.from_delayed(output, population.shape, np.float32)

    return output.rechunk(chunks)


@jit(nopython=True, nogil=True, cache=True)
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
    """
    'density network dispersal'

    .. note:: In Development

    Compute dispersal based on a density gradient along a least cost path network analysis using a cost surface.

    :raises: NotImplementedError
    :param args:
    """
    raise NotImplementedError('Not implemented yet')

def fixed_network(args):
    """
    'fixed network movement'

    .. note:: In Development

    Compute dispersal based on a least cost path network analysis using a cost surface.

    :raises: NotImplementedError
    :param args:
    """
    raise NotImplementedError('Not implemented yet')


def minimum_viable_population(population, min_pop, area, csx, csy, domain_area, filter_std=5):
    """
    Eliminate clusters of low populations using a minimum population and area thresholds

    The spatial distribution of populations are assessed using a gaussian filter over a neighourhood of elements that is
    calculated using the ``filter_std`` (standard deviation) argument:

    .. math::
        k=(4\\sigma)+0.5

    where :math:`k` is the neighbourhood size (in elements), and :math:`\\sigma` is the standard deviation.

    A threshold :math:`T` is then calculated to be used as a contour to constrain regions:

    Calculate an areal density per element, :math:`\\rho_a`:

    .. math::
        \\rho_a=\\frac{p}{(A/(dx\\cdot dy)}

    where :math:`p` is the population, :math:`A` is the minimum area, and :math:`dx` and :math:`dy` are the spatial
    gradients in the x and y direction, respectively.

    Calculate the threshold within the filtered regions by normalizing the region range with the population range

    .. math::
        T=min\{k\}+\\bigg[\\frac{((\\rho_a/p_m)+(\\rho_a/max\{p\})}{2}\\cdot (max\{k\}-min\{k\})\\bigg]

    Populations in the study area within the threshold contour are removed and applied to mortality as a result of the
    minimum viable population.

    :param dask.Array population: Input population
    :param float min_pop: Minimum population of cluster
    :param float area: Minimum cluster area
    :param int filter_std: Standard deviation of gaussian filter to find clusters
    :return: A coefficient for populations. I.e. 1 (No Extinction) or 0 (Extinction)
    """
    # If the area is zero, just filter the population values directly
    if area == 0:
        return ~(population < min_pop)

    chunks = tuple(c[0] if c else 0 for c in population.chunks)[:2]

    @delayed
    def _label(population):
        # If the region is close to the study area size, avoid region delineation
        # ---------------------------------------------
        if area > domain_area * .9:
            p = min(1., population.sum() / min_pop)
            ext = np.random.choice([0, 1], p=[1 - p, p])
            return np.full(population.shape, ext, np.bool)

        # Normalize population using gaussian kernel
        # ------------------------------------------
        regions = ndimage.gaussian_filter(population, filter_std)

        # Create a breakpoint at one standard deviation below the mean to create regions
        breakpoint = regions.mean() - np.std(regions)

        # Label the output and collect sums
        # ---------------------------------
        loc = regions < breakpoint
        labels, num = ndimage.label(loc, np.ones(shape=(3, 3)))
        areas = ndimage.sum(np.ones(shape=labels.shape) * (csx * csy), labels, np.arange(num) + 1)
        pops = ndimage.sum(population, labels, np.arange(num) + 1)
        takeLabels = (np.arange(num) + 1)[(pops < min_pop) & (areas >= area)]
        indices = np.argsort(labels.ravel())
        bins = np.bincount(labels.ravel())
        indices = np.split(indices.ravel(), np.cumsum(bins[bins > 0][:-1]))
        indices = dict(zip(np.unique(labels.ravel()), indices))
        output = np.ones(shape=labels.ravel().shape, dtype='bool')
        for lab in takeLabels:
            # The probability of region-based extinction is scaled using the population
            p = min(1, pops[lab - 1] / min_pop)
            ext = np.random.choice([0, 1], p=[1 - p, p])
            output[indices[lab]] = ext
        return output.reshape(population.shape)

    # Note - this is delayed and not chunked. The entire arrays will be loaded into memory upon execution
    output = da.from_delayed(_label(population), population.shape, np.bool)

    return output.rechunk(chunks)


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
           'masked density-based dispersion': masked_density_flux,
           'density network dispersal': density_network,
           'fixed network movement': fixed_network}
