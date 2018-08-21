.. _domain:

============
Model Domain
============

.. contents:: Table of contents
    :depth: 5
    :local:

.. py:currentmodule:: popdyn

A popdyn model :ref:`domain` provides a framework for simulations so they may be spatially and temporally discretized,
and parameterized. The primary purposes of a :ref:`domain` are to:

* Manage a file to store study area data in HDF5 format, and act as a quasi-database for the model domain attributes.
  The file will be automatically created with the extension ``.popdyn``, and can be used to instantiate :class:`Domain`
  instances.
* The :ref:`domain` has a consistent spatial extent and anisotropic grid resolution
* Data added to a domain are automatically read, spatially transformed, converted, and stored on the disk.
* Data are commonly added with :ref:`species` or :ref:`parameters` instances to add them and corresponding data to the
  :ref:`domain`
* Domain instances are provided to selected :ref:`solvers` to calculate derived parameters and simulate populations in
  the model
* Computation "chunks" are also specified in the domain to make use of distributed schedulers that allow parallel
  computation and make memory (RAM) usage predictable

Species (and accompanying data) are added to a model domain using parameters or population data using one of the
following methods:

- :meth:`~Domain.add_population`
- :meth:`~Domain.add_mortality`
- :meth:`~Domain.add_carrying_capacity`
- :meth:`~Domain.add_fecundity`
- :meth:`~Domain.add_mask`

.. Attention:: All parameters or data that are added to a model domain must be in the same spatial and temporal units

Data associated with :ref:`species` or :ref:`parameters` may be added to the model domain using one of the following
formats:

* A raster dataset. If the raster spatial reference and geotransform do not match the domain, it will be automatically
  resampled using nearest neighbour interpolation when added to the domain.
* A `numpy <https://docs.scipy.org/doc/numpy/user/basics.creation.html>`_ array-like object with a shape matching the
  model domain, or in a shape that can be
  `broadcasted <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ to the shape of the model domain.
* A scalar that is repeated over all domain elements.

Data added to a model domain may also be distributed using any of the following ``keyword arguments``:

**distribute**: If the input data are scalars, the value will be divided evenly among all domain elements:

.. math::
    param/n

where :math:`param` is the value, and :math:`n` is the number of domain elements).

If data are arrays, the sum will
be evenly divided among all domain elements:

.. math::
    \frac{\sum_{i=1}^{n}param(i)}{n}

**distribute_by_habitat**: Use habitat quality to linearly distribute the input parameter data. Data that are
scalars or arrays are aggregated using the same method as ``distribute``. The :math:`param` value is distributed over
habitat (:math:`k`) using the following:

.. math::
    param\cdot \frac{k(i)[k(i)\neq 0]}{\sum_{i=1}^{n}k(i)[k(i)\neq 0]}

Once added to a domain, species with the same name automatically become stratified by their name, sex, and age groups,
and are able to inherit mortality and fecundity based on a species - sex - age group hierarchy.

.. note:: Remember, for inheritance to work, the members of the same species added to the model domain must have matching names

Time is also discretized in the model domain. Data related to species and parameters are tied to a specific time, which
is used in the solvers. Time is always represented as whole numbers, although these numbers are not constrained to any
interval or time format. For example, a model domain that is solved annually will contain data that is tied to a specific
year. A model domain that is solved every second will have data added at a specific second over the duration of the model.
The start and end times are specified in the solvers.

.. autoclass:: Domain
    :members:

    .. method:: add_population(species, data, time, **kwargs)

        Add population data for a given species at a specific time slice in the domain.

        :param Species species: A Species instance
        :param data: Population data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)
        :param time: The time slice to insert the population data into

        :Keyword Arguments:
            **distribute_by_habitat** (*bool*) --
                Divide the sum of the input population linearly among domain elements using the covariate
                carrying capacity (Default: False)
            **discrete_age** (*int*) --
                Apply the input population to a single discrete age in the age group (Default: None)
            **distribute** (*bool*) --
                Divide the sum of the input data evenly among all domain nodes (grid cells) (default: True)
            **overwrite** (*str*) --
                Overwrite method for replacing existing data. Use one of ``['replace', 'add']`` (Default 'replace')

    .. method:: add_mortality(species, mortality, time, data=None, **kwargs)

        Mortality is added to the domain with species and Mortality objects in tandem.

        Multiple mortality datasets may added to a single species,
        as they are stacked in the domain for each species (or sex/age group).

        :param Species species: Species instance
        :param Morality mortality: Mortality instance.
        :param data: Mortality data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)
        :param time: The time slice to insert mortality

        :Keyword Arguments:
            **distribute** (*bool*) --
                Divide the sum of the input data evenly among all domain nodes (grid cells) (default: True)
            **overwrite** (*str*) --
                Overwrite method for replacing existing data. Use one of ``['replace', 'add']`` (Default 'replace')

    .. method:: add_carrying_capacity(species, carrying_capacity, time, data=None, **kwargs)

        Carrying capacity is added to the domain with species and CarryingCapacity objects in tandem.

        Multiple carrying capacity datasets may added to a single species, as they are stacked in the domain for each species (or sex/age group).

        :param Species species: Species instance
        :param CarryingCapacity carrying_capacity: Carrying capacity instance.
        :param time: The time slice to insert the carrying capacity
        :param data: Carrying Capacity data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)

        :Keyword Arguments:
            **is_density** (*bool*) --
                The input data are a density, and not an absolute population (Default: False)
            **distribute** (*bool*) --
                Divide the sum of the input data evenly among all domain nodes (grid cells) (default: True)
            **overwrite** (*str*) --
                Overwrite method for replacing existing data. Use one of ``['replace', 'add']`` (Default 'replace')

    .. method:: add_fecundity(species, fecundity, time, data=None, **kwargs)

        Fecundity is added to the domain with species and Fecundity objects in tandem.

        Multiple fecundity datasets may added to a single species, as they are stacked in the domain for each
        species (or sex/age group).

        :param Species species: Species instance
        :param Fecundity fecundity: Fecundity instance.
        :param data: Fecundity data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)
        :param time: The time slice to insert fecundity

        :Keyword Arguments:
            **distribute** (*bool*) --
                Divide the sum of the input data evenly among all domain nodes (grid cells) (default: True)
            **overwrite** (*str*) --
                Overwrite method for replacing existing data. Use one of ``['replace', 'add']`` (Default 'replace')

    .. method:: add_mask(species, time, data=None, **kwargs)

        A mask is a general-use dataset associated with a species. It is currently only used for masked density-based
        dispersal. Only one mask may exist for a species - sex - age group.

        :param Species species: Species instance
        :param data: Mask data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)
        :param time: The time slice to insert fecundity

        :Keyword Arguments:
            **distribute** (*bool*) --
                Divide the sum of the input data evenly among all domain nodes (grid cells) (default: True)
            **overwrite** (*str*) --
                Overwrite method for replacing existing data. Use one of ``['replace', 'add']`` (Default 'replace')
