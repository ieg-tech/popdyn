.. _domain:

============
Model Domain
============

.. contents:: Table of contents
    :depth: 4
    :local:

.. py:currentmodule:: popdyn

A popdyn model :ref:`domain` provides a framework for simulations so they may be spatially and temporally discretized,
and parameterized. The primary purposes of a :ref:`domain` are to:

* Manage a file to store study area data in HDF5 format, and act as a quasi-database for the model domain attributes. The file will be automatically created with the extension ``.popdyn``, and can be used to instantiate :class:`Domain` instances.
* The :ref:`domain` has a consistent spatial extent and anisotropic grid resolution
* Data added to a domain are automatically read, spatially transformed, converted, and stored on the disk.
* Data are commonly added with :ref:`species` or :ref:`parameters` instances to add them and corresponding data to the :ref:`domain`
* Domain instances are provided to selected :ref:`solvers` to calculate derived parameters and simulate populations in the model

Species are added to a model domain using parameters or population data using one of the following methods:

- :meth:`~Domain.add_population`
- :meth:`~Domain.add_mortality`
- :meth:`~Domain.add_carrying_capacity`
- :meth:`~Domain.add_fecundity`

.. Attention:: All parameters or data that are added to a model domain must be in the same spatial and temporal units

Once added to a domain, species with the same name automatically become stratified by their name, sex, and age groups,
and are able to inherit mortality and fecundity based on a species - sex - age group hierarchy.

.. autoclass:: Domain
    :members:
