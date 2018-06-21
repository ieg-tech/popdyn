.. Population Dynamics Model documentation master file, created by
   sphinx-quickstart on Wed Jun 13 13:14:42 2018

.. _popdyn:

======
Popdyn
======

*Population Dynamics Modelling toolset*

.. image:: images/popdyn.png
    :align: center

Popdyn allows multiple populations of species to **reproduce**, **age**, **die**, **move**, and **interact** with one
another. This library strives to be pythonic, and easily abstracted through simple syntax to enable users to focus on
their own science. Models use initial populations and parameter sets to interpolate or predict populations at other
times and locations. Popdyn models utilize and combine theory related to:

- Biological systems
- Land cover and land use
- Geographical information systems (GIS)
- Deterministic modelling
- Stochastic modelling
- Cellular automata
- Agent-based modelling

The popdyn module is comprised of four main components:

-  **Species (Populations)** - which possess individual traits and optionally, sex (gender)/age (stage) attributes
-  **Parameters** - which affect species through various means
-  **Domain** - which defines a study area through spatial discretization
-  **Solvers** - which provides modular numerical solvers to compute systems of species and parameters within a domain

A typical popdyn model workflow includes:

#. Create one or more species, specifying their attributes through the provided :ref:`species` abstraction. Species are
   templates that are created prior to adding them to a model domain, where their population magnitude and distribution
   are specified independently.

#. Create carrying capacity (i.e. habitat), mortality, and fecundity :ref:`parameters` independently. Similar to :ref:`species`,
   Parameters are added to the model domain with spatially-distributed data that match the spatial and temporal
   specifications of the model :ref:`domain`.

#. Attach population data and parameter data to a species by adding them to a :ref:`domain` at specified time slices.

#. Solve the populations of all added species in a domain by choosing any of the available :ref:`solvers`, and the time duration.


.. toctree::
    :hidden:

    species
    dispersal
    parameters
    domain
    solvers
    data_summary
