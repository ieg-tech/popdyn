.. Population Dynamics Model documentation master file, created by
   sphinx-quickstart on Wed Jun 13 13:14:42 2018

.. _popdyn:

======
Popdyn
======

.. image:: images/popdyn.png
    :align: center

*Population Dynamics Modelling toolset*

Popdyn allows multiple species to **reproduce**, **die**, **move** around the landscape, and **interact** with one
another. This library strives to be pythonic, and easily abstracted through simple syntax to enable users to focus on
their own science.

The popdyn module is comprised of four main components:

-  **Species** - which possess individual traits
-  **Parameters** - which affect species through various means
-  **Domain** - which defines a study area through spatial discretization
-  **Solvers** - which provides modular numerical solvers to compute systems of species and parameters within a domain

A typical popdyn model workflow includes:


1. Create one or more species, specifying their behaviour through the provided :ref:`species` abstraction.
2. Create carrying capacity (i.e. habitat), mortality, and fecundity :ref:`parameters` independently.
3. Attach population data and parameters to a species by adding them to a :ref:`domain`.
4. Solve the populations of all added species in a domain by choosing any of the available :ref:`solvers`, and the time duration.

.. toctree::
    :hidden:

    species
    dispersal
    parameters
    domain
    solvers
    data_summary
