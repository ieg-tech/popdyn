.. _parameters:

==========
Parameters
==========

.. contents:: Table of Contents
   :depth: 3

.. py:currentmodule:: popdyn

Sets of parameters may be customized and prepared prior to adding them to a :ref:`species` or a :ref:`domain`. They
may be created to be used with multiple :ref:`species` or :ref:`domain` objects, and are usually accompanied by a gridded
(or *raster*) dataset when added to the :ref:`domain`. Inter-species relationships are also defined using :ref:`parameters`,
whereby tables of correlations between population density and parameter values or coefficients are provided.

Paramters consist of:

.. autosummary::
    CarryingCapacity
    Mortality
    Fecundity

Carrying Capacity
-----------------

Carrying Capacity is used in popdyn models to specify the population potential on a landscape, expressed as a total
population. Carrying Capacity may also be modified using the density of another species, which may be invoked using the
:func:`add_as_species` method. Multiple Carrying Capacity instances may be added to a given :ref:`species` in a model
:ref:`domain`.

.. autoclass:: CarryingCapacity
    :inherited-members:
    :members:

Mortality
---------



.. autoclass:: Mortality
    :inherited-members:
    :members:

Fecundity
---------

.. autoclass:: Fecundity
    :inherited-members:
    :members:
