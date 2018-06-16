.. _dispersal:

=========
Dispersal
=========

.. contents:: Table of contents
    :depth: 2
    :local:

.. py:currentmodule:: dispersal

Species may move independently in a model domain using any number of available dispersal methods. They are added to a
species using the :ref:`species` :func:`add_dispersal` method, which is specified using one of the available methods listed in
``popdyn.dispersal.METHODS``. These methods, and their keys are used to add each to a species are listed below:

.. autosummary::
    density_flux
    distance_propagation
    masked_density_flux
    density_network
    fixed_network

Dispersal methods are described as follows:

.. autofunction:: density_flux
.. autofunction:: distance_propagation
.. autofunction:: masked_density_flux
.. autofunction:: density_network
.. autofunction:: fixed_network

