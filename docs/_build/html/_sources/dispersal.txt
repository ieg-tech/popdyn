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
``popdyn.dispersal.METHODS``.

.. note:: Poplulation is never moved outside of the model domain or to where carrying capacity is 0

These methods, and their keys are used to add each to a species are listed below:

.. autosummary::
    density_flux
    distance_propagation
    masked_density_flux
    density_network
    fixed_network

Dispersal methods are described as follows:

Interhabitat Dispersal
-----------------------

.. autofunction:: density_flux

Max Dispersal
-------------

.. autofunction:: distance_propagation

Masked Dispersal
----------------

.. autofunction:: masked_density_flux

Density Network Dispersal
-------------------------

.. autofunction:: density_network

Fixed Network Dispersal
-----------------------

.. autofunction:: fixed_network

