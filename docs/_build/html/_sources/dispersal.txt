.. _dispersal:

=========
Dispersal
=========

.. contents:: Table of contents
    :depth: 3
    :local:

.. py:currentmodule:: dispersal

Species may move independently in a model domain using any number of available dispersal methods. They are added to a
species using the :ref:`species` :func:`add_dispersal` method, which is specified using one of the available methods listed in
``popdyn.dispersal.METHODS``.

These methods, and their keys are used to add each to a species are listed below:

.. autosummary::
    density_flux
    distance_propagation
    masked_density_flux
    density_network
    fixed_network

.. note:: Poplulation is never moved outside of the model domain or to where carrying capacity is 0

A description of :ref:`mvp` is included herein as well, although it is a mortality driver. This method relies on
spatial distributions and is considered appropriate for this section.

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

.. _mvp:

Minimum Viable Population
-------------------------

.. autofunction:: minimum_viable_population