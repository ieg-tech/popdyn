.. _solvers:

=================
Numerical Solvers
=================

.. contents:: Table of contents
    :depth: 5
    :local:

.. py:currentmodule:: popdyn

Species Inheritance
-------------------

Prior to commencing population calculations over time, solvers must perform error check and species inheritance:

.. autosummary::
    solvers.error_check
    solvers.inherit

Both population and carrying capacity data in the domain are distributed evenly among their children species.

Discrete Explicit
-----------------

.. autofunction:: solvers.discrete_explicit
