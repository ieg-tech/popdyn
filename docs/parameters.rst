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

When inter-species relationships are specified, the following describes the input lookup table format:

.. code-block:: python
   :emphasize-lines: 1

   [(x1, y1), (x2, y2), ..., (xn, yn)]

Where `x` is the density of the *affecting* species, and `y` is the parameter value applied to the *affected* species.
Parameter values :math:`y_m` are linearly interpolated between lookup points, given the density :math:`x_m`:

.. math::
    y_m=y_0+\frac{x_m-x_0}{x_1-x_0}(y_1-y_0)

Stochastic processes may also be represented using Parameters. A single random number generator may be chosen to perturb
parameter data in a :ref:`domain` in one of two ways:

#. Random numbers are generated at each element (i.e. node or grid cell) in the domain during a simulation; or
#. A single random number is generated and applied evenly to all elements in the domain

The behaviour and type of the random number generators is flexible, and each of the Distributions available in the
`numpy.random module <https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html>`_ are available. For
example, if random numbers (:math:`X`) are chosen from a normal distribution, the input argument for the method would be the ``scale``,
or one standard deviation. Using method 1 (above), the value of each element :math:`param(i)` would be used for the mean value in the
normal distribution:

.. math::
    param_o(i)=X\sim\mathcal{N}(param(i),\sigma^2)

where,

:math:`param_o` is the stochastic parameter

If using method 2 (above), the mean value of the entire study area (:math:`\\mu`) would be used as the mean in the
normal distribution and a delta will be calculated at each element using the random value:

.. math::
    \mu=\frac{\sum_{i=1}^{n}param(i)[k_T(i)\neq0]}{n}

.. math::
    param_o(i)=param(i)+(X\sim\mathcal{N}(\mu,\sigma^2)-param(i))

.. math::
    param_o(i)=max\{0, param_o(i)\}

where,

:math:`k_T` is the total carrying capacity of the species.

Carrying Capacity
-----------------

Carrying Capacity is used in popdyn models to define means to enhance or reduce the ability for populations to exist in
a model domain. It is used numerically to relate populations to density, which drives a number of
model processes. When added to a model domain, Carrying Capacity data are effectively a population potential in the study
area, and are expressed as a total population for each data point. Multiple Carrying Capacity instances may be added to
a given :ref:`species` in a model :ref:`domain`, and they will be summed to yield the total Carrying Capacity :math:`k`
at time :math:`t` for each source :math:`i`:

.. math::
    k(t)=\sum_{i=1}^{n} k_i(t)

Carrying Capacity may also be derived using the density of another species and a lookup table (as outlined above),
which may be invoked using the :func:`add_as_species` method. The :math:`y` value in the lookup table is a coefficient
that is applied to the Carrying Capacity of the *affected* species:

.. math::
    k=yk

.. autoclass:: CarryingCapacity
    :inherited-members:
    :members:

Mortality
---------

Mortality is used to define and constrain ways for Species populations to decline. Any number of mortality drivers may
be created and added to a model domain. They are usually added to a model domain with rate data :math:`q`, which are used to
calculate a number of deaths (:math:`m`):

.. math::
    m=pop\cdot q

where :math:`pop` is the population that mortality is applied to.

Mortality may also be derived using the density of another species and a lookup table (as outlined above),
which may be invoked using the :func:`add_as_species` method. The :math:`y` value in the lookup table is a mortality
rate (:math:`q` above) used on the *affected* species.

One additional option when creating mortality is to define a recipient species to apply deaths to. This functionality was
conceived to perform infection and disease modelling, whereby infected and non-infected populations are treated as
separate species with their own unique traits. The rate of mortality in these cases would be the rate of infection, and
individuals would be transferred from one species to another.

.. autoclass:: Mortality
    :inherited-members:
    :members:

Fecundity
---------

Fecundity is used to create means for species to reproduce. Fecundity is used to specify:

* Whether a :ref:`species` is able to reproduce
* The effect of density on reproduction
* Allocation of offspring
* Effects another species may have on reproduction

Just as the other parameters, Fecundity can have stochasticity and may be derived using the density of another species,
based on a lookup table (as outlined above). Inter-species effects on fecundity are specified using the
:func:`add_as_species` method, where :math:`y` in the lookup table is a Fecundity rate (:math:`q`) used to determine the
number of offspring per individual (:math:`n`) of the *affected* species:

.. math::
    n=pop\cdot q

.. autoclass:: Fecundity
    :inherited-members:
    :members:
