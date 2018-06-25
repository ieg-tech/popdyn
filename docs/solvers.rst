.. _solvers:

=================
Numerical Solvers
=================

.. contents:: Table of contents
    :depth: 6
    :local:

.. py:currentmodule:: popdyn

The object-oriented nature of :ref:`species`, :ref:`parameters`, and the :ref:`domain` allows flexibility and modularity
while solving of populations over time. Numerical solvers may be developed to address unique logic associated with projects
and may be swapped or combined while simulating populations.

All solvers include calls to the :ref:`domain` pre-solve check functions:

- :ref:`error_check`
- :ref:`inheritance`

Available, documented solvers include:

- :ref:`discrete_explicit` - :class:`solvers.discrete_explicit`

.. _error_check:

Error Checking
--------------

Error checking is performed using the following:

.. autofunction:: solvers.error_check

.. _inheritance:

Species Inheritance
-------------------

Population and Carrying Capacity data may be applied to a Species-level instance, although Sex and AgeGroup-level species
may exist in the model Domain. As such, these data are cascaded to children classes in advance of solving, which reduces
complexity of numerical solvers and their ability to recognize data inheritance. Inheritance typically involves:

#. Check whether children of Species and Sex-level classes exist in the model domain
#. Collect data related to parent classes
#. Divide parent data (:math:`param`) by the number of children (:math:`n`), :math:`\frac{param}{n}`

.. autofunction:: solvers.inherit

.. _discrete_explicit:

Discrete Explicit
-----------------

This solver is designed to solve species populations on a discrete time interval using explicit functions. At each time
step between the given start time and end time, species populations from the previous time step are collected and combined
with parameters from the current time step to calculate and save the populations at the current time step. As such, populations
must be provided at the first time step, and the first time step is not solved in the ``discrete_explicit`` solver.

Calculation graphs are created for each species, sex, age (stage) group, at each time using the
`dask library <https://dask.pydata.org/en/latest/>`_, which enables:

- Disk input/output to be optimized while reading HDF5 data
- Mathematical parameter optimization
- Parallelism where possible
- Memory-management to ensure RAM usage is predictable and constrained

**At each time step (starting with the second time step), populations are solved using the following methods**

.. note:: When describing calculations, note that they are performed independently at each element in the model domain

The two primary drivers during simulations are the carrying capacity and populations. At each time step, the total
populations of each sex (gender) if they are included in the domain are collected from the previous
time step:

.. math::
    p_{o(t)}=\sum_{i=1}^{n}p_{a(t-1)}

where :math:`p_o` is the initial population of each sex (gender) used in calculations,
:math:`p_a` is the population associated with each age and, and :math:`t` is time.

If the ``total_density`` keyword argument is specified, the total population of the entire species is also calcualted,
which will be used to calculate density relationships (as opposed to doing so independently for each sex or age (stage)
group).

Each species, sex (gender) and age (stage) group are then iterated and populations of each are iterated and solved
independently. While solving the population, parameters at the current time step are first collected for the species.

If parameters are dependent on inter-species relationships, the density of the contributing species :math:`\rho` is
calculated using the population :math:`p_o` from the previous time step, and the carrying capacity :math:`k` at the
current time step:

.. _density:

.. math::
    \rho=\frac{p_o}{k}

Circularity may exist when calculating inter-species relationships (i.e. two or more species have derived :math:`k`
based on one another), which is why derived carrying capacity is performed with a coefficient. In such cases, the
carrying capacity data are first collected for all interdependent species, and are subsequently used to calculate
derived values, rather than using derived values to calculate other derived values.

.. note:: Stochastic variability included in parameters is applied after values are derived from inter-species relationships

Parameters collected at the beginning of the time step include:

- :ref:`Mortality <mortality>` for each mortality driver
- The sum of Carrying Capacity - :math:`\sum_{i=1}^{n}k_{i(t)}`
- Total populations for the species, males, and females (if this level of stratification exists in the domain)
- Total populations of reproducing species
- Total populations of species that contribute to density (specified in the :ref:`species` keyword arg
  ``contributes_to_density``)
- :ref:`Density <density>`
- The sum of :ref:`Fecundity <fecundity>` - :math:`\sum_{i=1}^{n}f_{i(t)}`, where :math:`f` is the fecundity type

.. _fecundity:

Fecundity Method
^^^^^^^^^^^^^^^^

Fecundity values are scaled by two coefficients, which yields the *effective fecundity*:

#. A coefficient (:math:`y`) derived from a lookup table (methods explained :ref:`here <lookup>`.) based on density
   (:math:`x`).
#. A coefficient derived from density dependency on fecundity. This value is calculated using the
   ``density_fecundity_threshold``, ``density_fecundity_max``, and ``fecundity_reduction_rate`` keyword arguments in
   the :ref:`Fecundity <parameters>` class.

Density dependency on fecundity (coefficient 2 above) is calculated linearly, as follows:

.. math::
    min\{1,\frac{\rho-l}{max\{0,u-l\}}\}\cdot \Delta

where:

:math:`u` is ``density_fecundity_max``, the upper threshold

:math:`l` is ``density_fecundity_threshold``, the lower threshold

:math:`\Delta` is ``fecundity_reduction_rate``, the rate of fecundity reduction at the upper threshold

Prior to other calculations, total offspring (:math:`b`) at the current time step are calculated using the total
population of the reproducing group (ex. females) and the *effective fecundity* rate :math:`f_e`,

.. math::
    b(t)=p_o(t)f_e(t)

recalling that :math:`p_o` is the total population from :math:`t-1`.

Offspring are allocated to either males, females, or neither (depending on model parameterization) of the minimum age
in the domain at :math:`t+1`.

Following the calcualtion of offspring, all ages in the group (stage) are iterated and mortality, dispersal, and
:math:`t+1` propagation are completed.

.. _mortality:

Mortality Method
^^^^^^^^^^^^^^^^

Mortality is applied following the calculation of offspring in four steps while solving the model. The first step
includes the calculation of mortality as a result of all **mortality driver** parameters. Mortality as a result of each
driver, :math:`m`, is calculated using :math:`p_o`:

.. math::

    m_{i(t)}=p_{o(t)}q_{i(t)}

where :math:`i` is the mortality driver with the rate :math:`q`.

In the case where the sum of all mortality driver rates exceed 1, they are scaled proportionally:

.. math::

    m_{i(t)}=\frac{q_{i(t)}}{\sum_{i=1}^{n}q_{i(t)}}\cdot p_{o(t)}

Following the drivers, the remaining mortality calculations are a result of **Implicit mortality types**. These include:

- Old age; and
- Density-dependent mortality.

Populations are first moved using any provided :ref:`Dispersal <dedispersal>` methods prior to applying
density-dependent mortality. Density-dependent mortality is calculated using the ``density_threshold`` and
``density_scale`` keyword arguments provided to the :ref:`species` constructor:

.. math::

    \frac{max\{0,\rho-l\}}{1-l}\cdot \Delta

where,

:math:`\rho` is the population density

:math:`l` is ``density_threshold``, and is the lower limit of density dependent mortality

:math:`\Delta` is ``density_scale``, and is the rate of density dependent mortality at a density of 1.

Old age mortality is only applied if the ``live_past_max`` keyword argument in the :ref:`species` instance is ``False``.
In this case, populations of the maximum specified age in the domain will not propagate to :math:`t+1`, and will be
tracked as mortality as a result of old age.

Prior to propagating a population to :math:`t+1`, the ``minimum_viable_population`` keyword argument of the species is
checked to determine whether a :ref:`minimum viable population <mvp>` calculation should take place. This form of
mortality is applied last at the current time step.

.. _dedispersal:

Dispersal Method
^^^^^^^^^^^^^^^^

Dispersal is applied prior to implementing density-dependent mortality to give populations a chance to move before they
succumb to the effects of density. Dispersal is applied using any number of the :ref:`dispersal` methods inherent to
species, in the order that they were applied to the :ref:`species` object.

Conversion & Immigration/Emigration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two processes that take place during solving are :ref:`conversion <conversion>` from one species to another, and
immigration/emigration. The latter are applied by adding populations to the domain at time slices other than the first
time in the simulation. Positive values will apply (immigrate) populations to the domain at a given time, while
negative values will enforce a population loss from the system (emigration).

**The Discrete Explicit Class**

.. autoclass:: solvers.discrete_explicit
    :members:
