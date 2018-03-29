# PopDyn
## Description
This library provides a population dynamics simulation toolset that is both simple and powerful, providing a deeply 
extensible platform that should be comfortable for those versed in Python.

## Installation
#### For Python versions 2.7 only (3.X support coming soon)

From the project root, use install the module using setup.py:

```bash
pip install .
```

**_or_**

Install only the library dependencies using requirements.txt:

```bash
pip install -r requirements.txt
```

Run the tests

```bash
python tests/got.py
```

## Getting Started

### The model domain
A population dynamics model is comprised of a domain that defines the boundary conditions of a study area. At a 
minimum, a domain:

* Defines the spatial extent (region) of a study area
* Defines a spatial resolution (cell size)
* Creates a `.popdyn` file in HDF5 format that acts as a pseudo-database to efficiently access gridded data
* Keeps track of other spatial parameters, such as the spatial reference, and geotransform

#### Defining a model domain
A model domain is created by instantiating the `popdyn.Domain` class, which must be directed to a `.popdyn`
 file path, and is constructed using:

1. An existing `.popdyn` file

```python
import popdyn as pd
my_popdyn_path = 'model_v1.popdyn'  # This file exists

my_domain = pd.Domain(my_popdyn_path)
```

2. A raster with the domain comprised of valid data values
    
```python
import popdyn as pd
my_popdyn_path = 'model_v1.popdyn'  # This file does not exist yet

my_domain = pd.Domain(my_popdyn_path, 'path_to_raster.tif')  # Any GDAL-supported raster format is acceptable
```
    
3. Using `kwargs` to manually define each domain parameter
    
```python
import popdyn as pd
my_popdyn_path = 'model_v1.popdyn'  # This file does not exist yet

kwargs = {'shape': (1500, 1200),   # (rows, cols)
          'csx': 100,              # Cell size in the x-direction
          'csy': 100,              # Cell size in the y-direction
          'top': 654378.26,        # Top edge northing
          'left': 5669309.19,      # Left edge easting
          'projection': 26911,     # Spatial references - must be an SRID (EPSG code)
}

my_domain = pd.Domain(my_popdyn_path, **kwargs)
```
    
#### Adding data to the Domain
Species, mortality, carrying capacity, and populations are systematically added to the model domain, but consist of 
separate objects. As such, they will first be described before explaining how they are linked to the domain.
    
### Species
Species are created using one of the `popdyn.Species`, `popdyn.Sex`, or `popdyn.AgeGroup` classes. Their inheritance 
is configured as follows:

```text
Species [Has a name, unique method of migration, and may or may not contribute to population density]
    |
    Sex [Child of Species, is a male or female, and has a specified fecundity]
      |
      AgeGroup [Child of Sex, has a group name, and a range of ages]
```

Any number of species may be added to the model domain, and become organized by their name.
For example, multiple species objects may be created for the same species:

```python
# An overarching moose species instance may be used to define carrying capacity, mortality, or migration methods that
# apply to all children (Sex and/or AgeGroup) instances
moose = pd.Species('Moose')

# Subclass males so they may have their own mortality types
moose_males = pd.Sex('Moose', 'male')  # The name of the sex is limited to 'male' and 'female'

# Subclass age groups for females so they may have varying fecundity by age
moose_female_yoy = pd.AgeGroup('Moose', 'female', 'Young of Year', fecundity=1.1)
moose_female_adult = pd.AgeGroup('Moose', 'female', 'Adult', fecundity=2.5)
# etc...
```

The important factor in the above example is that each Species instance has an identical name - `Moose`. This tells the
domain they are all of a single species.

### Carrying Capacity
Carrying capacity (_k_) objects are created using the `popdyn.CarryingCapacity` class. Instances of _k_ are created in 
the simplest form by providing a name.

```python
moose_k = pd.CarryingCapacity('Moose Habitat')
```

_k_ instances also serve to define stochastic perturbations, and inter-species relationships in the model.
For example, randomness may be applied to the above _k_ instance:

```python
# Use a random number generator with a noraml distribution and a standard deviation of 10
moose_k.random('normal', args=(10,))
```

Available distributions for the random number generator can be obtained by checking the `dynamic.RANDOM_METHODS` 
variable. Also, the `args` keyword contains the necessary arguments for the random number function. In this case, the 
`normal` function only requires a standard deviation value.

_k_ may also be defined dynamically based on the population of another species. This relationship is added using a
 species instance, and a lookup table.
 
```python
# Create a species that affects Moose habitat
willow = pd.Species('Willow')
 
# For demonstration purposes, the density of willows enhances moose habitat using the following relationship:
willow_lookup = [(0., -0.9), (0.5, 0.7), (1., 0.9)]
'''
Which effectively means:
    -if the willow density is 0, there is a 90% reduction in Moose habitat
    -if the willow density is 0.5, there is an 70% increase in Moose habitat
    -if the willow  density is 1, there is a 90% increase in Moose habitat
Any values between are calculated using linear interpolation.
'''

# Create a habitat based on willow density
k_willows = pd.CarryingCapacity('Willow Habitat')
k_willows.add_as_species(willow, willow_lookup)
```

### Mortality
Mortality objects are created using the `popdyn.Mortality` class. Instances of Mortality are created in 
the simplest form by providing a name.

```python
poaching = pd.Mortality('Poaching')
```

Just as Carrying Capacity, Mortality instances serve to define stochastic perturbations, and inter-species 
relationships in the model. These are added to mortality instances in an identical manner; for example:

```python
# Create a species that may be used as a means of moose mortality
ticks = pd.Species('Ticks')

# For demonstration purposes, ticks impose mortality rates on moose using the following relationship:
tick_lookup = [(0., 0.), (1., 0.1)]
'''
Which effectively means:
    -if the tick density is 0, 0% of Moose succumb to ticks
    -if the willow density is 1, 10% of Moose succumb to ticks
Any values between are calculated using linear interpolation.
'''

tick_mortality = pd.Mortality('Ticks')
tick_mortality.add_as_species(ticks, tick_lookup)

# Add some randomness
tick_mortality.random('chi-square', args=(2,))
```

### Adding data to the Domain
Species, Mortality, Carrying Capacity, and population (which has not yet been covered) are added into any model domain
 instance using methods tailored to each one.
 
Species are implicitly added into the domain by calling any of:

* `popdyn.Domain.add_carrying_capacity`
* `popdyn.Domain.add_mortality`
* `popdyn.Domain.add_population`

When any of these parameters rely on data, they may be added using one of:

1. A raster. Rasters added to the domain will be resampled to match the spatial reference, extent, and cell size using
nearest neighbour interpolation.
2. A `numpy.ndarray` with dimensions matching the domain.
3. A scalar

Data added to the domain may also be distributed using various methods, which are specified by `kwargs` and include:

* `overwrite` - the method used to apply data where they already exist in the domain. 
* `distribute` - Divide the sum of the input data (or input scalar) among the active cells in the domain. For example, 
if `1000` is the input, and there are `100` active cells in the domain, each one will receive a value of `10`.
* Specific to population:
    - `distribute_by_k` - Divide the sum of the input data (or input scalar) among the domain using the values of 
    the species carrying capacity. The relationship between population and _k_ is linear in this case.
    - `discrete_age` - If the species is an age group, population may be applied to a discrete age in the group. 
    Otherwise, they are divided evenly among the range of ages.
* Specific to Carrying Capacity:
    - `is_density` - Defines whether the input data are a density, or are a discrete number 
    of the population.
* Specific to mortality:
    - `distribute_by_co` - Divide the sum of the input data (or input scalar) among the domain using the values of a
    covariate parameter. The relationship between mortality and the covariate is linear.
 
Adding data is best demonstrated by extending the previous examples (time will be discussed in further detail later):
 
```python
# Add a carrying capacity dataset to moose at time 0
#    add_carrying_capacity(self, species, carrying_capacity, time, data=None, **kwargs)
my_domain.add_carrying_capacity(moose, moose_k, 0, 'moose_k.tif')

# Add an initial populations of moose to the model domain using an empirical dataset at time 0
#    add_population(self, species, data, time, **kwargs)
my_domain.add_population(moose, 'moose_population.tif', 0)  # Third argument is time

# Note, both were added using the moose "Species" instance. This is applied to the Sex and AgeGroup children,
# And will be discussed later in Inheritance

# Apply poaching to moose males at times 0 to 50, with a rate of 0.05 everywhere
for time in range(50):
    my_domain.add_mortality(moose_males, poaching, time, 0.05)

# Because moose habitat and mortality are reliant on willows and ticks, they must also be solved in the model
# Create willow habitat, and assume it is 10 everywhere
my_domain.add_carrying_capacity(willow, pd.CarryingCapacity('Willow Habitat'), 0, 10.)

# Add an initial population of 100000 willows
my_domain.add_population(willow, 100000, 0)  # the "distribute" kwarg is True by default

# Add tick habitat using a raster
my_domain.add_carrying_capacity(ticks, pd.CarryingCapacity('Tick Habitat'), 0, 'tick_habitat.tif')

# Add a population of 1000000 ticks at time 0, and distribute it based on k
my_domain.add_population(ticks, 1000000., 0, distribute_by_habitat=True)

# Add willows to moose habitat
# No data are provided because the willows species was added to k_willows
my_domain.add_carrying_capacity(moose, k_willows, 0)

# Add ticks to moose mortality, but only for females of two age groups at time 10
my_domain.add_mortality(moose_female_yoy, tick_mortality, 10)
my_domain.add_mortality(moose_female_adult, tick_mortality, 10)
```

### Inheritance
As many species and respective populations, mortalities, and carrying capacity may be placed into the domain as needed.
Before solving the domain, both population and _k_ are distributed among children species instances. For example, if a
population for a species is introduced into the model domain at a given time, and one Sex, and two AgeGroups also 
exist, the population will be divided as follows:

```text
                Species Population
                        |
                   divided by 2
                        |
                /--------------\
              male           female
                                |
                           divided by 2
                                |
                        /---------------\
                   Age Group 1      Age Group 2
```

Mortality is different, in that inheritance occurs during model execution. Mortality that exists for a parent species
is inherited by children species. Inheritance for mortality may be turned off globally by modifying the `Domain`
attribute:

```python
popdyn.Domain.avoid_inheritance = True
```

When multiple _k_ data are introduced into the domain for a given species, they are summed to produce
an aggregate _k_ dataset at each time in the simulation.

### Time
Time is not limited to a single unit, however the time unit in a domain must be static, and is limited to integer
(whole number) precision.

**Ensuring that all rates (fecundity, mortality, etc.) adhere to the chosen time units is imperative**

When data are added into the domain, they must be attached to a time slice. In the Moose example above, poaching
mortality data are added at all possible times, ranging from `0` to `49`. If mortality is to have an effect during a
model simulation, it must be included at the time step of execution.

_k_ is different than mortality, in that if no _k_ data are specified for a given time, a _k_
dataset backwards in time will be used. If no carrying capacity data are specified, the species will not propagate.

### Migration

Any number of dispersal methods may be used for a given species, and are added to species instances individually
 using the method. As a result, any age group, sex, or entire species may have different migration methods.
 
```popdyn.Species.add_dispersal(dispersal_type, args)```

Implemented methods of dispersal can be obtained by calling `popdyn.dispersal.METHODS`

### Solving the model
Currently only one solver (`discrete_explicit`) is implemented, but any number of solvers may be coded to solve 
population in the domain. The current solver works by passing the model domain and the desired time duration:

```python
solvers.discrete_explicit(my_domain, 0, 49)  # 0 is the start time, and 49 is the last time to simulate
```

The `.popdyn` file will be updated with simulated populations during execution, while also storing results of other
calculations, such as _k_ (in aggregate), migration, and mortality.