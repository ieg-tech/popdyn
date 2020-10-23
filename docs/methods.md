# Population Dynamics Modelling Framework Methodology

The Population Dynamics modelling framework (herein referred to as PopDyn) is an open-source tool coded in Python that began development in 2016 under the direction of parties in government, academia, and industry. The PopDyn model is configured to use a spatially explicit domain with set boundaries and resolution, with flexibility integrated through temporal dynamics, modular species, modular mortality, and simulation methods. This study used the following components and modules to complete the simulation of population.

### Domain (spatial discretization)

The model domain is configured to align with a spatial grid, which is usually defined by a raster. The bounding box, cell width, cell height, and spatial reference may also be used directly in the absence of a dataset.

### Domain Parameterization

Model parameterization serves to add temporal variability, introduce species into the domain, and configure their dedicated parameters for sex and stage. Overarching parameters added to the domain include carrying capacity, population, fecundity, and mortality. These parameters and their data are defined outside of the model domain in modular fashion, where specific sub-parameters are defined. The parameters are then added to the model domain and associated with a species, sex, stage, and time in tandem. These datasets may be aggregated, spatially distributed, or derived from interspecies relationships, depending on how they are configured with sub-parameters. They may also be stochastic, where a random number generator picking from a specified distribution perturbs the parameter at prescribed times during simulation. The following describes the specifics of each parameter and how they are added to the model domain.

**_Carrying Capacity_**

Carrying capacity is represented as a maximum population possible at each element in the simulation grid. Multiple sources of carrying capacity may be used for a single species, which are aggregated during simulation. There are two ways to create carrying capacity parameters, which includes the use of static data, or defining a relationship with another species to create derived data. The former may be added to the model domain as a scalar or as a vector with variability throughout the study area. Derived carrying capacity is introduced into the domain as a relationship. During simulation, the relationship between the population or density of one or more other species is used to calculate a scaling factor for carrying capacity. 

**_Population_**

The model must be initialized with starting populations and carrying capacity at the first simulation time. Population is added as a scalar (population per cell), an aggregated total population, or as a vector to represent spatial variability. If aggregated initial populations are used, static carrying capacity at the cell-level is used to proportionally distribute the population over the landscape.  Population may also be added to times other than the initial time step, which serve to explicitly add or remove population.

**_Fecundity_**

Fecundity is an optional value of total offspring from a stage at a time in the model domain, with a ratio of male to female offspring (which may also be set to random). The value of fecundity is added to a stage as a scalar, a vector that is spatially distributed, or using an interspecies relationship. Fecundity may also vary using population density, where coefficients of minimum and maximum density, and rates of change scale fecundity linearly. Similar to carrying capacity, a relationship between the population or density of one or more species is used to derive a value of fecundity, or a scaling factor for fecundity during model simulation.

**_Mortality_**

One or more rates of mortality can be added to the model, which are aggregated and normalized (to not exceed 1) at each time. Similar to carrying capacity and fecundity, mortality can be added as a scalar, as a vector to represent spatial variability, or as an interspecies relationship. When interspecies relationships are used, the mortality rate is directly calculated during simulation using the provided relationship between population or density of other species. In addition to added mortality, two types are mortality are implicitly calculated in the model. The first is density-dependent mortality, which is defined using a threshold for density-dependent mortality to occur, and a scaling factor that dictates the maximum mortality rate, scaled linearly dependent on density. The second is utilized when the maximum age of a stage class is reached. If the stage is not able to live beyond their maximum specified age, mortality as a result of age will be directly proportional to the population that ages beyond the maximum value.

### Dispersal

Movement of populations from cell to cell in the domain is modular, as many or no methods may be used, and methods can be assigned variably to a species, sex, stage, or time. Two of the methods are described below.

**_Inter-habitat dispersal_**

Inter-habitat dispersal is calculated using a dispersion model that is driven by population density. A maximum distance is added to the domain, which is converted to a convolution matrix of cells filled with zero, with a circular pattern of ones. When dispersal occurs, the average population density in the convolution matrix is calculated for each cell. Density gradients are then derived between each cell and the average, which is used to calculate a linear proportion of population that moves to other cells in the convolution matrix, thus flattening the density gradient at each cell. 

**_Outward or “maximum” dispersal_**

Maximum dispersal is similar to inter-habitat dispersal, as it is added by providing a distance to the model and is driven by population density. The difference between the two is the convolution matrix used to flatten the density gradient at each cell. The distance provided as input is used to create a convolution matrix filled with zeros, with a single-width circle of ones. This dispersal method efficiently moves population to and from large distances, as it ignores the space between.

### Solving of Population

The PopDyn framework enables multiple methods to be used to solve population in the domain in a modular form. The most widely used solver, which is the only open-source solver included in the software package, is the “Discrete Explicit” method. This method solves population on a discrete time step from a start time (t0) to an end time (tn), while explicitly calculating the population using those of the previous time step. As such, the initial population of each age at t0 is not solved and is used to compute the numerical processes and graduate populations to the next age at t1 and so forth. Because of the explicit nature of the calculations, the order of operations within a timestep is described. At the beginning of each time step, the population for each stage from the previous time step is used to generate parameters, which includes single aggregated mortality, fecundity, and habitat values for each species, sex and stage at each cell. Total mortality and offspring are calculated, and the resulting populations are calculated for the current time step. Dispersal methods are then applied in a user-specified order, which results in populations that are susceptible to density-dependent and age-based mortality. Following the calculation of derived mortality, the final population and population distribution for the current time step are recorded.
