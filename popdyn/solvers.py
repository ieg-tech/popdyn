"""
Population dynamics numerical solvers

Devin Cairns, 2018
"""
from popdyn import *
import dask.array as da


class SolverError(Exception):
    pass


def error_check(domain):
    """
    Make sure there are no conflicts within a domain.

    :param Domain domain: Domain instance
    :return:
    """
    def test(sex_d):
        """Collect all discrete ages from a sex"""
        ages = []
        for age_group in sex_d.keys():
            if age_group is None:
                continue

            instance = sex_d[age_group]
            if any([isinstance(instance, obj) for obj in [Species, Sex, AgeGroup]]):
                ages += range(instance.min_age, instance.max_age + 1)

        if np.any(np.diff(np.sort(ages)) == 0):
            raise SolverError(
                'The species (key) {} {}s have group age ranges that overlap'.format(
                    species_key, sex)
            )

    # Make sure at least one species exists in the domain
    species_keys = domain.species.keys()
    if len(species_keys) == 0:
        raise SolverError('At least one species must be placed into the domain')

    # Iterate species and ensure there are no overlapping age groups
    for species_key in species_keys:
        # Iterate sexes
        for sex in domain.species[species_key].keys():
            # Collect the ages associated with this sex if possible
            if sex is None:
                continue
            test(domain.species[species_key][sex])

    # Check that species with relationships are both included
    def get_species(d):
        for key, val in d.items():
            if isinstance(val, dict):
                get_species(val)
            elif isinstance(val, tuple):
                if val[0].species is not None:
                    species.append(val[0].species.name_key)

    for ds_type in ['carrying_capacity', 'mortality']:
        species = []
        get_species(getattr(domain, ds_type))
        if not np.all(np.in1d(np.unique(species), species_keys)):
            # Flesh out the species the hard way for a convenient traceback
            for sp in species:
                if sp not in species_keys:
                    raise SolverError(
                        'The domain requires the species (key) {} to calculate {}'.format(
                            sp, ds_type.replace('_', ' '))
                    )


def inherit(domain, start_time, end_time):
    """
    Divide population and carrying capacity among children of species if they exist
    :param domain:
    :param int start_time:
    :param int end_time:
    :return:
    """
    # Iterate all species and times in the domain
    for species in domain.species.keys():
        sex_keys = [key for key in domain.species[species].keys() if key is not None]
        for time in range(start_time, end_time + 1):
            # Collect the toplevel species datasets if they exist, and distribute them among the sexes
            population_ds = cc_ds = None
            species_population_key = domain.get_population(species, time)
            # Multiple carrying capacity datasets may exist
            carrying_capacity = domain.get_carrying_capacity(species, time, snap_to_time=False)

            # Population
            if species_population_key is not None and len(sex_keys) > 0:
                # Read the species-level population data and remove from the domain
                population_ds = domain[species_population_key][:] / len(sex_keys)
                domain.remove_dataset('population', species, None, None, time, None)  # Group, sex and age are None
            # Carrying capacity
            if len(carrying_capacity) > 0 is not None and len(sex_keys) > 0:
                # Read the species-level carrying capacity data and remove from the domain
                cc_ds = []
                for cc in carrying_capacity:
                    if cc[1] is not None:
                        cc_ds.append((cc[0], cc[1][:] / len(sex_keys)))
                    else:
                        cc_ds.append((cc[0], None))
                    # Remove each dataset under the name key
                    domain.remove_dataset('carrying_capacity', species, None, None, time, cc[0].name_key)

            # Iterate any sexes in the domain
            for sex in sex_keys:
                # Collect groups
                group_keys = [key for key in domain.species[species][sex].keys() if key is not None]
                if len(group_keys) == 0:
                    # Apply any species-level data to the sex dataset
                    if population_ds is not None:
                        instance = domain.species[species][sex][None]
                        print("Cascading population {} --> {}".format(species, sex))
                        domain.add_population(instance, population_ds, time,
                                              distribute=False, overwrite='add')
                    if cc_ds is not None:
                        instance = domain.species[species][sex][None]
                        for cc in cc_ds:
                            print("Cascading carrying capacity {} {} --> {}".format(cc[0].name, species, sex))
                            if cc[1] is not None:
                                domain.add_carrying_capacity(instance, cc[0], time, cc[1],
                                                             distribute=False, overwrite='add')
                            else:
                                domain.add_carrying_capacity(instance, cc[0], time, distribute=False, overwrite='add')
                else:
                    # Combine the species-level and sex-level datasets for population
                    # ---------------------------------------------------------------
                    sex_population_key = domain.get_population(species, time, sex)
                    if sex_population_key is None:
                        if population_ds is None:
                            # No populations at species or sex level
                            continue
                        else:
                            # Use only the species-level key divided among groups
                            sp_sex_prefix = '{} --> [{}] --> '.format(species, sex)
                            next_ds = population_ds / len(group_keys)
                    else:
                        # Collect the sex-level data and remove them from the Domain
                        next_ds = domain[sex_population_key][:]
                        domain.remove_dataset('population', species, sex, None, time, None)  # Group and age are None

                        # Add the species-level data if necessary and divide among groups
                        if population_ds is not None:
                            sp_sex_prefix = '{} --> {} --> '.format(species, sex)
                            next_ds += population_ds
                        else:
                            sp_sex_prefix = '[{}] --> {} --> '.format(species, sex)
                        next_ds /= len(group_keys)

                    # Combine the species-level and sex-level datasets for carrying capacity
                    # ----------------------------------------------------------------------
                    # All carrying capacity datasets are appended, and not added together
                    next_cc_ds = []
                    if cc_ds is not None:
                        # Use only the species-level keys divided among groups
                        for cc in cc_ds:
                            if cc[1] is None:
                                next_cc_ds.append(cc)
                            else:
                                next_cc_ds.append((cc[0], cc[1] / len(group_keys)))

                    sex_carrying_capacity = domain.get_carrying_capacity(species, time, sex, snap_to_time=False)
                    # Collect the sex-level data, append them to groups, and remove them from the Domain
                    for cc in sex_carrying_capacity:
                        if cc[1] is None:
                            next_cc_ds.append(cc)
                        else:
                            next_cc_ds.append((cc[0], cc[1][:] / len(group_keys)))
                        # Remove each dataset under the name key
                        domain.remove_dataset('carrying_capacity', species, sex, None, time, cc[0].name_key)

                    # Apply populations and carrying capacity to groups
                    for group in group_keys:
                        instance = domain.species[species][sex][group]
                        print("Cascading population {}{}".format(sp_sex_prefix, group))
                        domain.add_population(instance, next_ds, time,
                                              distribute=False, overwrite='add')
                        for cc in next_cc_ds:
                            print("Cascading carrying capacity {} -- > [{}] --> {}".format(cc[0].name, sex, group))
                            if cc[1] is None:
                                domain.add_carrying_capacity(instance, cc[0], time,
                                                             distribute=False, overwrite='add')
                            else:
                                domain.add_carrying_capacity(instance, cc[0], time, cc[1],
                                                             distribute=False, overwrite='add')


class discrete_explicit(object):
    """
    Solve the domain from a start time to an end time on a discrete time step,
    which is dictate by the Domain configuration.

    The total populations from the previous year are used to calculate derived
    parameters (density, fecundity, inter-species relationships, etc.).

    To run the simulation, use .execute()
    """
    def time_this(func):
        """Decorator for profiling methods"""
        def inner(*args, **kwargs):
            instance = args[0]
            start = profile_time.time()
            execution = func(*args, **kwargs)
            try:
                instance.D.profiler['solver ' + func.__name__] += profile_time.time() - start
            except KeyError:
                instance.D.profiler['solver ' + func.__name__] = profile_time.time() - start
            return execution
        return inner

    def __init__(self, domain, start_time, end_time, **kwargs):
        """
        :param domain: Domain instance populated with at least one species
        :param start_time: Start time of the simulation
        :param end_time: End time of the simulation
        :param kwargs:
            :total_density: Use the total species population for density calculations (as opposed to
            populations of sexes or groups)
        :return: None
        """
        # Prepare time
        start_time = Domain.get_time_input(start_time)
        end_time = Domain.get_time_input(end_time)
        self.simulation_range = range(start_time + 1, end_time + 1)

        self.D = domain
        self.prepare_domain(start_time, end_time)

        # Extra toggles
        # -------------
        self.total_density = kwargs.get('total_density', True)


    @time_this
    def prepare_domain(self, start_time, end_time):
        """Error check and apply inheritance to domain"""
        error_check(self.D)
        inherit(self.D, start_time, end_time)

    def execute(self):
        """Run the simulation"""
        # Iterate time. The first time step cannot be solved and serves to provide initial parameters
        for time in self.simulation_range:
            # Hierarchically traverse [species --> sex --> groups --> age] and solve children populations
            # Each are solved independently, deriving relationships from baseline data
            for species in self.D.species.keys():

                # Collect totals for entire species-based calculations
                # The self.population_arrays and self.carrying_capacity_arrays are updated
                self.species_totals(species, time)

                # Collect sex keys
                sex_keys = self.D.species[species].keys()  # Sex - may not exist and will be None

                # All births are recorded from each sex are added to the age 0 slot
                self.age_zero_population = {key: da.zeros(self.D.shape, 'float32', chunks=self.D.chunks)
                                            for key in sex_keys}

                # Iterate sexes and groups and calculate each individually
                for sex in sex_keys:
                    for group in self.D.species[species][sex].keys():  # Group - may not exist and will be None
                        self.propagate(species, sex, group, time)

    @time_this
    def propagate(self, species, sex, group, time):
        """
        INTERNAL factory for propagating solved populations and recording intermediates

        Note:
            -If age gaps exist, they will be filled by empty groups and will propagate
            -Currently, female:male ratio uses the total species population even if densities are calculated at the
            age group level
        :param species:
        :param sex:
        :param group:
        :param time:
        :return:
        """
        # Collect parameters from the current time step.
        # All dynamic modifications are also applied in this step
        # =============================================
        params = self.calculate_parameters(species, sex, group, time)

        # Collect some meta on the species
        species_instance = self.D.species[species][sex][group]
        ages = species_instance.age_range
        max_age = self.D.discrete_ages(species, sex)[-1]
        key_prefix = '{}/{}/{}/{}/params'.format(species, sex, group, time)

        # Collect mortality names
        mort_types = [mort_type for mort_type in self.D.mortality[species][sex][group][time].keys()
                      if len(mort_type) > 0]

        # All outputs are written at once, so create a dict to do so (pre-populated with mortality types as zeros)
        output = {'{}/mortality/{}'.format(key_prefix, mort_type):
                  da.zeros(self.D.shape, 'float32', chunks=self.D.chunks) for mort_type in mort_types}

        # Calculate births using the effective fecundity if the group is female
        if sex == 'female':
            # The total population of this group is used, as fecundity will be 0
            # within groups that do not reproduce
            births = params['Population'] * params['Fecundity']
            male_births = births * params['Birth Ratio']
            female_births = births - male_births

            # Add new offspring to males or females
            for _sex in self.age_zero_population.keys():
                zero_group = self.D.group_from_age(species, _sex, 0)
                if zero_group is None:
                    # No ages would exist if there is no group
                    _age = None
                key = '{}/{}/{}/{}/{}'.format(species, _sex, zero_group, time, _age)
                self.D.population[species][_sex][zero_group][time][_age] = key
                if _sex == 'female':
                    # Add female portion of births
                    self.age_zero_population[key] += female_births
                    output['{}/offspring/{}'.format(key_prefix, 'Female Offspring')] = female_births
                else:
                    self.age_zero_population[key] += male_births
                    output['{}/offspring/{}'.format(key_prefix, 'Male Offspring')] = male_births

        for age in ages:
            # Collect the population of the age from the previous time step
            population = self.D[self.D.get_population(species, time - 1, sex, group, age)]
            population = da.from_array(population)

            # Mortaliy is applied on each age, as they need to be removed from the population
            for mort_type in mort_types:
                output['{}/mortality/{}'.format(key_prefix, mort_type)] = params[mort_type] * population
            # Reduce the population by the mortality
            for mort_type in mort_types:
                population -= output[mort_type]

            # Apply old age mortality if necessary to avoid unnecessary dispersal calculations
            if not species_instance.live_past_max and max_age is not None and age == max_age:
                output['{}/mortality/{}'.format(key_prefix, 'Old Age')] = population

            else:
                # Apply dispersal to remaining population
                for dispersal_method, args in species_instance.dispersal.items():
                    population = dispersal.apply(population, dispersal_method, args)

                # Apply density-dependent mortality
                output['{}/mortality/{}'.format(key_prefix, 'Density Dependent')] = da.where(
                    population > params['Carrying Capacity'], population - params['Carrying Capacity'], population
                )
                population -= output['{}/mortality/{}'.format(key_prefix, 'Density Dependent')]

                # Propagate age by one increment and save to the current time step.
                # If the input age is None, the population will simply propagate
                if age is not None:
                    new_age = age + 1
                new_group = self.D.group_from_age(species, sex, new_age)
                if new_group is None:
                    # This age does not exist in the domain
                    new_age = None
                key = '{}/{}/{}/{}/{}'.format(species, sex, new_group, time, new_age)
                self.D.population[species][sex][new_group][time][new_age] = key
                # In case a duplicate key exists, addition is first attempted
                try:
                    output[key] += population
                except KeyError:
                    output[key] = population

        # Add carrying capacity and zero populations to writeable datasets
        output['{}/carrying capacity/{}'.format(key_prefix, 'Carrying Capacity')] = params['Carrying Capacity']
        output.update(self.age_zero_population)

        # Compute and write output
        self.D.domain_compute(output)


    @time_this
    def collect_parameter(self, param, data, time):
        """Collect keyword args for dynamic data collection from a species"""
        # Collect the total species population from the previous time step
        kwargs = {}
        if data is None:  # There must be a species attached to the parameter instance if this is True
            # Collect the total population of the species
            ds = self.D.all_population(param.species.name_key, time - 1, param.species.sex, param.species.group_key)
            a = da.dstack([da.from_array(d, d.chunks) for d in ds]).sum(axis=-1)
            # Add to the kwargs for dynamic collection
            kwargs = {'lookup_data': a, 'lookup_table': param.species.species_table}
        else:
            data = da.from_array(data, data.chunks)

        # Include random parameters
        if param.get('random', False):
            kwargs.update({'random_method': param.random, 'random_args': param.random_args})

        return dynamic.collect(data, **kwargs)  # A dask array

    @time_this
    def species_totals(self, species, time):
        """
        Collect all population and carrying capacity datasets, and calculate species totals if necessary
        :param species:
        :param time:
        :return:
        """
        # Collect all population keys (from previous time step for baseline) to avoid repetitive IO
        population_arrays = {key: da.from_array(ds, ds.chunks)
                             for key, ds in self.D.all_population(species, time - 1).items()}
        # Collect all carrying capacity keys to avoid repetitive IO and calculations
        carrying_capacity_arrays = self.D.all_carrying_capacity(species, time)
        carrying_capacity_arrays = {
            cc[0]: self.collect_parameter(cc[0], cc[1], time)
            for cc in carrying_capacity_arrays
        }
        if len(carrying_capacity_arrays) == 0:
            carrying_capacity_arrays = {None: da.from_array(np.broadcast_to(0, self.D.shape), self.D.chunks)}

        # Need to collect the total population for fecundity and density if not otherwise specified
        if self.total_density:
            # Population
            ds = []
            for key in population_arrays.keys():
                if self.D.instance_from_key(key).contributes_to_density:
                    ds.append(population_arrays[key])
            population_arrays['total'] = da.dstack(ds).sum(axis=-1)

            # Carrying Capacity
            carrying_capacity_arrays['total'] = da.dstack(
                [ds[1] for ds in carrying_capacity_arrays]
            ).sum(axis=-1)
            carrying_capacity_arrays['total'] = da.where(
                carrying_capacity_arrays['total'] < 0, 0, carrying_capacity_arrays['total']
            )

        # Calculate species-wide male and female populations that reproduce
        ds = []
        for key in self.D.all_population(species, time - 1, 'male').keys():
            if self.D.instance_from_key(key).get('fecundity', False):
                ds.append(population_arrays[key])
        population_arrays['Reproducing Males'] = da.dstack([ds]).sum(axis=-1)
        ds = []
        for key in self.D.all_population(species, time - 1, 'female').keys():
            if self.D.instance_from_key(key).get('fecundity', False):
                ds.append(population_arrays[key])
        population_arrays['Reproducing Females'] = da.stack([ds]).sum(axis=-1)

        # Calculate Density
        population_arrays['Female to Male Ratio'] = da.where(
            population_arrays['Reproducing Males'] > 0,
            population_arrays['Reproducing Females'] / population_arrays['Reproducing Males'],
            np.finfo('float32').max
        )

        self.population_arrays = population_arrays
        self.carrying_capacity_arrays = carrying_capacity_arrays

    @time_this
    def calculate_parameters(self, species, sex, group, time):
        """INTERNAL Collect all required parameters at a time step"""
        parameters = {}

        # Mortality
        # ---------------------
        # Collect Mortality instances and/or data from the domain at the current time step
        mortality = self.D.get_mortality(species, time, sex, group)
        if len(mortality) == 0:
            # Defaults to 0
            parameters['Mortality'] = da.from_array(np.broadcast_to(0, self.D.shape), self.D.chunks)
        else:
            for instance, data in mortality:
                # Mortality types remain separated by name in order to track numbers associated with types,
                # and cannot be less than 0
                parameters[instance.name] = self.collect_parameter(instance, data, time)
                parameters[instance.name] = da.where(parameters[instance.name] < 0, 0, parameters[instance.name])

        # Carrying Capacity -
        # ---------------------
        # Collect CarryingCapacity instances and/or data from the domain at the current time step
        if not self.total_density:
            carrying_capacity = self.D.get_carrying_capacity(species, time, sex, group)
            if len(carrying_capacity) == 0:
                # It must default to 0
                parameters['Carrying Capacity'] = da.from_array(np.broadcast_to(0, self.D.shape), self.D.chunks)
            else:
                # Collect datasets from all carrying capacity data and calculate their sum
                parameters['Carrying Capacity'] = da.dstack(
                    [self.carrying_capacity_arrays[key] for key, _ in carrying_capacity]
                ).sum(axis=-1)
                parameters['Carrying Capacity'] = da.where(
                    parameters['Carrying Capacity'] < 0, 0, parameters['Carrying Capacity']
                )

        else:
            parameters['Carrying Capacity'] = self.carrying_capacity_arrays['total']

        # Total Population from previous time step
        # ---------------------
        population, total_population = [], []
        for key in self.D.all_population(species, time - 1, sex, group).keys():
            total_population.append(self.population_arrays[key])
            if self.D.instance_from_key(key).contributes_to_density:
                population.append(self.population_arrays[key])
        parameters['Population'] = da.dstack([total_population]).sum(axis=-1)

        # Density calculation only includes contributing species instances
        if not self.total_density:
            population = da.dstack([population]).sum(axis=-1)
        else:
            population = self.population_arrays['total']

        # Calculate density
        # ---------------------
        # This will reflect the child density if domain.total_density is False
        parameters['Density'] = da.where(parameters['Carrying Capacity'] > 0,
                                         population / parameters['Carrying Capacity'],
                                         0)

        # Collect fecundity parameters if the sex is female
        # ---------------------
        if sex == 'female':
            # Change species into the instance
            species = self.D.species[species][sex][group]
            # Create a fecundity modifier array that includes some input parameters,
            # and is also used to broadcast the species fecundity value
            no_mod = True
            if species.get('fecundity_lookup', None) is not None:
                no_mod = False
                fec_mod = dynamic.collect(
                    None, lookup_data=self.population_arrays['Female to Male Ratio'],
                    lookup_table=species.fecundity_lookup
                )
            # TODO: Make these attributes, and complete the math here
            # if species.min_fecundity and species.max_fecundity:
            #     no_mod = False
            #     fec_mod = calc_stuff

            if no_mod:
                fec_mod = da.from_array(np.ones(shape=self.D.shape, dtype='float32'), self.D.chunks)

            parameters['Fecundity'] = fec_mod * species.get('fecundity', 0)

            if species.get('fecundity_random', False):
                parameters['Fecundity'] = dynamic.collect(
                    parameters['Fecundity'], random_method=species.fecundity_random,
                    random_args=species.fecundity_random_args
                )

            parameters['Fecundity'] = da.where(parameters['Fecundity'] > 0, parameters['Fecundity'], 0)

            if species.get('birth_random', None) == 'random':
                parameters['Birth Ratio'] = np.random.random()
            else:
                parameters['Birth Ratio'] = species.get('birth_ratio', None)

        return parameters


class retired(object):
    def __init__(self):
        pass

    def retired(self, duration, **kwargs):
        """
        Compute populations from start_time over a duration using the inherent
        time step.

        max_age provides the ability to automatically truncate the population
        to a maximum age.

        max_rate determines whether animals may move to other cells.
        """
        # Collect arguments
        mf_ratio = kwargs.get('mf_ratio', 'random')
        max_age = kwargs.get('max_age', True)
        max_rate = kwargs.get('max_rate', 0.)
        movement_alg = kwargs.get('movement_alg', 'dispersive-cancer')
        relocation_distance = kwargs.get('relocation_distance', 0.)
        fecundity_threshold = kwargs.get('fecunidity_threshold', 0.)
        fecundity_proportion = kwargs.get('fecundity_proportion', 1.)
        fecundity_upper_density = kwargs.get('fecundity_upper_density', 1.)
        general_mortality_threshold = kwargs.get('general_mortality_threshold', 1.)
        general_mortality_proportion = kwargs.get('general_mortality_proportion', 1.)
        self.contributingFemales = kwargs.get('contributing_females', None)
        self.contributingMales = kwargs.get('contributing_males', None)
        minimum_viable_pop = kwargs.get('minimum_viable_pop', 0.)  # As a total population (n)
        minimum_viable_area = kwargs.get('minimum_viable_area', 1E6)  # In metres squared

        self.reproducingFemales = np.unique(self.reproducingFemales)
        self.reproducingMales = np.unique(self.reproducingMales)

        self.formalLog['Parameterization'].update({
            'Birth sex allocation': mf_ratio,
            'Live beyond maximum age group': max_age,
            'Inter-cellular migration algorithm': movement_alg,
            'Inter-habitat dispersal distance': max_rate,
            'Maximum migration distance': relocation_distance,
            'Fecundity reduction density threshold': fecundity_threshold,
            'Maximum fecundity reduction density': fecundity_upper_density,
            'Maximum fecundity reduction percentage': fecundity_proportion,
            'General mortality reduction density threshold': general_mortality_threshold,
            'Maximum general mortality reduction percentage': general_mortality_proportion,
            'Minimum viable population per region': minimum_viable_pop,
            'Minimum area for minimum viable population': minimum_viable_area,
            'Reproducing male groups': str(self.reproducingMales),
            'Reproducing female groups': str(self.reproducingFemales)
        })

        distance = max_rate * self.time_step
        time_log = ['Solving time breakdown:']

        # Enclose all in the file context manager
        with self.file as f:
            # Check if required parameters exist for all age groups
            #   Get species
            species_keys = f.keys()
            if len(species_keys) == 0:
                raise PopdynError('Domain is empty.  Add initial populations,'
                                  ' fecundity rates, and mortality rates.')

            for species in species_keys:
                # Get fecundity of all age groups at start time
                try:
                    time_keys = f['%s/%s' % (species, self.start_time)].keys()
                except:
                    raise PopdynError(
                        'The start time {} does not exist in the model domain'.format(self.start_time)
                    )
                # Check that males, females and carrying capacity exist
                for item in ['male', 'female', 'k']:
                    if item not in time_keys:
                        raise PopdynError('%s not found at start time for'
                                          ' species "%s".' % (item, species))
                # Get age group mortalities from each sex at start time
                males = f['%s/%s/%s' % (species, self.start_time,
                                        'male')].keys()
                females = f['%s/%s/%s' % (species, self.start_time,
                                          'female')].keys()
                # Iterate groups and ensure all have fecundity arrays
                self.formalLog['Population'][species] = {}
                self.formalLog['Natality'][species] = {}
                self.formalLog['Mortality'][species] = {}
                for gp in self.ages:
                    # TODO: Make these zeros, and create a UI warning
                    if gp not in time_keys:
                        raise PopdynError('The age group "%s" does not have a'
                                          ' defined fecundity array.' % gp)
                    if gp not in males:
                        raise PopdynError('Males of the age group "%s" do not'
                                          ' have any defined mortality arrays.' %
                                          gp)
                    if gp not in females:
                        raise PopdynError('Females of the age group "%s" do not'
                                          ' have any defined mortality arrays.'
                                          % gp)
                    self.formalLog['Population'][species][gp] = {}
                    self.formalLog['Natality'][species][gp] = {}
                    self.formalLog['Mortality'][species][gp] = {}

                self.formalLog['Population'][species]['NA'] = {}
                self.formalLog['Natality'][species]['NA'] = {}
                self.formalLog['Mortality'][species]['NA'] = {}
                self.formalLog['Habitat'][species] = []

            # Iterate time
            distance_track = 0

            # For profiling
            disperse_time = 0
            get_carry_capacity_time = 0
            tot_pop_time = 0
            birth_calc_time = 0
            collect_time_slice_time = 0
            collect_dynamic_time = 0
            total_age_gnd_gp_time = 0
            fec_from_lookup_time = 0
            mortality_and_propagate_time = 0
            births_and_density_dep_time = 0
            end_time = self.start_time + duration + self.time_step

            simulationRange = range(self.start_time, end_time, self.time_step)

            for timeIndex, time_step in enumerate(simulationRange):

                percentage = abs(time_step - self.start_time) / float(end_time - self.start_time)
                percentage *= 100
                self.scenario_run.percentage = int(abs(percentage))
                self.scenario_run.current_period = time_step
                self.scenario_run.save()

                self.formalLog['Time'].append(time_step)

                distance_track += distance
                # Iterate species and solve
                #   Future: Solve all species together so they may interact
                for species in species_keys:
                    # Get closest carrying capacity in time (backwards only)
                    now = profile_time.time()
                    k = f[self.get_carry_capacity_key(species, time_step,
                                                      file_instance=f)]
                    self.formalLog['Habitat'][species].append(np.sum(k))

                    get_carry_capacity_time += profile_time.time() - now

                    # Collect total population using contributing age groups
                    now = profile_time.time()

                    def contributing_total(inclusion_males, inclusion_females, migrated=False):
                        if inclusion_males is not None:
                            males = np.zeros(shape=self.shape, dtype='float32')
                            for pop_gp in inclusion_males:
                                males += self.total_population(species, time_step, age_gp=pop_gp,
                                                               sex='male', migrated=migrated, file_instance=f)
                        else:
                            males = self.total_population(species, time_step, sex='male', migrated=migrated,
                                                          file_instance=f)
                        if inclusion_females is not None:
                            females = np.zeros(shape=self.shape, dtype='float32')
                            for pop_gp in inclusion_females:
                                females += self.total_population(species, time_step, age_gp=pop_gp,
                                                                 sex='female', migrated=migrated, file_instance=f)
                        else:
                            females = self.total_population(species, time_step, sex='female', migrated=migrated,
                                                            file_instance=f)
                        return males, females

                    contMales, contFemales = contributing_total(self.contributingMales, self.contributingFemales)
                    tot_pop = contMales + contFemales

                    try:
                        self.formalLog['Population'][species]['NA']['Contributing Males'].append(np.sum(contMales))
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Contributing Males'] = [np.sum(contMales)]
                    try:
                        self.formalLog['Population'][species]['NA']['Contributing Females'].append(np.sum(contFemales))
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Contributing Females'] = [np.sum(contFemales)]
                    try:
                        self.formalLog['Population'][species]['NA']['Total Population'].append(np.sum(
                            self.total_population(species, time_step)
                        ))
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Total Population'] = [np.sum(
                            self.total_population(species, time_step)
                        )]
                    try:
                        self.formalLog['Population'][species]['NA']['Total Males'].append(np.sum(
                            self.total_population(species, time_step, sex='male')
                        ))
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Total Males'] = [np.sum(
                            self.total_population(species, time_step, sex='male')
                        )]
                    try:
                        self.formalLog['Population'][species]['NA']['Total Females'].append(np.sum(
                            self.total_population(species, time_step, sex='female')
                        ))
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Total Females'] = [np.sum(
                            self.total_population(species, time_step, sex='female')
                        )]
                    # Average ages
                    try:
                        self.formalLog['Population'][species]['NA']['Average Age'].append(
                            self.average_age(species, time_step)
                        )
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Average Age'] = [
                            self.average_age(species, time_step)
                        ]
                    try:
                        self.formalLog['Population'][species]['NA']['Average Male Age'].append(
                            self.average_age(species, time_step, 'male')
                        )
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Average Male Age'] = [
                            self.average_age(species, time_step, 'male')
                        ]
                    try:
                        self.formalLog['Population'][species]['NA']['Average Female Age'].append(
                            self.average_age(species, time_step, 'female')
                        )
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Average Female Age'] = [
                            self.average_age(species, time_step, 'female')
                        ]

                    tot_pop_time += profile_time.time() - now

                    # Create immigration flux array if rate is > 0
                    now = profile_time.time()
                    population_moved = False
                    if max_rate > 0:
                        # Check if the rate is sub-cell size
                        if distance >= max(self.csx, self.csy) or distance_track >= max(self.csx, self.csy):
                            population_moved = True
                            if distance < max(self.csx, self.csy):
                                use_distance = distance_track
                            else:
                                use_distance = distance
                            if movement_alg == 'dispersive-cancer':
                                self.dispersive_flux(species, time_step, tot_pop, k[:], use_distance, f,
                                                     relocate_distance=relocation_distance)
                            else:
                                raise PopdynError('Unrecognized movement algorithm "{}"'.format(movement_alg))
                            # Total population collected again due to migration
                            contMales, contFemales = contributing_total(self.contributingMales, self.contributingFemales,
                                                                        True)
                            tot_pop = contMales + contFemales
                        if distance_track >= max(self.csx, self.csy):
                            distance_track -= max(self.csx, self.csy)
                    disperse_time += profile_time.time() - now

                    # Calculate the population of reproducing groups
                    now = profile_time.time()
                    reproducingMales, reproducingFemales = contributing_total(self.reproducingMales,
                                                                              self.reproducingFemales, population_moved)

                    try:
                        self.formalLog['Population'][species]['NA']['Reproducing Males'].append(np.sum(reproducingMales))
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Reproducing Males'] = [np.sum(reproducingMales)]
                    try:
                        self.formalLog['Population'][species]['NA']['Reproducing Females'].append(
                            np.sum(reproducingFemales))
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Reproducing Females'] = [np.sum(reproducingFemales)]
                    total_age_gnd_gp_time += profile_time.time() - now

                    # Create arrays for tracking births
                    now = profile_time.time()
                    m = k > tot_pop
                    D = ne.evaluate('where(k>0,tot_pop/k,0)')
                    if fecundity_threshold < fecundity_upper_density:
                        fecundity_prop = ne.evaluate(
                            'where(D>fecundity_threshold,(D-fecundity_threshold)/(fecundity_upper_density-fecundity_threshold),1)')
                        fecundity_prop = ne.evaluate(
                            'where(fecundity_prop<=1,1-(fecundity_prop*fecundity_proportion),1-fecundity_proportion)')
                    elif fecundity_threshold == fecundity_upper_density:
                        fecundity_prop = ne.evaluate('where(D>=fecundity_upper_density,1-fecundity_proportion,1)')
                    else:
                        fecundity_prop = 1.
                    zero_males = np.zeros(shape=self.shape,
                                             dtype='float32')
                    zero_females = np.zeros(shape=self.shape,
                                               dtype='float32')

                    birth_calc_time += profile_time.time() - now

                    # Iterate age groups
                    density_dependent_sum = np.zeros(shape=self.shape, dtype='float32')
                    new_keys = []
                    sumBirths, sumDeaths, sumMaleDeaths, sumFemaleDeaths = 0., 0., 0., 0.
                    sumGpDeaths, sumGpFemaleDeaths, sumGpMaleDeaths = {}, {}, {}
                    modeTemplateDict = {mortName: 0. for mortName in self.mortality_names.values()}
                    modeDeaths, modeFemaleDeaths, modeMaleDeaths = (modeTemplateDict, modeTemplateDict.copy(),
                                                                    modeTemplateDict.copy())
                    for dur, gp in enumerate(self.ages):
                        sumGpDeaths[gp] = 0.
                        sumGpFemaleDeaths[gp] = 0.
                        sumGpMaleDeaths[gp] = 0.
                        max_duration = max([m_dur[1] for m_dur in self.durations])
                        # Get age range in this group
                        dur_from, dur_to = self.durations[dur]

                        # Check if current time step exists in the model and
                        #   get most relevant mortality and fecundity keys
                        now = profile_time.time()
                        fec_key, male_mort_key, fem_mort_key = \
                            self.check_time_slice(species, gp, time_step,
                                                  file_instance=f)
                        collect_time_slice_time += profile_time.time() - now
                        now = profile_time.time()
                        fec = self.collect_dynamic(fec_key, file_instance=f)
                        collect_dynamic_time += profile_time.time() - now

                        now = profile_time.time()
                        if 'lookup' in f[fec_key].attrs.keys():
                            # Calculate gender ratio
                            g_rat = ne.evaluate(
                                'where(reproducingMales>0,'
                                'reproducingFemales/reproducingMales,1E38)'
                            )  # Ratio is infinitely large if males == 0, so use high number

                            # Scale based on gender ratio lookup table
                            g_rat = self.collect_from_lookup(fec_key, g_rat, f)

                            fec = ne.evaluate('fec*g_rat*fecundity_prop')
                        else:
                            fec *= fecundity_prop

                        try:
                            self.formalLog['Natality'][species][gp]['Minimum fecundity'].append(np.min(fec))
                        except KeyError:
                            self.formalLog['Natality'][species][gp]['Minimum fecundity'] = [np.min(fec)]
                        try:
                            self.formalLog['Natality'][species][gp]['Maximum fecundity'].append(np.max(fec))
                        except KeyError:
                            self.formalLog['Natality'][species][gp]['Maximum fecundity'] = [np.max(fec)]
                        try:
                            self.formalLog['Natality'][species][gp]['Mean fecundity'].append(np.mean(fec))
                        except KeyError:
                            self.formalLog['Natality'][species][gp]['Mean fecundity'] = [np.mean(fec)]

                        fec_from_lookup_time += profile_time.time() - now

                        # Calculate total population of males and females in gp
                        now = profile_time.time()
                        total_females = self.total_population(
                            species, time_step, 'female', gp, file_instance=f
                        )
                        total_males = self.total_population(
                            species, time_step, 'male', gp, file_instance=f
                        )

                        # Calculate births
                        births = ne.evaluate(
                            'where(m,total_females*fec,0)'
                        )

                        sumBirths += births.sum()

                        sumFemales, sumMales = np.sum(total_females), np.sum(total_males)

                        try:
                            self.formalLog['Population'][species][gp]['Males'].append(sumMales)
                        except KeyError:
                            self.formalLog['Population'][species][gp]['Males'] = [sumMales]
                        try:
                            self.formalLog['Population'][species][gp]['Females'].append(sumFemales)
                        except KeyError:
                            self.formalLog['Population'][species][gp]['Females'] = [sumFemales]
                        try:
                            self.formalLog['Population'][species][gp]['Total'].append(sumMales + sumFemales)
                        except KeyError:
                            self.formalLog['Population'][species][gp]['Total'] = [sumMales + sumFemales]

                        # Distribute births among males/females depending on
                        #   relationship
                        if mf_ratio == 'random':
                            birth_proportion = np.random.random(self.shape)
                            male_births = ne.evaluate(
                                'births*birth_proportion'
                            )
                            female_births = ne.evaluate(
                                'births*(1-birth_proportion)'
                            )
                            del birth_proportion
                        else:
                            male_births = births * mf_ratio
                            female_births = births * (1 - mf_ratio)

                        zero_males += male_births
                        zero_females += female_births

                        sumFemaleBirths, sumMaleBirths = np.sum(female_births), np.sum(male_births)
                        try:
                            self.formalLog['Natality'][species][gp]['Male offspring'].append(sumMaleBirths)
                        except KeyError:
                            self.formalLog['Natality'][species][gp]['Male offspring'] = [sumMaleBirths]
                        try:
                            self.formalLog['Natality'][species][gp]['Female offspring'].append(sumFemaleBirths)
                        except KeyError:
                            self.formalLog['Natality'][species][gp]['Female offspring'] = [sumFemaleBirths]
                        try:
                            self.formalLog['Natality'][species][gp]['Total offspring'].append(
                                sumFemaleBirths + sumMaleBirths)
                        except KeyError:
                            self.formalLog['Natality'][species][gp]['Total offspring'] = [
                                sumFemaleBirths + sumMaleBirths]

                        birth_calc_time += profile_time.time() - now

                        # Mortality
                        total_sexes = {'female': total_females, 'male': total_males}

                        # Iterate sexes for mortality and age propagation
                        for m_key, sex in [(male_mort_key, 'male'),
                                           (fem_mort_key, 'female')]:
                            # Collect mortality rates
                            mortality_rates = {}
                            for mortality_name in f[m_key].keys():
                                ds_key = '%s/%s' % (m_key, mortality_name)
                                prop = f[ds_key].attrs['proportion']
                                now = profile_time.time()
                                if prop:
                                    # Directly use rate
                                    mortality_rates[mortality_name] = self.collect_dynamic(ds_key, file_instance=f)
                                else:
                                    # Calculate the rate if mortality is a number of individuals
                                    ds = self.collect_dynamic(ds_key, file_instance=f)
                                    ts = total_sexes[sex]
                                    mortality_rates[mortality_name] = ne.evaluate('ds/ts')

                                try:
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} minimum'.format(sex, self.mortality_names[mortality_name])][
                                        timeIndex] = np.min(mortality_rates[mortality_name])
                                except KeyError:
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} minimum'.format(sex, self.mortality_names[mortality_name])] = \
                                        np.zeros(shape=len(simulationRange), dtype='float32')
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} minimum'.format(sex, self.mortality_names[mortality_name])][
                                        timeIndex] = np.min(mortality_rates[mortality_name])
                                try:
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} maximum'.format(sex, self.mortality_names[mortality_name])][
                                        timeIndex] = np.max(mortality_rates[mortality_name])
                                except KeyError:
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} maximum'.format(sex, self.mortality_names[mortality_name])] = \
                                        np.zeros(shape=len(simulationRange), dtype='float32')
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} maximum'.format(sex, self.mortality_names[mortality_name])][
                                        timeIndex] = np.max(mortality_rates[mortality_name])
                                try:
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} mean'.format(sex, self.mortality_names[mortality_name])][
                                        timeIndex] = np.mean(mortality_rates[mortality_name])
                                except KeyError:
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} mean'.format(sex, self.mortality_names[mortality_name])] = \
                                        np.zeros(shape=len(simulationRange), dtype='float32')
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} mean'.format(sex, self.mortality_names[mortality_name])][
                                        timeIndex] = np.mean(mortality_rates[mortality_name])

                                collect_dynamic_time += profile_time.time() - now
                            # Iterate ages, apply mortality, and propagate
                            now = profile_time.time()
                            totalMortality = ne.evaluate('{}'.format('+'.join(mortality_rates.keys())),
                                                         local_dict=mortality_rates)
                            # Apply mortality in aggregate
                            for mortality_name, mort_rate in mortality_rates.iteritems():
                                mortality_rates[mortality_name] = ne.evaluate(
                                    'where(totalMortality>0,mort_rate/totalMortality,0)'
                                )
                            for age in f['%s/%s/%s' % (species, time_step, sex)].keys():
                                try:
                                    age = int(age)
                                except:
                                    continue
                                # Make sure age is in the group or above the max (if not max_age)
                                if ((age < dur_from or age > dur_to) and
                                        (age <= max_duration or dur_to != max_duration)):
                                    continue

                                # Keys
                                key = '%s/%s/%s/%s' % (species, time_step,
                                                       sex, age)
                                new_key = '%s/%s/%s/%s' % (species, time_step + self.time_step,
                                                           sex, age + self.time_step)

                                # Grab redistributed population if it can move
                                if population_moved:
                                    _pop = f['migrations/%s/%s' % (sex, age)][:]
                                else:
                                    _pop = self.get_dataset(key, 'float32',
                                                            file_instance=f)[:]

                                # Allow or disallow the max age to live
                                if age == max_duration:
                                    maxAgeMort = False
                                    if max_age:
                                        # This whole age population has to die
                                        maxAgeMort = True
                                        self._create_dataset('%s/%s/%s/mortality/%s/Old Age' %
                                                             (species, time_step, sex, age), _pop, f)
                                        maxAgeDeaths = np.sum(_pop)
                                    else:
                                        maxAgeDeaths = 0
                                    sumDeaths += maxAgeDeaths
                                    if sex == 'male':
                                        sumGpMaleDeaths[gp] += maxAgeDeaths
                                        sumMaleDeaths += maxAgeDeaths
                                        try:
                                            modeMaleDeaths['Old Age'] += maxAgeDeaths
                                        except KeyError:
                                            modeMaleDeaths['Old Age'] = maxAgeDeaths
                                    else:
                                        try:
                                            modeFemaleDeaths['Old Age'] += maxAgeDeaths
                                        except KeyError:
                                            modeFemaleDeaths['Old Age'] = maxAgeDeaths
                                        sumGpFemaleDeaths[gp] += maxAgeDeaths
                                        sumFemaleDeaths += maxAgeDeaths
                                    sumGpDeaths[gp] += maxAgeDeaths
                                    try:
                                        modeDeaths['Old Age'] += maxAgeDeaths
                                    except KeyError:
                                        modeDeaths['Old Age'] = maxAgeDeaths
                                    try:
                                        self.formalLog['Mortality'][species][gp][
                                            '{} old age deaths'.format(sex)][timeIndex] = maxAgeDeaths
                                    except KeyError:
                                        self.formalLog['Mortality'][species][gp][
                                            '{} old age deaths'.format(sex)] = \
                                            np.zeros(shape=len(simulationRange), dtype='float32')
                                        self.formalLog['Mortality'][species][gp][
                                            '{} old age deaths'.format(sex)][timeIndex] = maxAgeDeaths
                                    if maxAgeMort:
                                        continue

                                deaths = ne.evaluate('totalMortality*_pop')
                                # Correct mortalities to avoid negatives and record
                                overdead = deaths > _pop
                                deaths[overdead] = _pop[overdead]
                                _pop -= deaths
                                for mortality_name, mort_rate in mortality_rates.iteritems():
                                    _deaths = ne.evaluate('mort_rate*deaths')
                                    self._create_dataset('%s/%s/%s/mortality/%s/%s' %
                                                         (species, time_step, sex, age, mortality_name), _deaths, f)

                                    mortDeaths = np.sum(_deaths)

                                    sumDeaths += mortDeaths
                                    if sex == 'male':
                                        sumGpMaleDeaths[gp] += mortDeaths
                                        sumMaleDeaths += mortDeaths
                                        modeMaleDeaths[self.mortality_names[mortality_name]] += mortDeaths
                                    else:
                                        sumGpFemaleDeaths[gp] += mortDeaths
                                        sumFemaleDeaths += mortDeaths
                                        modeFemaleDeaths[self.mortality_names[mortality_name]] += mortDeaths
                                    sumGpDeaths[gp] += mortDeaths
                                    modeDeaths[self.mortality_names[mortality_name]] += mortDeaths

                                    try:
                                        self.formalLog['Mortality'][species][gp][
                                            '{} age {} {} deaths'.format(
                                                sex, age, self.mortality_names[mortality_name])][timeIndex] = mortDeaths
                                    except KeyError:
                                        self.formalLog['Mortality'][species][gp][
                                            '{} age {} {} deaths'.format(
                                                sex, age, self.mortality_names[mortality_name])] = \
                                            np.zeros(shape=len(simulationRange), dtype='float32')
                                    self.formalLog['Mortality'][species][gp][
                                        '{} age {} {} deaths'.format(
                                            sex, age, self.mortality_names[mortality_name])][timeIndex] = mortDeaths

                                # If a population already exists in the
                                #   new time slot, it will be added
                                new_keys.append(new_key)
                                self._create_dataset(new_key, _pop,
                                                     'add_no_neg', file_instance=f)
                                density_dependent_sum += f[new_key]
                            mortality_and_propagate_time += profile_time.time() - now

                    try:
                        self.formalLog['Natality'][species]['NA']['Total new offspring'].append(sumBirths)
                    except KeyError:
                        self.formalLog['Natality'][species]['NA']['Total new offspring'] = [sumBirths]
                    now = profile_time.time()
                    # Add births as zero age in next time step after truncating
                    for births, sex in [(zero_females, 'female'),
                                        (zero_males, 'male')]:
                        zero_key = '%s/%s/%s/%s' % (species, time_step +
                                                    self.time_step, sex, 0)
                        # Add population in time slot if it exists (immigration/emigration)
                        if np.sum(births) > 0:
                            new_keys.append(zero_key)
                            self._create_dataset(zero_key, births, 'add_no_neg',
                                                 file_instance=f)
                            density_dependent_sum += f[zero_key]
                    try:
                        del f['migrations']
                    except:
                        pass

                    # Use density-dependent total to apply density-dependent mortality
                    kM = (k[:] > 0) & (density_dependent_sum > 0)
                    kThresh = ne.evaluate('general_mortality_threshold*k')
                    if general_mortality_threshold < 1.:
                        deaths = ne.evaluate(
                            'where(kM,'
                            '(((density_dependent_sum/k)-general_mortality_threshold)/(1-general_mortality_threshold))*general_mortality_proportion'
                            ',0)'
                        )
                        deaths[deaths > general_mortality_proportion] = general_mortality_proportion
                        deaths = ne.evaluate(
                            'where(kM,1-(((density_dependent_sum-kThresh)*deaths)/density_dependent_sum),0)'
                        )
                    else:
                        deaths = ne.evaluate('where(kM,'
                                             '1-(((density_dependent_sum-k)*general_mortality_proportion)/density_dependent_sum),'
                                             '0)')
                    deaths[kM & (density_dependent_sum < kThresh)] = 1

                    # Calculate minimum viable population threshold
                    if minimum_viable_pop > 0:
                        deaths *= self.minimum_viable_population(density_dependent_sum,
                                                                 minimum_viable_pop,
                                                                 minimum_viable_area)

                    for new_key in new_keys:
                        ds = f[new_key][:]
                        _pop = ne.evaluate('ds*deaths')
                        death = ne.evaluate('ds-_pop')
                        _sex = new_key.split('/')[2]
                        _age = int(new_key.split('/')[3]) - self.time_step
                        if _age >= 0:
                            gp = self.group_from_age(_age)
                            self._create_dataset('%s/%s/%s/mortality/%s/Density Dependent Mortality' %
                                                 (species, time_step, _sex,
                                                  _age), death, f)
                            dDdeath = np.sum(death)
                            sumDeaths += dDdeath
                            if _sex == 'male':
                                sumGpMaleDeaths[gp] += dDdeath
                                sumMaleDeaths += dDdeath
                                printSex = 'Male'
                                try:
                                    modeMaleDeaths['Density Dependent'] += dDdeath
                                except KeyError:
                                    modeMaleDeaths['Density Dependent'] = dDdeath
                            else:
                                sumGpFemaleDeaths[gp] += dDdeath
                                sumFemaleDeaths += dDdeath
                                printSex = 'Female'
                                try:
                                    modeFemaleDeaths['Density Dependent'] += dDdeath
                                except KeyError:
                                    modeFemaleDeaths['Density Dependent'] = dDdeath
                            sumGpDeaths[gp] += dDdeath
                            try:
                                modeDeaths['Density Dependent'] += dDdeath
                            except KeyError:
                                modeDeaths['Density Dependent'] = dDdeath
                            try:
                                self.formalLog['Mortality'][species][gp][
                                    '{} density dependent mortality deaths'.format(printSex)][timeIndex] = dDdeath
                            except KeyError:
                                self.formalLog['Mortality'][species][gp][
                                    '{} density dependent mortality deaths'.format(printSex)] = \
                                    np.zeros(shape=len(simulationRange), dtype='float32')
                                self.formalLog['Mortality'][species][gp][
                                    '{} density dependent mortality deaths'.format(printSex)][timeIndex] = dDdeath

                        f[new_key][:] = _pop

                    try:
                        self.formalLog['Mortality'][species]['NA']['All deaths'].append(sumDeaths)
                    except KeyError:
                        self.formalLog['Mortality'][species]['NA']['All deaths'] = [sumDeaths]
                    try:
                        self.formalLog['Mortality'][species]['NA']['Total female deaths'].append(sumFemaleDeaths)
                    except KeyError:
                        self.formalLog['Mortality'][species]['NA']['Total female deaths'] = [sumFemaleDeaths]
                    try:
                        self.formalLog['Mortality'][species]['NA']['Total male deaths'].append(sumMaleDeaths)
                    except KeyError:
                        self.formalLog['Mortality'][species]['NA']['Total male deaths'] = [sumMaleDeaths]
                    for mortMode in modeDeaths.keys():
                        try:
                            self.formalLog['Mortality'][species]['NA']['Total deaths from {}'.format(mortMode)][
                                timeIndex] = modeDeaths[mortMode]
                        except KeyError:
                            self.formalLog['Mortality'][species]['NA']['Total deaths from {}'.format(mortMode)] = \
                                np.zeros(shape=len(simulationRange), dtype='float32')
                            self.formalLog['Mortality'][species]['NA']['Total deaths from {}'.format(mortMode)][
                                timeIndex] = modeDeaths[mortMode]
                        try:
                            self.formalLog['Mortality'][species]['NA']['Male deaths from {}'.format(mortMode)][
                                timeIndex] = modeMaleDeaths[mortMode]
                        except KeyError:
                            self.formalLog['Mortality'][species]['NA']['Male deaths from {}'.format(mortMode)] = \
                                np.zeros(shape=len(simulationRange), dtype='float32')
                            self.formalLog['Mortality'][species]['NA']['Male deaths from {}'.format(mortMode)][
                                timeIndex] = modeMaleDeaths[mortMode]
                        try:
                            self.formalLog['Mortality'][species]['NA']['Female deaths from {}'.format(mortMode)][
                                timeIndex] = modeFemaleDeaths[mortMode]
                        except KeyError:
                            self.formalLog['Mortality'][species]['NA']['Female deaths from {}'.format(mortMode)] = \
                                np.zeros(shape=len(simulationRange), dtype='float32')
                            self.formalLog['Mortality'][species]['NA']['Female deaths from {}'.format(mortMode)][
                                timeIndex] = modeFemaleDeaths[mortMode]
                    for gp in self.ages:
                        try:
                            self.formalLog['Mortality'][species][gp]['Total deaths'].append(sumGpDeaths[gp])
                        except KeyError:
                            self.formalLog['Mortality'][species][gp]['Total deaths'] = [sumGpDeaths[gp]]
                        try:
                            self.formalLog['Mortality'][species][gp]['Male deaths'].append(sumGpMaleDeaths[gp])
                        except KeyError:
                            self.formalLog['Mortality'][species][gp]['Male deaths'] = [sumGpMaleDeaths[gp]]
                        try:
                            self.formalLog['Mortality'][species][gp]['Female deaths'].append(sumGpFemaleDeaths[gp])
                        except KeyError:
                            self.formalLog['Mortality'][species][gp]['Female deaths'] = [sumGpFemaleDeaths[gp]]

                    births_and_density_dep_time += profile_time.time() - now

        timeLogTimes = ['Intercellular dispersal,{}'.format(round(disperse_time, 4)),
                        'Query habitat,{}'.format(round(get_carry_capacity_time, 4)),
                        'Collect population sums,{}'.format(round(tot_pop_time, 4)),
                        'Compute natality,{}'.format(round(birth_calc_time, 4)),
                        'Query data for time step,{}'.format(round(collect_time_slice_time, 4)),
                        'Collect dynamic data (random or other),{}'.format(round(collect_dynamic_time, 4)),
                        'Total population of age group/sex calculation,{}'.format(round(total_age_gnd_gp_time, 4)),
                        'Calculate fecundity using ratios,{}'.format(round(fec_from_lookup_time, 4)),
                        'Mortality calculation and cohort graduation,{}'.format(round(mortality_and_propagate_time, 4)),
                        'Propagate offspring and apply density-dependent mortality,{}'.format(
                            round(births_and_density_dep_time, 4))]

        time_log += timeLogTimes
        self.formalLog['Solver'] += timeLogTimes

        return self.formalLog
