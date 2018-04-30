"""
Population dynamics numerical solvers

Devin Cairns, 2018
"""
from popdyn import *
from logger import time_this
import dask.array as da


class SolverError(Exception):
    pass


def da_zeros(shape, chunks):
    """Create a dask array of broadcasted zeros"""
    return da.from_array(np.broadcast_to(np.float32(0), shape), chunks)


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
                    pop_to_groups = True
                    if sex_population_key is None:
                        if population_ds is None:
                            # No populations at species or sex level
                            pop_to_groups = False
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
                        if pop_to_groups:
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
            self.current_time = time
            # Hierarchically traverse [species --> sex --> groups --> age] and solve children populations
            # Each are solved independently, deriving relationships from baseline data
            # -------------------------------------------------------------------------------------------
            # Create dicts to store IO to avoid repetitive reads in the dask graph
            all_species = self.D.species.keys()
            self.population_arrays = {sp: {} for sp in all_species}
            self.carrying_capacity_arrays = {sp: {} for sp in all_species}

            # Collect totals for entire species-based calculations, and dicts to store all output and offspring
            self.totals(all_species, time)
            self.age_zero_population = {}
            output = {}

            for species in all_species:
                # Collect sex keys
                sex_keys = self.D.species[species].keys()
                # Sex may not exist and will be None, but if one of males or females are included,
                # do not propagate Nonetypes
                if any([fm in sex_keys for fm in ['male', 'female']]):
                    sex_keys = [key for key in sex_keys if key is not None]

                # All births are recorded from each sex are added to the age 0 slot
                self.age_zero_population[species] = {key: da_zeros(self.D.shape, self.D.chunks)
                                                     for key in sex_keys}

                # Iterate sexes and groups and calculate each individually
                for sex in sex_keys:
                    for group in self.D.species[species][sex].keys():  # Group - may not exist and will be None
                        output.update(self.propagate(species, sex, group, time))

                # Update the output with the age 0 populations
                for _sex in self.age_zero_population[species].keys():
                    zero_group = self.D.group_from_age(species, _sex, 0)
                    if zero_group is None:
                        age = None
                    else:
                        age = 0
                    key = '{}/{}/{}/{}/{}'.format(species, _sex, zero_group, time, age)
                    try:
                        # Add them together if the key exists in the output
                        output[key] = output[key] + self.age_zero_population[species][_sex]
                    except KeyError:
                        output[key] = self.age_zero_population[species][_sex]

            # Compute this time slice and write to the domain
            self.D.domain_compute(output)

    @time_this
    def totals(self, all_species, time):
        """
        Collect all population and carrying capacity datasets, and calculate species totals if necessary.
        Dynamic calculations of carrying capacity are made by calling collect_parameter, which may also
        update the carrying_capacity or population array dicts.
        :param list all_species: All species in the domain
        :param int time: Time slice to collect totals
        :return: None
        """
        for species in all_species:
            # Need to collect the total population for fecundity and density if not otherwise specified
            if self.total_density:
                # Population
                population = self.D.all_population(species, time - 1)
                self.population_arrays[species]['total'] = self.population_total(species, population)

                # Carrying Capacity
                # -----------------------------------
                # Collect all carrying capacity keys
                cc = self.D.all_carrying_capacity(species, time)
                self.carrying_capacity_arrays[species]['total'] =  self.carrying_capacity_total(species, cc)

            # Calculate species-wide male and female populations that reproduce
            # Note: Species that do not contribute to density contribute to the reproduction totals
            # -----------------------------------------------------------------
            # Males
            self.population_arrays[species]['Reproducing Males'] = self.population_total(
                species, self.D.all_population(species, time - 1, 'male'), False, True
            )

            # Females
            self.population_arrays[species]['Reproducing Females'] = self.population_total(
                species, self.D.all_population(species, time - 1, 'female'), False, True
            )

            # Calculate Density
            self.population_arrays[species]['Female to Male Ratio'] = da.where(
                self.population_arrays[species]['Reproducing Males'] > 0,
                self.population_arrays[species]['Reproducing Females'] /
                self.population_arrays[species]['Reproducing Males'],
                np.inf
            )

    @time_this
    def collect_parameter(self, param, data):
        """
        Collect data associated with a Fecundity, CarryingCapacity, or Mortality instance. If a species is linked,
        data will be None and density of the linked species will be calculated.
        :param param: CarryingCapacity, Fecundity, or Mortality instance
        :param data: array data or None
        :return: A dask array with the collected data
        """
        kwargs = {}

        # Collect the total species population from the previous time step
        if data is None:
            # There must be a species attached to the parameter instance if there are no data
            # Density derived from other species- directly applied to the lookup table kwargs
            kwargs.update(self.interspecies_density(param))
        else:
            data = da.from_array(data, data.chunks)

        # Include random parameters (except not if carrying capacity is another species)
        if getattr(param, 'random_method', False) and not (isinstance(param, CarryingCapacity) and data is None):
            kwargs.update({'random_method': param.random_method, 'random_args': param.random_args})

        return dynamic.collect(data, **kwargs)  # A dask array

    @time_this
    def interspecies_density(self, param):
        """
        Returns the kwargs of density and lookup for a species-dependent parameter.
        Because interspecies dependencies may be nested or circular, the carrying capacity of all connected
        species is calculated herein and added to the self.carrying_capacity_arrays object. Future calls of any
        of the connected species will have data available, avoiding redundant calculations.
        :param species:
        :param param:
        :return: dict of lookup data and table to update kwargs in dynamic call
        """

        # Check for circular references
        # ------------------------------------------------------
        def next_cc(param):
            tree.append(param.species.name_key)
            cc = self.D.all_carrying_capacity(
                param.species.name_key, self.current_time, param.species.sex, param.species.group_key
            )
            for _cc in cc:
                if _cc[0] is not None:
                    if _cc[0].species.name_key in tree:
                        raise SolverError('Britches must be held until circular references are allowed')
                    next_cc(_cc[0])

        tree = []
        next_cc(param)
        # ------------------------------------------------------

        # Collect the population of the interdependent species
        # Collect the total baseline (time - 1) population of the species
        population = self.D.all_population(param.species.name_key, self.current_time - 1,
                                           param.species.sex, param.species.group_key)
        a = self.population_total(param.species.name_key, population)

        carrying_capacity = self.D.all_carrying_capacity(
            param.species.name_key, self.current_time, param.species.sex, param.species.group_key
        )

        c = self.carrying_capacity_total(param.species.name_key, carrying_capacity)

        # the kwargs for the lookup table related to the original query are returned along with the density
        return {'lookup_data': da.where(c > 0, a / c, np.inf), 'lookup_table': param.species_table}

    @time_this
    def population_total(self, species, datasets, honor_density=True, honor_reproduction=False):
        """
        Calculate the total population for a species with the provided datasets, usually a result of a
        Domain.all_population call. This updates self.population_arrays to avoid repetitive IO
        :param species:
        :param datasets:
        :return:
        """
        pop_arrays = []

        for key, d in datasets.items():
            instance = self.D.instance_from_key(key)

            if honor_density and not instance.contributes_to_density:
                continue
            if honor_reproduction:
                # Collect fecundity to see if it is present
                species_key, sex, group_key, time = self.D.deconstruct_key(key)[:4]

                if len(self.D.get_fecundity(species_key, time, sex, group_key)) == 0:
                    continue

            try:
                pop_arrays.append(self.population_arrays[species][key])
            except KeyError:
                da_array = da.from_array(d, d.chunks)
                pop_arrays.append(da_array)
                self.population_arrays[species][key] = da_array

        if len(pop_arrays) > 0:
            return da.dstack(pop_arrays).sum(axis=-1)
        else:
            return da_zeros(self.D.shape, self.D.chunks)

    @time_this
    def carrying_capacity_total(self, species, datasets):
        """
        Calculate the total carrying capacity for a species with the provided datasets, usually a result of a
        Domain.get_carrying_capacity call. This updates self.carrying_capacity_arrays to avoid repetitive IO
        :param datasets: list of tuples with instance - data pairs
        :param list dependent: Tracks the species used in species-dependent carrying capacity calculations
        :return: dask array of total carrying capacity
        """
        cc_coeff = []
        cc_arrays = []
        for cc in datasets:
            # Check if this has already been computed. If any interspecies data are included, they will
            #  all be computed at the first occurrence and added to the arrays.
            try:
                if cc[1] is None:
                    cc_coeff.append(self.carrying_capacity_arrays[species][cc[0]])
                else:
                    cc_arrays.append(self.carrying_capacity_arrays[species][cc[0]])
            except KeyError:
                # The dataset must be calculated or added to the arrays dict
                ds = self.collect_parameter(cc[0], cc[1])
                self.carrying_capacity_arrays[species][cc[0]] = ds

                if cc[1] is None:
                    cc_coeff.append(ds)
                else:
                    cc_arrays.append(ds)

        if len(cc_coeff) > 0:
            coeff = da.dstack(cc_coeff).sum(axis=-1)
        else:
            coeff = 1.
        if len(cc_arrays) == 0:
            arrays = da_zeros(self.D.shape, self.D.chunks)
        else:
            arrays = da.dstack(cc_arrays).sum(axis=-1)
        arrays *= coeff
        return da.where(arrays > 0, arrays, 0)

    @time_this
    def propagate(self, species, sex, group, time):
        """
        Factory for propagating solved populations and recording intermediates

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
        # Collect some meta on the species
        # -----------------------------------------------------------------
        species_instance = self.D.species[species][sex][group]
        ages = species_instance.age_range
        max_age = self.D.discrete_ages(species, sex)[-1]
        # Output datasets are stored in "flux" and "params" directories,
        # which are for populations changes and model parameters, respectively
        flux_prefix = '{}/{}/{}/{}/flux'.format(species, sex, group, time)
        param_prefix = '{}/{}/{}/{}/params'.format(species, sex, group, time)

        # Ensure there are populations to propagate
        # -----------------------------------------------------------------
        if all([self.D.get_population(species, time - 1, sex, group, age) is None for age in ages]):
            return {}

        # Collect parameters from the current time step.
        # All dynamic modifications are also applied in this step
        # -----------------------------------------------------------------
        params = self.calculate_parameters(species, sex, group, time)

        # Collect mortality names
        mort_types = [mort_type[0].name for mort_type in self.D.get_mortality(species, time, sex, group)]

        # All outputs are written at once, so create a dict to do so (pre-populated with mortality types as zeros)
        output = {'{}/mortality/{}'.format(flux_prefix, mort_type):
                  da_zeros(self.D.shape, self.D.chunks) for mort_type in mort_types}

        # Calculate births using the effective fecundity
        if sex is not None:
            # The total population of this group is used, as fecundity will be 0
            # within groups that do not reproduce
            output['{}/fecundity/{}'.format(param_prefix, 'Fecundity')] = params['Fecundity']
            births = params['Population'] * params['Fecundity']
            male_births = births * params['Birth Ratio']
            female_births = births - male_births

            # Add new offspring to males or females
            for _sex in self.age_zero_population[species].keys():
                if _sex == 'female':
                    # Add female portion of births
                    self.age_zero_population[species][_sex] += female_births
                    output['{}/offspring/{}'.format(flux_prefix, 'Female Offspring')] = female_births
                else:
                    self.age_zero_population[species][_sex] += male_births
                    output['{}/offspring/{}'.format(flux_prefix, 'Male Offspring')] = male_births

        for age in ages:
            # Collect the population of the age from the previous time step
            population = self.D.get_population(species, time - 1, sex, group, age)
            # Use population total to avoid re-read from disk
            population = self.population_total(species, {population: self.D[population]}, False)

            # Mortaliy is applied on each age, as they need to be removed from the population
            for mort_type in mort_types:
                output['{}/mortality/{}'.format(param_prefix, mort_type)] = params[mort_type]
                output['{}/mortality/{}'.format(flux_prefix, mort_type)] = params[mort_type] * population
            # Reduce the population by the mortality
            for mort_type in mort_types:
                population -= output['{}/mortality/{}'.format(flux_prefix, mort_type)]

            # Apply old age mortality if necessary to avoid unnecessary dispersal calculations
            if not species_instance.live_past_max and max_age is not None and age == max_age:
                output['{}/mortality/{}'.format(flux_prefix, 'Old Age')] = population

            else:
                # Apply dispersal to remaining population

                # TODO: Do children receive dispersal of any parent species classes?

                for dispersal_method, args in species_instance.dispersal:
                    args = args + (self.D.csx, self.D.csy)
                    population = dispersal.apply(population,
                                                 self.population_arrays[species]['total'],
                                                 self.carrying_capacity_arrays[species]['total'],
                                                 dispersal_method, args)
                # Apply density-dependent mortality
                output['{}/mortality/{}'.format(flux_prefix, 'Density Dependent')] = da.where(
                    population > params['Carrying Capacity'], population - params['Carrying Capacity'], 0
                )
                population -= output['{}/mortality/{}'.format(flux_prefix, 'Density Dependent')]

                # Propagate age by one increment and save to the current time step.
                # If the input age is None, the population will simply propagate
                if age is not None:
                    new_age = age + 1
                else:
                    new_age = age
                new_group = self.D.group_from_age(species, sex, new_age)
                if new_group is None:
                    # This age does not exist in the domain
                    new_age = None
                key = '{}/{}/{}/{}/{}'.format(species, sex, new_group, time, new_age)
                # In case a duplicate key exists, addition is first attempted
                try:
                    output[key] += population
                except KeyError:
                    output[key] = population

        # TODO: Deal with populations that already exist in the domain at a given time

        # Add carrying capacity and fecundity to datasets
        output['{}/carrying capacity/{}'.format(param_prefix, 'Carrying Capacity')] = params['Carrying Capacity']

        # Return output so that it may be included in the final compute call
        return output

    @time_this
    def calculate_parameters(self, species, sex, group, time):
        """Collect all required parameters at a time step"""
        parameters = {}

        # Mortality
        # ---------------------
        # Collect Mortality instances and/or data from the domain at the current time step
        mortality = self.D.get_mortality(species, time, sex, group)
        if len(mortality) == 0:
            # Defaults to 0
            parameters['Mortality'] = da_zeros(self.D.shape, self.D.chunks)
        else:
            for instance, data in mortality:
                # Mortality types remain separated by name in order to track numbers associated with types,
                # and cannot be less than 0
                parameters[instance.name] = self.collect_parameter(instance, data)
                parameters[instance.name] = da.where(parameters[instance.name] < 0, 0, parameters[instance.name])

        # Carrying Capacity -
        # ---------------------
        # Collect CarryingCapacity instances and/or data from the domain at the current time step
        if not self.total_density:
            carrying_capacity = self.D.get_carrying_capacity(species, time, sex, group)
            parameters['Carrying Capacity'] = self.carrying_capacity_total(species, carrying_capacity)

        else:
            parameters['Carrying Capacity'] = self.carrying_capacity_arrays[species]['total']

        # Total Population from previous time step
        # ---------------------
        population = self.D.all_population(species, time - 1, sex, group)
        parameters['Population'] = self.population_total(species, population, False)  # All, regardless

        # Density calculation only includes contributing species instances
        if not self.total_density:
            population = self.population_total(species, population)
        else:
            population = self.population_arrays[species]['total']

        # Calculate density
        # ---------------------
        # This will reflect the child density if domain.total_density is False
        parameters['Density'] = da.where(parameters['Carrying Capacity'] > 0,
                                         population / parameters['Carrying Capacity'],
                                         np.inf)

        # Collect fecundity parameters
        # These may be collected for any species, whether it be male or female. If fecundity exists for
        # both males and females, the female rate will override the male rate.
        # ---------------------
        # Collect the fecundity instances - HDF5[or None] pairs
        fecundity = self.D.get_fecundity(species, time, sex, group)

        # Calculate the fecundity values
        if len(fecundity) == 0:
            # Defaults to 0
            parameters['Fecundity'] = da_zeros(self.D.shape, self.D.chunks)
        else:
            # Fecundity is collected then aggregated
            fec_arrays = []
            for instance, data in fecundity:
                # Filtered through its density lookup table
                fec = self.collect_parameter(instance, data)
                fec *= dynamic.collect(
                    None, lookup_data=self.population_arrays[species]['Female to Male Ratio'],
                    lookup_table=instance.fecundity_lookup
                )
                # Linearly adjusted based on threshold values
                density_fecundity_threshold = getattr(instance, 'density_fecundity_threshold')
                density_fecundity_max = getattr(instance, 'density_fecundity_max')
                fecundity_reduction_rate = getattr(instance, 'fecundity_reduction_rate')
                if density_fecundity_threshold < density_fecundity_max:
                    fec_mod = da.where(
                        parameters['Density'] > density_fecundity_threshold,
                        (parameters['Density'] - density_fecundity_threshold) /
                        (density_fecundity_max - density_fecundity_threshold),
                        1.
                    )
                    fec_mod = da.where(
                        fec_mod <= 1, 1 - (fec_mod * fecundity_reduction_rate), 1 - fecundity_reduction_rate
                    )
                elif density_fecundity_threshold == density_fecundity_max:
                    fec_mod = da.where(
                        parameters['Density'] >= density_fecundity_max, 1. - fecundity_reduction_rate, 1.
                    )
                else:
                    fec_mod = 1.
                fec *= fec_mod
                # Add to stack
                fec_arrays.append(fec)
            parameters['Fecundity'] = da.dstack(fec_arrays).sum(axis=-1)

        if getattr(species, 'fecundity_random', False):
            parameters['Fecundity'] = dynamic.collect(
                parameters['Fecundity'], random_method=species.fecundity_random,
                random_args=species.fecundity_random_args
            )

        parameters['Fecundity'] = da.where(parameters['Fecundity'] > 0, parameters['Fecundity'], 0)

        birth_ratio = getattr(species, 'birth_ratio', 'random')

        if birth_ratio == 'random':
            parameters['Birth Ratio'] = np.random.random()
        else:
            parameters['Birth Ratio'] = birth_ratio

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
