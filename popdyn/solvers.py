"""
Population dynamics numerical solvers

Devin Cairns, 2018
"""
from popdyn import *
from logger import time_this
import dask.array as da


class SolverError(Exception):
    pass


def error_check(domain):
    """
    Ensure there are no data conflicts within a domain prior to calling a solver. Current checks include:

    - Make sure at least one Species is included
    - Make sure age (stage) groups do not have overlapping ages
    - Make sure species defined in inter-species relationships are included

    :param Domain domain: Domain instance
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

        if np.any(np.diff(np.sort(ages)) != 1):
            raise SolverError(
                'The species (key) {} {}s have group age ranges that overlap or have gaps'.format(
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
    recipient_ranges = {}
    def get_species(d):
        for key, val in d.items():
            if isinstance(val, dict):
                get_species(val)
            elif isinstance(val, tuple):
                if val[0].species is not None:
                    species.append(val[0].species.name_key)
                if hasattr(val[0], 'recipient_species') and val[0].recipient_species is not None:
                    species.append(val[0].recipient_species.name_key)

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

    def all_times(domain):
        """duplicate of summary.all_times"""
        def _next(gp_cnt):
            group, cnt = gp_cnt
            # time is always 4 nodes down
            if cnt == 4:
                times.append(int(group.name.split('/')[-1]))
            else:
                for key in group.keys():
                    _next((group[key], cnt + 1))

        times = []

        for key in domain.file.keys():
            _next((domain.file[key], 1))

        return np.unique(times)


def inherit(domain, start_time, end_time):
    """
    Divide population and carrying capacity among children of species if they exist, and apply max age truncation to
    all children species.

    :param domain: Domain instance
    :param int start_time: A simulation start time that coincides with that of the specified solver time range
    :param int end_time: A simulation stop time that coincides with that of the specified solver time range
    """
    # Iterate all species and times in the domain
    for species in domain.species.keys():
        # Check on the age truncation
        live_past_max = False
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

            if None in sex_keys:
                if (None in domain.species[species][None].keys() and
                    isinstance(domain.species[species][None][None], Species)):
                    live_past_max = domain.species[species][None][None].live_past_max

            # Iterate any sexes in the domain
            for sex in sex_keys:
                if live_past_max and None in domain.species[species][sex].keys():
                    instance = domain.species[species][sex][None]
                    if isinstance(instance, Species):
                        print('{} {} inheriting live_past_max from {}'.format(species, sex, species))
                        instance.live_past_max = True

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
                        if live_past_max:
                            # Apply the sex-level live-past-max attribute to the child
                            print('{} {} {} inheriting live_past_max from {} {}'.format(species, sex, group,
                                                                                        species, sex))
                            instance.live_past_max = True

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
    Solve the domain from a start time to an end time on a discrete time step associated with the Domain.

    The total populations from the previous year are used to calculate derived
    parameters (density, fecundity, inter-species relationships, etc.).

    Parameters used during execution are stored in the domain under the ``params`` key, while magnitude
    related to mortality and fecundity are stored under the ``flux`` key at each time step.

    To run the simulation, use .execute()
    """
    def __init__(self, domain, start_time, end_time, **kwargs):
        """
        :param Domain domain: Domain instance populated with at least one species
        :param int start_time: Start time of the simulation
        :param int end_time: End time of the simulation

        :Keyword Arguments:
            **total_density** (*bool*) --
                Use the total species population for density calculations (as opposed to
                populations of sexes or groups) (Default: True)
        """
        # Prepare time
        start_time = Domain._get_time_input(start_time)
        end_time = Domain._get_time_input(end_time)
        self.simulation_range = range(start_time + 1, end_time + 1)

        self.D = domain
        self.prepare_domain(start_time, end_time)

        # Extra toggles
        # -------------
        self.total_density = kwargs.get('total_density', True)

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
            output, _delayed = {}, {}

            for species in all_species:
                # Collect sex keys
                sex_keys = self.D.species[species].keys()
                # Sex may not exist and will be None, but if one of males or females are included,
                # do not propagate Nonetypes
                if any([fm in sex_keys for fm in ['male', 'female']]):
                    sex_keys = [key for key in sex_keys if key is not None]

                # All births are recorded from each sex and are added to the age 0 slot
                self.age_zero_population[species] = {key: da_zeros(self.D.shape, self.D.chunks)
                                                     for key in sex_keys}

                # Iterate sexes and groups and calculate each individually
                for sex in sex_keys:
                    for group in self.D.species[species][sex].keys():  # Group - may not exist and will be None
                        _output, __delayed = self.propagate(species, sex, group, time)
                        output.update(_output)
                        _delayed.update(__delayed)

                # Update the output with the age 0 populations
                for _sex in self.age_zero_population[species].keys():

                    zero_group = self.D.group_from_age(species, _sex, 0)
                    if zero_group is None:

                        # Apply the offspring to the youngest group or do not record an age
                        zero_group = self.D.youngest_group(species, _sex)
                        if zero_group is not None:
                            if time == self.simulation_range[0]:
                                print('Warning: no group with age 0 exists for {} {}s.\n'
                                      'Offspring will be added to the group {}.'.format(
                                    zero_group.name, _sex, zero_group.group_name
                                ))

                            age = zero_group.min_age
                            zero_group = zero_group.group_key

                        else:
                            # Age is not recorded
                            age = None
                    else:
                        age = 0
                    key = '{}/{}/{}/{}/{}'.format(species, _sex, zero_group, time, age)
                    try:
                        # Add them together if the key exists in the output
                        output[key] += self.age_zero_population[species][_sex]
                    except KeyError:
                        output[key] = self.age_zero_population[species][_sex]

            # Compute this time slice and write to the domain file
            self.D._domain_compute(output, _delayed)

    def totals(self, all_species, time):
        """
        Collect all population and carrying capacity datasets, and calculate species totals if necessary.
        Dynamic calculations of carrying capacity are made by calling collect_parameter, which may also
        update the carrying_capacity or population array dicts.

        :param list all_species: All species in the domain
        :param int time: Time slice to collect totals
        """
        for species in all_species:
            # Need to collect the total population for fecundity and density if not otherwise specified
            if self.total_density:
                # Population
                population = self.D.all_population(species, time - 1)
                # Total population of contributing groups
                self.population_arrays[species]['total'] = self.population_total(species, population)

                # Carrying Capacity
                # -----------------------------------
                # Collect all carrying capacity keys
                cc = self.D.all_carrying_capacity(species, time)
                self.carrying_capacity_arrays[species]['total'] = self.carrying_capacity_total(species, cc)

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
            # Generate lookup data using the density of the attached species
            population = self.D.all_population(param.species.name_key, self.current_time - 1,
                                      param.species.sex, param.species.group_key)
            p = self.population_total(param.species.name_key, population)

            carrying_capacity = self.D.all_carrying_capacity(
                    param.species.name_key, self.current_time, param.species.sex, param.species.group_key
                )
            cc = self.carrying_capacity_total(param.species.name_key, carrying_capacity)

            kwargs.update({'lookup_data': da.where(cc > 0, p / cc, np.inf), 'lookup_table': param.species_table})
        else:
            data = da.from_array(data, data.chunks)

        # Include random parameters (except not if carrying capacity is another species)
        if getattr(param, 'random_method', False) and not (isinstance(param, CarryingCapacity) and data is None):
            kwargs.update({'random_method': param.random_method, 'random_args': param.random_args,
                           'apply_using_mean': param.apply_using_mean})

        kwargs.update({'chunks': self.D.chunks})
        return dynamic.collect(data, **kwargs)

    def interspecies_carrying_capacity(self, parent_species, param):
        """
        Because inter-species dependencies may be nested or circular, the carrying capacity of all connected
        species is calculated herein and added to the ``self.carrying_capacity_arrays`` object. Future calls of any
        of the connected species will have data available, avoiding redundant calculations.

        :param param: input instance of CarryingCapacity
        :return: coefficient array for the given carrying capacity parameter
        """
        def _population(species):
            """Population at the previous time"""
            population = self.D.all_population(species.name_key, self.current_time - 1, species.sex, species.group_key)
            return self.population_total(species.name_key, population)

        def next_species_node(param):
            """Prepare the next node in the species graph"""
            # Condition that terminates recursion
            if not param.species in density:
                density[param.species] = {'p': _population(param.species), 'd': da_zeros(self.D.shape, self.D.chunks)}
                cc = self.D.all_carrying_capacity(
                    param.species.name_key, self.current_time, param.species.sex, param.species.group_key
                )
                for cc_instance, ds in cc:
                    if ds is None:
                        next_species_node(cc_instance)
                    else:
                        if cc_instance not in self.carrying_capacity_arrays[param.species.name_key]:
                            d = self.collect_parameter(cc_instance, ds)
                            self.carrying_capacity_arrays[param.species.name_key][cc_instance] = d
                        else:
                            d  = self.carrying_capacity_arrays[param.species.name_key][cc_instance]
                        density[param.species]['d'] += d

        # Collect the sum of all density parameters without inter-species perturbation
        density = {}  # Global in func next_species_node
        next_species_node(param)

        # Calculate the coefficient for each carrying capacity instance
        def next_species_edge(param, parent_species):
            """Traverse the edges and calculate coefficients"""
            if param not in complete:
                complete.append(param)
                p, d = density[param.species]['p'], density[param.species]['d']
                kwargs = {
                    'lookup_data': da.where(d > 0, p / d, np.inf),
                    'lookup_table': param.species_table,
                    'chunks': self.D.chunks
                }

                self.carrying_capacity_arrays[parent_species][param] = dynamic.collect(None, **kwargs)

                cc = self.D.all_carrying_capacity(
                    param.species.name_key, self.current_time, param.species.sex, param.species.group_key
                )
                for cc_instance, ds in cc:
                    if ds is None:
                        next_species_edge(cc_instance, param.species.name_key)

        complete = []
        next_species_edge(param, parent_species)

    def population_total(self, species, datasets, honor_density=True, honor_reproduction=False):
        """
        Calculate the total population for a species with the provided datasets, usually a result of a
        :func:`Domain.all_population` call. This updates ``self.population_arrays`` to avoid repetitive IO

        :param species: Species key
        :param datasets: population datasets, retrieved from popdyn.Domain.all_population()
        :param bool honor_density: Use the density contribution attribute from a species
        :param bool honor_reproduction: Use the density contribution to fecundity attribute from a species
        :return: Dask array of total population
        """
        pop_arrays = []

        for key, d in datasets.items():
            instance = self.D._instance_from_key(key)

            if honor_density and not instance.contributes_to_density:
                continue
            if honor_reproduction:
                # Collect fecundity to see if it is present
                species_key, sex, group_key, time = self.D._deconstruct_key(key)[:4]

                if len(self.D.get_fecundity(species_key, time, sex, group_key)) == 0:
                    continue

            try:
                pop_arrays.append(self.population_arrays[species][key])
            except KeyError:
                da_array = da.from_array(d, d.chunks)
                pop_arrays.append(da_array)
                self.population_arrays[species][key] = da_array

        if len(pop_arrays) > 0:
            out = dstack(pop_arrays).sum(axis=-1)
        else:
            out = da_zeros(self.D.shape, self.D.chunks)

        return out

    def carrying_capacity_total(self, species, datasets):
        """
        Calculate the total carrying capacity for a species with the provided datasets, usually a result of a
        :func:`Domain.get_carrying_capacity` call. This updates ``self.carrying_capacity_arrays`` to avoid repetitive IO

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
                if cc[1] is None:
                    # Circular relationships are solved here
                    self.interspecies_carrying_capacity(species, cc[0])
                    cc_coeff.append(self.carrying_capacity_arrays[species][cc[0]])
                else:
                    ds = self.collect_parameter(cc[0], cc[1])
                    self.carrying_capacity_arrays[species][cc[0]] = ds
                    cc_arrays.append(ds)

        if len(cc_arrays) == 0:
            array = da_zeros(self.D.shape, self.D.chunks)
        else:
            array = dstack(cc_arrays).sum(axis=-1)

        if len(cc_coeff) == 0:
            coeff = 1.
        else:
            # Use the mean of the coefficients to avoid compounding perturbation
            coeff = dstack(cc_coeff).mean(axis=-1)

        return da.where(array > 0, array * coeff, 0)

    def propagate(self, species, sex, group, time):
        """
        Create a graph of calculations to propagate populations and record parameters.
        A dict of key pointers - dask array pairs are created and returned

        .. Note::
            - If age gaps exist, they will be filled by empty groups and will propagate
            - Currently, female:male ratio uses the total species population even if densities are calculated at the
              age group level

        :param str species: Species key (``Species.name_key``)
        :param str sex: Sex (``'male'``, ``'female'``, or ``None``)
        :param str group: Group key (``AgeGroup.group_key``)
        :param int time: Time slice
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
            return {}, {}

        # Collect parameters from the current time step.
        # All dynamic modifications are also applied in this step
        # -----------------------------------------------------------------
        params = self.calculate_parameters(species, sex, group, time)

        # Collect mortality instances
        mort_types = [mort_type[0] for mort_type in self.D.get_mortality(species, time, sex, group)]

        # All outputs are written at once, so create a dict to do so (pre-populated with mortality types as zeros)
        output, _delayed = {}, {}
        for mort_type in mort_types:
            mort_name = mort_type.name
            output['{}/mortality/{}'.format(flux_prefix, mort_name)] = da_zeros(self.D.shape, self.D.chunks)
            # Write the output mortality parameters
            output['{}/mortality/{}'.format(param_prefix, mort_name)] = params[mort_name]
        for mort_name in ['Old Age', 'Density Dependent']:
            output['{}/mortality/{}'.format(flux_prefix, mort_name)] = da_zeros(self.D.shape, self.D.chunks)

        # Add carrying capacity to the output datasets
        if self.carrying_capacity_total:
            # For the total species population only
            total_prefix = '{}/{}/{}/{}/params'.format(species, None, None, time)
            output['{}/carrying capacity/{}'.format(total_prefix, 'Carrying Capacity')] = params['Carrying Capacity']
        else:
            # Specific to the species
            output['{}/carrying capacity/{}'.format(param_prefix, 'Carrying Capacity')] = params['Carrying Capacity']

        if sex is not None:
            # Check if this species is capable of offspring
            # Calculate births using the effective fecundity
            # The total population of this group is used, as fecundity will be 0
            # within groups that do not reproduce
            output['{}/fecundity/{}'.format(param_prefix, 'Fecundity')] = params['Fecundity']
            output['{}/fecundity/{}'.format(param_prefix, 'Density-Based Fecundity Reduction Rate')] = \
                params['Density-Based Fecundity Reduction Rate']
            births = params['Population'] * params['Fecundity']
            male_births = births * params.get('Birth Ratio', 0.5)
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

        # Density-dependent mortality
        # ---------------------------
        # Constants that scale density-dependent mortality
        density_threshold = species_instance.density_threshold
        density_scale = species_instance.density_scale
        linear_range = max(0., 1. - density_threshold)

        # Calculate a density-dependent mortality rate if this species contributes to density
        # The density parameter is either the total species or group value, depending on the flag
        if species_instance.contributes_to_density:
            # Make the density 1 or the threshold if it is inf
            # Density dependent rate
            dd_range = da.maximum(0., params['Density'] - density_threshold)
            # Apply the scale
            if linear_range == 0:
                linear_proportion = 1.
            else:
                linear_proportion = da.minimum(1., dd_range / linear_range)
            ddm_rate = da.where(
                (params['Density'] > 0) & ~da.isinf(params['Density']),
                (dd_range * (density_scale * linear_proportion)) / params['Density'], 0
            )
            # Make the rate 1 where there is an infinite density
            ddm_rate = da.where(da.isinf(params['Density']), 1., ddm_rate)

        for age in ages:
            # Collect the population of the age from the previous time step
            population = self.D.get_population(species, time - 1, sex, group, age)

            # Use population total to avoid re-read from disk
            population = self.population_total(species, {population: self.D[population]}, False)

            # Mortality is applied on each age, as they need to be removed from the population
            # Aggregate mortality must be scaled so as to not exceed 1.
            if len(mort_types) > 0:
                agg_mort = dstack([params[mort_type.name] for mort_type in mort_types]).sum(axis=-1)
                mort_coeff = da.where(agg_mort > 1., 1. - ((agg_mort - 1.) / agg_mort), 1.)

            mort_fluxes = []
            for mort_type in mort_types:
                mort_fluxes.append(params[mort_type.name] * mort_coeff * population)
                output['{}/mortality/{}'.format(flux_prefix, mort_type.name)] += mort_fluxes[-1]

                # Apply mortality to any recipients
                if mort_type.recipient_species is not None:
                    # The population is added to the model domain for the same age,
                    # and is added to any existing population
                    other_species = mort_type.recipient_species
                    existing_pop = self.D.get_population(
                        other_species.name_key, time, other_species.sex, other_species.group_key, age
                    )

                    if age not in other_species.age_range:
                        raise PopdynError('Could not apply mortality to recipient species {} because the age {} '
                                          'does not exist for the group {}'.format(
                            other_species.name, age, other_species.group_name)
                        )

                    try:
                        output['{}/mortality/Converted to {}'.format(
                            flux_prefix, other_species.name)] += mort_fluxes[-1]
                    except KeyError:
                        output['{}/mortality/Converted to {}'.format(
                            flux_prefix, other_species.name)] = mort_fluxes[-1]

                    other_species_key = '{}/{}/{}/{}/{}'.format(
                        other_species.name_key, other_species.sex, other_species.group_key, time, age
                    )

                    if existing_pop is not None:
                        other_species_data = (
                                da.from_array(self.D[existing_pop], self.D.chunks) + mort_fluxes[-1]
                        )
                    else:
                        other_species_data = mort_fluxes[-1]

                    # If multiple species are contributing to the recipient, they must be added in a delayed fashion
                    try:
                        _delayed[other_species_key] += other_species_data
                    except KeyError:
                        _delayed[other_species_key] = other_species_data

            # Reduce the population by the mortality
            for mort_flux in mort_fluxes:
                population -= mort_flux

            # Apply old age mortality if necessary to avoid unnecessary dispersal calculations
            if not species_instance.live_past_max and max_age is not None and age == max_age:
                output['{}/mortality/{}'.format(flux_prefix, 'Old Age')] += population
                # All done with this population
                continue

            # Apply dispersal
            # TODO: Do children receive dispersal of any parent species classes?

            static_population = population.copy()

            for dispersal_method, args in species_instance.dispersal:
                args = args + (self.D.csx, self.D.csy)
                # Gather a mask if it exists
                mask_ds = self.D.get_mask(species, self.current_time, sex, group)
                if mask_ds is not None:
                    mask_ds = da.from_array(self.D[mask_ds], self.D.chunks)
                disp_kwargs = {'mask': mask_ds}
                population = dispersal.apply(population,
                                             self.population_arrays[species]['total'],
                                             self.carrying_capacity_arrays[species]['total'],
                                             dispersal_method, args, **disp_kwargs)

            if species_instance.contributes_to_density:
                # Avoid density-dependent mortality when dispersal has occurred
                # NOTE: This may still leave a higher density than the threshold if dispersal has exceeded the
                #  carrying capacity, as the DDM rate does not increase (only decreases to ensure dispersal
                #  results in populations being allowed to live where their density has become lower).
                #  Increasing the rate to account for dispersed populations may result in excessive
                #  population reduction to below the carrying capacity. Leave those to the next time step.
                new_ddm_rate = da.where(
                    population < static_population, ddm_rate - (1 - (population / static_population)),
                    ddm_rate
                )

                ddm = da.where(new_ddm_rate > 0., population * new_ddm_rate, 0)

                output['{}/mortality/{}'.format(param_prefix, 'Density Dependent Rate')] = new_ddm_rate
                output['{}/mortality/{}'.format(flux_prefix, 'Density Dependent')] += ddm

                population -= ddm

            # Propagate age by one increment and _save to the current time step.
            # If the input age is None, the population will simply propagate
            if age is not None:
                new_age = age + 1
            else:
                new_age = age
            new_group = self.D.group_from_age(species, sex, new_age)
            if new_group is None:
                # No group past this point. If there is a legitimate age, this means that live_past_max is True
                if max_age is not None and age == max_age:
                    # Need to add an age sequentially to the current group
                    self.D.species[species][sex][group].max_age += 1
                    new_group = group
                else:
                    new_age = None

            # Lastly, apply minimum viable population calculations if necessary
            if species_instance.minimum_viable_population > 0:
                population, mvp_mort = dispersal.minimum_viable_population(
                    population, species_instance.minimum_viable_population, species_instance.minimum_viable_area,
                    self.D.csx, self.D.csy
                )

            key = '{}/{}/{}/{}/{}'.format(species, sex, new_group, time, new_age)
            try:
                # In case a duplicate key exists, addition is first attempted at the
                # current time step for immigration/emigration
                _delayed[key] = da.from_array(self.D[key], chunks=self.D.chunks) + population
                # Do not allow negative populations
                _delayed[key] = da.where(_delayed[key] < 0, 0, _delayed[key])
            except KeyError:
                output[key] = da.where(population < 0, 0, population)

        # Return output so that it may be included in the final compute call
        return output, _delayed

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

        # Carrying Capacity
        # ---------------------
        # Collect CarryingCapacity instances and/or data from the domain at the current time step
        if not self.total_density:
            carrying_capacity = self.D.get_carrying_capacity(species, time, sex, group)
            parameters['Carrying Capacity'] = self.carrying_capacity_total(species, carrying_capacity)

        else:
            parameters['Carrying Capacity'] = self.carrying_capacity_arrays[species]['total']

        # Total Population from previous time step (all population, without regard to contribution filtering)
        # ---------------------
        population_dsts = self.D.all_population(species, time - 1, sex, group)
        parameters['Population'] = self.population_total(species, population_dsts, False)

        # Population only includes contributing species instances
        if not self.total_density:
            population = self.population_total(species, population_dsts)
        else:
            # Look for a minimum viable population value in a Species-level instance, and apply it to the total
            # population in advance of propagating the species
            mvp = 0
            for sex in self.D.species[species].keys():
                if sex is None:
                    for gp in self.D.species[species][sex].keys():
                        if gp is None:
                            if isinstance(self.D.species[species][sex][gp], Species):
                                species_instance = self.D.species[species][sex][gp]
                                mvp = species_instance.minimum_viable_population
            if mvp > 0:
                self.population_arrays[species]['total'], mvp_mort = dispersal.minimum_viable_population(
                    self.population_arrays[species]['total'], species_instance.minimum_viable_population,
                    species_instance.minimum_viable_area, self.D.csx, self.D.csy
                )
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

        # Calculate the fecundity values]
        # Fecundity is collected then aggregated
        fec_arrays = []
        avg_mod = 0
        for instance, data in fecundity:
            # First, check if the species multiplies
            if getattr(instance, 'multiplies'):
                # Filtered through its density lookup table
                fec = self.collect_parameter(instance, data)

                # Apply random perturbation first
                # Apply randomness
                if getattr(instance, 'random_method', False):
                    fec = dynamic.collect(
                        fec, random_method=instance.random_method, random_args=instance.random_args,
                        apply_using_mean=instance.apply_using_mean, **{'chunks': self.D.chunks}
                    )

                fec *= dynamic.collect(
                    None, lookup_data=self.population_arrays[species]['Female to Male Ratio'],
                    lookup_table=instance.fecundity_lookup, **{'chunks': self.D.chunks}
                )

                # Fecundity scales from a low threshold to high and is reduced linearly using a specified rate
                density_fecundity_threshold = getattr(instance, 'density_fecundity_threshold')
                density_fecundity_max = getattr(instance, 'density_fecundity_max')
                fecundity_reduction_rate = min(1., getattr(instance, 'fecundity_reduction_rate'))

                input_range = max(0., density_fecundity_max - density_fecundity_threshold)
                density_range = parameters['Density'] - density_fecundity_threshold

                if input_range == 0:
                    fec_mod = da.where(density_range > 0, fecundity_reduction_rate, 0.)
                else:
                    fec_mod = da.where(
                        density_range > 0, da.minimum(1., (density_range / input_range)) * fecundity_reduction_rate, 0.
                    )

                fec -= fec * fec_mod

                avg_mod += fec_mod

                fec = da.where(fec > 0, fec, 0)

                # If multiple fecundity instances are specified for this species, only the last birth ratio will
                #  be used
                birth_ratio = instance.birth_ratio

                if birth_ratio == 'random':
                    parameters['Birth Ratio'] = np.random.random()
                else:
                    parameters['Birth Ratio'] = birth_ratio

                # Add to stack
                fec_arrays.append(fec)

        if len(fec_arrays) == 0:
            # Defaults to 0
            parameters['Fecundity'] = da_zeros(self.D.shape, self.D.chunks)
            parameters['Density-Based Fecundity Reduction Rate'] = da_zeros(self.D.shape, self.D.chunks)
        else:
            parameters['Fecundity'] = dstack(fec_arrays).sum(axis=-1)
            parameters['Density-Based Fecundity Reduction Rate'] = avg_mod / len(fec_arrays)

        return parameters
