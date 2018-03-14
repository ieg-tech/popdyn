"""
Population dynamics numerical solvers

Devin Cairns, 2018
"""
from popdyn import *
import dask.array as da
from dask import delayed


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


def cascade_population(domain, start_time, end_time):
    """
    Divide population among children of species if they exist
    :param domain:
    :param start_time:
    :return:
    """
    # Iterate all species and times in the domain
    for species in domain.species.keys():
        sex_keys = [key for key in domain.species[species].keys() if key is not None]
        for time in range(start_time, end_time + 1):

            # Collect the toplevel species dataset if it exists, and distribute it among the sexes
            ds = None
            species_population_key = domain.get_population(species, time)
            if species_population_key is not None and len(sex_keys) > 0:
                # Read the species-level population data and remove from the domain
                ds = domain[species_population_key][:] / len(sex_keys)
                del domain.population[species][None][None][time][None]  # Group, sex and age are None
                del domain.file[species_population_key]

            # Iterate any sexes in the domain
            for sex in sex_keys:
                # Collect groups
                group_keys = [key for key in domain.species[species][sex].keys() if key is not None]
                if len(group_keys) == 0:
                    # Apply any species-level data to the sex dataset
                    if ds is not None:
                        instance = domain.species[species][sex][None]
                        print("Cascading population {} --> {}".format(species, sex))
                        domain.add_population(instance, ds, time,
                                              distribute=False, overwrite='add')
                else:
                    # Combine the species-level and sex-level datasets
                    sex_population_key = domain.get_population(species, time, sex)
                    if sex_population_key is None:
                        if ds is None:
                            # No populations at species or sex level
                            continue
                        else:
                            # Use only the species-level key divided among groups
                            sp_sex_prefix = '{} --> [{}] --> '.format(species, sex)
                            next_ds = ds / len(group_keys)
                    else:
                        # Collect the sex-level data and remove it from the Domain
                        next_ds = domain[sex_population_key][:]
                        del domain.population[species][sex][None][time][None]  # Group and age are None
                        del domain.file[sex_population_key]

                        # Add the species-level data if necessary and divide among groups
                        if ds is not None:
                            sp_sex_prefix = '{} --> {} --> '.format(species, sex)
                            next_ds += ds
                        else:
                            sp_sex_prefix = '[{}] --> {} --> '.format(species, sex)
                        next_ds /= len(group_keys)

                    # Apple populations to groups
                    for group in group_keys:
                        instance = domain.species[species][sex][group]
                        print("Cascading population {}{}".format(sp_sex_prefix, group))
                        domain.add_population(instance, next_ds, time,
                                              distribute=False, overwrite='add')


class discrete_explicit(object):
    """
    Solve the domain from a start time to an end time on a discrete time step,
    which is dictate by the Domain configuration.

    The total populations from the previous year are used to calculate derived
    parameters (density, fecundity, inter-species relationships, etc.).
    """
    def time_this(func):
        """Decorator for profiling methods"""
        def inner(*args, **kwargs):
            instance = args[0]
            start = profile_time.time()
            execution = func(*args, **kwargs)
            try:
                instance.profiler[func.__name__] += profile_time.time() - start
            except KeyError:
                instance.profiler[func.__name__] = profile_time.time() - start
            return execution
        return inner

    @time_this
    def __init__(self, domain, start_time, end_time):
        """
        :param domain: Domain instance populated with at least one species
        :param start_time: Start time of the simulation
        :param end_time: End time of the simulation
        :return: None
        """
        # Prepare time
        start_time = Domain.get_time_input(start_time)
        end_time = Domain.get_time_input(end_time)
        self.simulation_range = range(start_time + 1, end_time + 1)

        # Domain preparation
        error_check(domain)

        # Distribute population to lowest possible levels
        cascade_population(domain, start_time, end_time)

        self.D = domain

    def execute(self):
        """Run the simulation"""
        # Iterate time. The first time step cannot be solved and serves to provide initial parameters
        for time in self.simulation_range:
            # base time is the previous time step, which is used to collect baseline data
            base_time = time - 1
            # Iterate species and solve each independently, deriving relationships from the baseline data
            for species in self.D.species.keys():
                # Collect the baseline populations and delay calculations of derived parameters
                pass

            # If all baseline populations are zero, fill output with zeros and propagate


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

        self.reproducingFemales = numpy.unique(self.reproducingFemales)
        self.reproducingMales = numpy.unique(self.reproducingMales)

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
                    self.formalLog['Habitat'][species].append(numpy.sum(k))

                    get_carry_capacity_time += profile_time.time() - now

                    # Collect total population using contributing age groups
                    now = profile_time.time()

                    def contributing_total(inclusion_males, inclusion_females, migrated=False):
                        if inclusion_males is not None:
                            males = numpy.zeros(shape=self.shape, dtype='float32')
                            for pop_gp in inclusion_males:
                                males += self.total_population(species, time_step, age_gp=pop_gp,
                                                               sex='male', migrated=migrated, file_instance=f)
                        else:
                            males = self.total_population(species, time_step, sex='male', migrated=migrated,
                                                          file_instance=f)
                        if inclusion_females is not None:
                            females = numpy.zeros(shape=self.shape, dtype='float32')
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
                        self.formalLog['Population'][species]['NA']['Contributing Males'].append(numpy.sum(contMales))
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Contributing Males'] = [numpy.sum(contMales)]
                    try:
                        self.formalLog['Population'][species]['NA']['Contributing Females'].append(numpy.sum(contFemales))
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Contributing Females'] = [numpy.sum(contFemales)]
                    try:
                        self.formalLog['Population'][species]['NA']['Total Population'].append(numpy.sum(
                            self.total_population(species, time_step)
                        ))
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Total Population'] = [numpy.sum(
                            self.total_population(species, time_step)
                        )]
                    try:
                        self.formalLog['Population'][species]['NA']['Total Males'].append(numpy.sum(
                            self.total_population(species, time_step, sex='male')
                        ))
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Total Males'] = [numpy.sum(
                            self.total_population(species, time_step, sex='male')
                        )]
                    try:
                        self.formalLog['Population'][species]['NA']['Total Females'].append(numpy.sum(
                            self.total_population(species, time_step, sex='female')
                        ))
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Total Females'] = [numpy.sum(
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
                        self.formalLog['Population'][species]['NA']['Reproducing Males'].append(numpy.sum(reproducingMales))
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Reproducing Males'] = [numpy.sum(reproducingMales)]
                    try:
                        self.formalLog['Population'][species]['NA']['Reproducing Females'].append(
                            numpy.sum(reproducingFemales))
                    except KeyError:
                        self.formalLog['Population'][species]['NA']['Reproducing Females'] = [numpy.sum(reproducingFemales)]
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
                    zero_males = numpy.zeros(shape=self.shape,
                                             dtype='float32')
                    zero_females = numpy.zeros(shape=self.shape,
                                               dtype='float32')

                    birth_calc_time += profile_time.time() - now

                    # Iterate age groups
                    density_dependent_sum = numpy.zeros(shape=self.shape, dtype='float32')
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
                            self.formalLog['Natality'][species][gp]['Minimum fecundity'].append(numpy.min(fec))
                        except KeyError:
                            self.formalLog['Natality'][species][gp]['Minimum fecundity'] = [numpy.min(fec)]
                        try:
                            self.formalLog['Natality'][species][gp]['Maximum fecundity'].append(numpy.max(fec))
                        except KeyError:
                            self.formalLog['Natality'][species][gp]['Maximum fecundity'] = [numpy.max(fec)]
                        try:
                            self.formalLog['Natality'][species][gp]['Mean fecundity'].append(numpy.mean(fec))
                        except KeyError:
                            self.formalLog['Natality'][species][gp]['Mean fecundity'] = [numpy.mean(fec)]

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

                        sumFemales, sumMales = numpy.sum(total_females), numpy.sum(total_males)

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
                            birth_proportion = numpy.random.random(self.shape)
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

                        sumFemaleBirths, sumMaleBirths = numpy.sum(female_births), numpy.sum(male_births)
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
                                        timeIndex] = numpy.min(mortality_rates[mortality_name])
                                except KeyError:
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} minimum'.format(sex, self.mortality_names[mortality_name])] = \
                                        numpy.zeros(shape=len(simulationRange), dtype='float32')
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} minimum'.format(sex, self.mortality_names[mortality_name])][
                                        timeIndex] = numpy.min(mortality_rates[mortality_name])
                                try:
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} maximum'.format(sex, self.mortality_names[mortality_name])][
                                        timeIndex] = numpy.max(mortality_rates[mortality_name])
                                except KeyError:
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} maximum'.format(sex, self.mortality_names[mortality_name])] = \
                                        numpy.zeros(shape=len(simulationRange), dtype='float32')
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} maximum'.format(sex, self.mortality_names[mortality_name])][
                                        timeIndex] = numpy.max(mortality_rates[mortality_name])
                                try:
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} mean'.format(sex, self.mortality_names[mortality_name])][
                                        timeIndex] = numpy.mean(mortality_rates[mortality_name])
                                except KeyError:
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} mean'.format(sex, self.mortality_names[mortality_name])] = \
                                        numpy.zeros(shape=len(simulationRange), dtype='float32')
                                    self.formalLog['Mortality'][species][gp][
                                        '{} {} mean'.format(sex, self.mortality_names[mortality_name])][
                                        timeIndex] = numpy.mean(mortality_rates[mortality_name])

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
                                        maxAgeDeaths = numpy.sum(_pop)
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
                                            numpy.zeros(shape=len(simulationRange), dtype='float32')
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

                                    mortDeaths = numpy.sum(_deaths)

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
                                            numpy.zeros(shape=len(simulationRange), dtype='float32')
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
                        if numpy.sum(births) > 0:
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
                            dDdeath = numpy.sum(death)
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
                                    numpy.zeros(shape=len(simulationRange), dtype='float32')
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
                                numpy.zeros(shape=len(simulationRange), dtype='float32')
                            self.formalLog['Mortality'][species]['NA']['Total deaths from {}'.format(mortMode)][
                                timeIndex] = modeDeaths[mortMode]
                        try:
                            self.formalLog['Mortality'][species]['NA']['Male deaths from {}'.format(mortMode)][
                                timeIndex] = modeMaleDeaths[mortMode]
                        except KeyError:
                            self.formalLog['Mortality'][species]['NA']['Male deaths from {}'.format(mortMode)] = \
                                numpy.zeros(shape=len(simulationRange), dtype='float32')
                            self.formalLog['Mortality'][species]['NA']['Male deaths from {}'.format(mortMode)][
                                timeIndex] = modeMaleDeaths[mortMode]
                        try:
                            self.formalLog['Mortality'][species]['NA']['Female deaths from {}'.format(mortMode)][
                                timeIndex] = modeFemaleDeaths[mortMode]
                        except KeyError:
                            self.formalLog['Mortality'][species]['NA']['Female deaths from {}'.format(mortMode)] = \
                                numpy.zeros(shape=len(simulationRange), dtype='float32')
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
