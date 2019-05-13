"""
Summarize data from a PopDyn simulation

Devin Cairns 2018
"""

import popdyn as pd
import h5fake as h5py
import dask.array as da
import numpy as np
from datetime import datetime
from dateutil.tz import tzlocal
from copy import deepcopy


class SummaryDataset(object):

    def __init__(self, shape=None, key=None):
        if shape is not None:
            self.value = np.zeros(shape, np.float32)
        if key is not None:
            self.key = key

    def __setitem__(self, s, item):
        if not hasattr(self, 'value'):
            # The slice is ignored, as all computations are reduced
            self.value = item
        else:
            # Use numpy-like setitem
            self.value[s] = item


class ModelSummary(object):
    """
    Collect summarized results from the model domain. With each summary function call, no computations take place.
    Rather, the ``to_compute`` attribute is appended with the output ``dask`` object, all of which are optimized and
    computed only when the :func:`compute` method is called.

    .. note:: The ``to_compute`` attribute maintains the order of summary function calls. The returned list from ``compute`` will maintain summarized data in the call order.
    """

    def __init__(self, domain):
        """
        :param domain: A Domain instance
        """
        # Create a log dictionary to populate with values
        self.domain = domain
        self.model_times = self.all_times

        log = {'Time': self.model_times,
               'Habitat': {}, 'Population': {}, 'Natality': {}, 'Mortality': {},
               'Domain': {'Domain size': str(domain.shape),
                          'Cell size (x)': domain.csx,
                          'Cell size (y)': domain.csy,
                          'Top corner': domain.top,
                          'Left corner': domain.left,
                          'Distributed scheduler chunk size': domain.chunks,
                          'Popdyn file': domain.path,
                          'Spatial Reference': domain.projection,
                          'Avoid Inheritance': domain.avoid_inheritance,
                          'Age Groups': {},
                          'Age Ranges': {}},
               'Species': {},
               'Mortality Params': {},
               'Fecundity Params': {},
               'K Params': {},
               'Solver': [datetime.now(tzlocal()).strftime('%A, %B %d, %Y %I:%M%p %Z')] + \
                         ['{},{:.4f}'.format(key, val) for key, val in domain.timer.methods.items()]
               }


        for spec in np.unique(domain.species_instances).tolist():
            for key, val in spec.__dict__.items():
                log['Species']['{} {} {} {}'.format(spec.name, spec.sex, spec.group_key, key)] = val

        for inst in domain.mortality_instances:
            for key, val in inst.__dict__.items():
                log['Mortality Params'][inst.name + ' ' + key] = val

        for inst in domain.fecundity_instances:
            for key, val in inst.__dict__.items():
                log['Fecundity Params'][inst.name + ' ' + key] = val

        for inst in domain.carrying_capacity_instances:
            for key, val in inst.__dict__.items():
                log['K Params'][inst.name + ' ' + key] = val

        self.summary = {sp: deepcopy(log) for sp in domain.species.keys()}

        # TODO: Make this an HDF5 file to avoid loading all computations into memory
        self.to_compute = []

        # Load all datasets in the domain into dask arrays so they may be uniquely tokenized
        def load_ds(key):
            ds = domain.file[key]
            if isinstance(ds, h5py.Dataset):
                self.arrays[key] = da.from_array(domain[key], domain.chunks)
            else:
                for _key in domain.file[key].keys():
                    load_ds('{}/{}'.format(key, _key))

        self.arrays = {}
        for key in domain.file.keys():
            load_ds(key)

    def compute(self):
        """
        Compute all loaded summaries in the ``to_compute`` attribute

        :returns: list of **computed** computed objects from ``to_compute`` attribute
        """
        # Create output summary dataset objects as store locations
        targets = [SummaryDataset(ds.shape) for ds in self.to_compute]

        # Optimize and compute the dask graph
        pd.store(self.to_compute, targets)

        # Return the output computed data
        return [target.value for target in targets]


    def total_population(self, species=None, time=None, sex=None, group=None, age=None):
        """
        Collect the sum of populations from a domain, filtering by sub-populations

        :param str species: Species name - may be an iterable
        :param time: time slice: int or iterable
        :param sex: Sex (one of 'male' or 'female') - may be an iterable
        :param group: group name - may be an iterable
        :param age: discrete Age - may be an iterable
        :return: ndarray with the total population
        """
        species, time, sex, group, ages = self.collect_iterables(species, time, sex, group, age)

        populations = []
        for t in time:
            for sp in species:
                for s in sex:
                    for gp in group:
                        # If ages are None, they must be collected for the group
                        if len(ages) == 0:
                            try:
                                _ages = self.domain.age_from_group(sp, s, gp)  # Returns a list
                            except pd.PopdynError:
                                # This group is not associated with the current species or sex in the iteration
                                continue
                        else:
                            _ages = ages
                        for age in _ages:
                            population = self.domain.get_population(sp, t, s, gp, age)
                            if population is not None:
                                populations.append(self.arrays[population])

        if len(populations) == 0:
            self.to_compute.append(pd.da_zeros(self.domain.shape, self.domain.chunks))
        else:
            self.to_compute.append(pd.dsum(populations))

    def average_age(self, species=None, time=None, sex=None, group=None):
        """
        Collect the average age from a domain, filtering by sub-populations

        :param str species: Species name - may be an iterable
        :param time: time slice: int or iterable
        :param sex: Sex (one of 'male' or 'female') - may be an iterable
        :param group: group name - may be an iterable\
        :return: ndarray with the average age
        """
        species, time, sex, group = self.collect_iterables(species, time, sex, group)

        ages = {}
        for t in time:
            for sp in species:
                for s in sex:
                    for gp in group:
                        try:
                            _ages = self.domain.age_from_group(sp, s, gp)  # Returns a list
                        except pd.PopdynError:
                            # This group is not associated with the current species or sex in the iteration
                            continue
                        for age in _ages:
                            population = self.domain.get_population(sp, t, s, gp, age)
                            if population is not None:
                                ds = self.arrays[population]
                                try:
                                    ages[age] += ds
                                except KeyError:
                                    ages[age] = ds

        if len(ages) == 0:
            raise pd.PopdynError('No ages were found using the given query')
        else:
            # Population-weighted mean
            pops = pd.dstack(ages.values())
            pops_sum = pops.sum(axis=-1, keepdims=True)
            self.to_compute.append(da.where(pops_sum > 0, ages.keys() * (pops / pops_sum), np.inf).sum(axis=-1))

    def total_carrying_capacity(self, species=None, time=None, sex=None, group=None):
        """
        Collect the total carrying capacity in the domain for the given query.

        .. Attention:: Carrying capacity may be calculated for an entire species in the solver (see :ref:`solvers`).
            As such, query results that are filtered by sex or age group will include carrying capacity for the entire species.

        :param str species: Species name
        :param time: time slice - may be an iterable or int
        :param str sex: Sex ('male' or 'female') - may be an iterable
        :param str group: Group name - may be an iterable
        :return: ndarray of total carrying capacity
        """

        # TODO: Raise exception if specified group or sex does not exist, as it will still return results with typos

        def retrieve(species, time, sex, group):
            species, time, sex, group = self.collect_iterables(species, time, sex, group)

            for t in time:
                for sp in species:
                    for s in sex:
                        for gp in group:
                            cc = '{}/{}/{}/{}/params/carrying capacity/Carrying Capacity'.format(sp, s, gp, t)
                            try:
                                carrying_capacity.append(self.arrays[cc])
                            except KeyError:
                                pass

        carrying_capacity = []
        retrieve(species, time, sex, group)

        # If the total species carrying capacity was used for density during the simulation,
        #  there will not be any carrying capacity associated with sexes/groups.
        if len(carrying_capacity) == 0 and (sex is not None or group is not None):
            # Try to collect the species-wide dataset
            sex, group = None, None
            carrying_capacity = []
            retrieve(species, time, sex, group)

        if len(carrying_capacity) == 0:
            self.to_compute.append(pd.da_zeros(self.domain.shape, self.domain.chunks))
        else:
            self.to_compute.append(pd.dsum(carrying_capacity))

    def list_mortality_types(self, species=None, time=None, sex=None, group=None):
        """
        List the mortality names in the domain for the given query

        :param str species: Species name
        :param time: time slice - may be an iterable or int
        :param str sex: Sex ('male' or 'female') - may be an iterable
        :param str group: Group name - may be an iterable
        :return: list of mortality names
        """
        species, time, sex, group = self.collect_iterables(species, time, sex, group)

        names = []
        for t in time:
            for sp in species:
                for s in sex:
                    for gp in group:
                        try:
                            names += self.domain.file['{}/{}/{}/{}/params/mortality'.format(sp, s, gp, t)].keys()
                            names += self.domain.file['{}/{}/{}/{}/flux/mortality'.format(sp, s, gp, t)].keys()
                        except KeyError:
                            pass

        return np.unique(names)

    def total_mortality(self, species=None, time=None, sex=None, group=None, mortality_name=None, as_population=True):
        """
        Collect either the total population or the mortality parameter value for the given query

        :param str species: Species name
        :param time: time slice - may be an iterable or int
        :param str sex: Sex ('male' or 'female') - may be an iterable
        :param str group: Group name - may be an iterable
        :param bool as_population: Use to determine whether the query output is the population that succumbs to the
            mortality, or the mortality rate
        :return: ndarray of the mortality population or parameter
        """
        species, time, sex, group = self.collect_iterables(species, time, sex, group)

        if as_population:
            _type = 'flux'
        else:
            _type = 'params'

        mortality = []
        for t in time:
            for sp in species:
                for s in sex:
                    for gp in group:
                        base_key = '{}/{}/{}/{}/{}/mortality'.format(sp, s, gp, t, _type)
                        if mortality_name is None:
                            try:
                                names = self.domain.file[base_key].keys()
                            except KeyError:
                                continue
                        else:
                            names = [mortality_name]
                        for name in names:
                            try:
                                mortality.append(self.arrays['{}/{}'.format(base_key, name)])
                            except KeyError:
                                continue

        if len(mortality) == 0:
            self.to_compute.append(pd.da_zeros(self.domain.shape, self.domain.chunks))
        else:
            self.to_compute.append(pd.dsum(mortality))

    def total_offspring(self, species=None, time=None, sex=None, group=None, offspring_sex=None):
        """
        Collect the total offspring using the given query

        :param str species: Species name
        :param time: time slice - may be an iterable or int
        :param str sex: Sex ('male' or 'female') - may be an iterable
        :param str group: Group name - may be an iterable
        :return: ndarray of total offspring
        """
        species, time, sex, group = self.collect_iterables(species, time, sex, group)

        if offspring_sex is None:
            offspring_sex = ['Male Offspring', 'Female Offspring']
        else:
            offspring_sex = [offspring_sex]

        offspring = []
        for t in time:
            for sp in species:
                for s in sex:
                    for gp in group:
                        for _type in offspring_sex:
                            off = '{}/{}/{}/{}/flux/offspring/{}'.format(sp, s, gp, t, _type)
                            try:
                                offspring.append(self.arrays[off])
                            except KeyError:
                                pass

        if len(offspring) == 0:
            self.to_compute.append(pd.da_zeros(self.domain.shape, self.domain.chunks))
        else:
            self.to_compute.append(pd.dsum(offspring))

    def fecundity(self, species=None, time=None, sex=None, group=None, coeff=False):
        """
        Collect the total fecundity using the given query

        :param str species: Species name
        :param time: time slice - may be an iterable or int
        :param str sex: Sex ('male' or 'female') - may be an iterable
        :param str group: Group name - may be an iterable
        :param bool coeff: If True, the average fecundity reduction rate based on density is returned
        :return: ndarray of total offspring
        """
        species, time, sex, group = self.collect_iterables(species, time, sex, group)

        # Determine whether the fecundity value, or the density-based coefficient should be return
        if coeff:
            _type = 'Density-Based Fecundity Reduction Rate'
        else:
            _type = 'Fecundity'

        _fecundity = []
        for t in time:
            for sp in species:
                for s in sex:
                    for gp in group:
                        fec = '{}/{}/{}/{}/params/fecundity/{}'.format(sp, s, gp, t, _type)
                        try:
                            _fecundity.append(self.arrays[fec])
                        except KeyError:
                            pass

        if len(_fecundity) == 0:
            self.to_compute.append(pd.da_zeros(self.domain.shape, self.domain.chunks))
        else:
            if coeff:
                self.to_compute.append(pd.dmean(_fecundity))
            else:
                self.to_compute.append(pd.dsum(_fecundity))

    def model_summary(self):
        """
        Summarize totals of each species and their parameters in the domain. This is used primarily to service the
        data requirements of an excel output in ``logger.write_xlsx``

        :return: dict of species and their parameters
        """
        # TODO: This is built to service the requirements of popdyn.logger.write_xlsx, and could use
        # TODO: a substantial amount of optimization

        model_times = self.model_times

        for species in self.summary.keys():
            lcl_cmp = {}  # Custom compute tree

            # Collect the species name
            try:
                species_name = [name for name in self.domain.species_names if species == pd.name_key(name)][0]
            except IndexError:
                raise pd.PopdynError('Unable to gather the species name from the key {}'.format(species))

            sp_log = self.summary[species]

            # Carrying Capacity NOTE: This should be summarized by group in the future
            ds = []
            change_ds = []
            first_cc = None
            cc_mean_zero = []
            cc_mean_nonzero = []
            for time in model_times:
                self.total_carrying_capacity(species, time)
                cc_a = self.to_compute[-1]
                total_cc = cc_a.sum()
                cc_a_mean = (cc_a.mean() / (self.domain.csx * self.domain.csy)) * 1E6  # Assuming metres
                cc_mean_zero.append(cc_a_mean)
                cc_mean_nonzero.append((cc_a[cc_a != 0].mean() / (self.domain.csx * self.domain.csy)) * 1E6)
                if first_cc is None:
                    first_cc = (cc_a.mean() / (self.domain.csx * self.domain.csy)) * 1E6
                change_ds.append(da.where(first_cc > 0, cc_a_mean / first_cc, 0.))
                ds.append(total_cc)
            key = 'Habitat/{}/Total n'.format(species_name)
            ds[0] = np.nan
            lcl_cmp[key] = da.concatenate(map(da.atleast_1d, ds))
            key = 'Habitat/{}/Relative Change'.format(species_name)
            change_ds[0] = np.nan
            lcl_cmp[key] = da.concatenate(map(da.atleast_1d, change_ds))
            key = 'Habitat/{}/Mean [including zeros] (n per km. sq.)'.format(species_name)
            cc_mean_zero[0] = np.nan
            lcl_cmp[key] = da.concatenate(map(da.atleast_1d, cc_mean_zero))
            key = 'Habitat/{}/Mean [excluding zeros] (n per km. sq.)'.format(species_name)
            cc_mean_nonzero[0] = np.nan
            lcl_cmp[key] = da.concatenate(map(da.atleast_1d, cc_mean_nonzero))

            # Collect average ages
            ave_ages = []
            for time in model_times:
                self.average_age(species, time)
                m = self.to_compute[-1]
                not_inf = ~da.isinf(m)
                ave_ages.append(m[not_inf].mean())
            key = 'Population/{}/NA/Average Age'.format(species_name)
            ave_ages[0] = np.nan
            lcl_cmp[key] = da.concatenate(map(da.atleast_1d, ave_ages))

            # Add total population and lambda population for species
            total_pop = []
            lambda_pop = []
            prv_pop = None
            for time in model_times:
                self.total_population(species, time)
                pop_sum = self.to_compute[-1].sum()
                total_pop.append(pop_sum)
                if prv_pop is not None:
                    lambda_pop.append(pop_sum / prv_pop)
                else:
                    lambda_pop.append(1)
                prv_pop = pop_sum
            key = 'Population/{}/NA/Total Population'.format(species_name)
            lcl_cmp[key] = da.concatenate(map(da.atleast_1d, total_pop))
            key = 'Population/{}/NA/Total Population Lambda'.format(species_name)
            lcl_cmp[key] = da.concatenate(map(da.atleast_1d, lambda_pop))

            # Collect all offspring
            tot_new_off = []
            for time in model_times:
                self.total_offspring(species, time)
                tot_new_off.append(self.to_compute[-1].sum())
            key = 'Natality/{}/NA/Total new offspring'.format(species_name)
            tot_new_off[0] = np.nan
            lcl_cmp[key] = da.concatenate(map(da.atleast_1d, tot_new_off))

            # Collect all deaths
            all_deaths = []
            for time in model_times:
                self.total_mortality(species, time)
                all_deaths.append(self.to_compute[-1].sum())
            key = 'Mortality/{}/NA/All deaths'.format(species_name)
            all_deaths[0] = np.nan
            lcl_cmp[key] = da.concatenate(map(da.atleast_1d, all_deaths))

            # Collect deaths by type
            mort_types = self.list_mortality_types(species, None)
            for mort_type in mort_types:
                if not ('Density Dependent Rate' in mort_type or 'Converted to ' in mort_type):
                    ds = []
                    for time in model_times:
                        self.total_mortality(species, time, mortality_name=mort_type)
                        ds.append(self.to_compute[-1].sum())
                    key = 'Mortality/{}/NA/Total deaths from {}'.format(species_name, mort_type)
                    ds[0] = np.nan
                    lcl_cmp[key] = da.concatenate(map(da.atleast_1d, ds))

            # Iterate groups and populate data
            for sex in ['male', 'female']:
                sp_log['Domain']['Age Groups'][sex] = []
                sp_log['Domain']['Age Ranges'][sex] = []
                sex_str = sex

                # Collect total population by sex
                sex_pop = []
                for time in model_times:
                    self.total_population(species, time, sex)
                    sex_pop.append(self.to_compute[-1].sum())
                key = 'Population/{}/NA/Total {}s'.format(species_name, sex_str[0].upper() + sex_str[1:])
                lcl_cmp[key] = da.concatenate(map(da.atleast_1d, sex_pop))

                if sex == 'female':
                    # Collect offspring / female
                    key = 'Natality/{}/NA/Total offspring per female'.format(species_name)
                    lcl_cmp[key] = da.concatenate(
                        [np.array([np.nan]), da.where(
                            lcl_cmp['Population/{}/NA/Total Females'.format(species_name)][:-1] > 0,
                            lcl_cmp['Natality/{}/NA/Total new offspring'.format(species_name)][1:] /
                            lcl_cmp['Population/{}/NA/Total Females'.format(species_name)][:-1], np.inf
                        )]
                    )

                # Calculate the F:M ratio if both sexes present
                if ('Population/{}/NA/Total Males'.format(species_name) in lcl_cmp.keys() and
                    'Population/{}/NA/Total Females'.format(species_name) in lcl_cmp.keys()):
                    key = 'Natality/{}/NA/F:M Ratio'.format(species_name)
                    lcl_cmp[key] = da.where(
                        lcl_cmp['Population/{}/NA/Total Males'.format(species_name)] > 0,
                        lcl_cmp['Population/{}/NA/Total Females'.format(species_name)] /
                        lcl_cmp['Population/{}/NA/Total Males'.format(species_name)], np.inf
                    )

                # Collect average ages by sex
                ave_ages = []
                for time in model_times:
                    self.average_age(species, time, sex)
                    m = self.to_compute[-1]
                    ave_ages.append(m[~da.isinf(m)].mean())
                key = 'Population/{}/NA/Average {} Age'.format(species_name, sex_str[0].upper() + sex_str[1:])
                ave_ages[0] = np.nan
                lcl_cmp[key] = da.concatenate(map(da.atleast_1d, ave_ages))

                # Offspring by sex
                for __sex in ['male', 'female']:
                    offspring_sex = __sex[0].upper() + __sex[1:] + ' Offspring'
                    ds = []
                    for time in model_times:
                        self.total_offspring(species, time, sex, offspring_sex=offspring_sex)
                        ds.append(self.to_compute[-1].sum())
                    key = 'Natality/{}/NA/{} offspring from {}s'.format(species_name, __sex[0].upper() + __sex[1:],
                                                                        sex_str[0].upper() + sex_str[1:])
                    ds[0] = np.nan
                    lcl_cmp[key] = da.concatenate(map(da.atleast_1d, ds))

                # Collect deaths by sex
                sex_death = []
                for time in model_times:
                    self.total_mortality(species, time, sex)
                    sex_death.append(self.to_compute[-1].sum())
                key = 'Mortality/{}/NA/Total {} deaths'.format(species_name, sex)
                sex_death[0] = np.nan
                lcl_cmp[key] = da.concatenate(map(da.atleast_1d, sex_death))

                # Collect deaths by type/sex
                mort_types = self.list_mortality_types(species, None, sex)
                for mort_type in mort_types:
                    if not ('Density Dependent Rate' in mort_type or 'Converted to ' in mort_type):
                        ds = []
                        for time in model_times:
                            self.total_mortality(species, time, sex, mortality_name=mort_type)
                            ds.append(self.to_compute[-1].sum())
                        key = 'Mortality/{}/NA/{} deaths from {}'.format(species_name, sex_str[0].upper() + sex_str[1:],
                                                                         mort_type)
                        ds[0] = np.nan
                        lcl_cmp[key] = da.concatenate(map(da.atleast_1d, ds))

                for gp in self.domain._group_keys(species):

                    if gp is None:
                        continue

                    try:
                        group_name = [name for name in self.domain.group_names(species) if gp == pd.name_key(name)][0]
                    except IndexError:
                        raise pd.PopdynError('Unable to gather the group name from the key {}'.format(gp))
                    if sex is not None:
                        sp_log['Domain']['Age Groups'][sex].append(group_name)
                        age_range = self.domain.age_from_group(species, sex, gp)
                        sp_log['Domain']['Age Ranges'][sex].append(age_range)

                    # Collect the total population of the group, which is only needed once
                    if 'Population/{}/{}/Total'.format(species_name, group_name) not in lcl_cmp.keys():
                        gp_pop = []
                        lambda_pop = []
                        prv_pop = None
                        for time in model_times:
                            self.total_population(species, time, group=gp)
                            pop_sum = self.to_compute[-1].sum()
                            gp_pop.append(pop_sum)
                            if prv_pop is None:
                                lambda_pop.append(1.)
                            else:
                                lambda_pop.append(pop_sum / prv_pop)
                            prv_pop = pop_sum.copy()
                        key = 'Population/{}/{}/Total'.format(species_name, group_name)
                        lcl_cmp[key] = da.concatenate(map(da.atleast_1d, gp_pop))
                        key = 'Population/{}/{}/Lambda'.format(species_name, group_name)
                        lcl_cmp[key] = da.concatenate(map(da.atleast_1d, lambda_pop))

                    # Collect average ages by gp
                    ave_ages = []
                    for time in model_times:
                        self.average_age(species, time, sex, gp)
                        m = self.to_compute[-1]
                        ave_ages.append(m[~da.isinf(m)].mean())
                    key = 'Population/{}/{}/Average {} Age'.format(species_name, gp, sex_str[0].upper() + sex_str[1:])
                    ave_ages[0] = np.nan
                    lcl_cmp[key] = da.concatenate(map(da.atleast_1d, ave_ages))

                    # Collect the population of this group and sex
                    gp_sex_pop = []
                    for time in model_times:
                        self.total_population(species, time, sex, gp)
                        gp_sex_pop.append(self.to_compute[-1].sum())
                    key = 'Population/{}/{}/{}s'.format(species_name, group_name, sex_str[0].upper() + sex_str[1:])
                    lcl_cmp[key] = da.concatenate(map(da.atleast_1d, gp_sex_pop))

                    # Calculate the F:M ratio if both sexes present
                    if ('Population/{}/{}/Males'.format(species_name, group_name) in lcl_cmp.keys() and
                        'Population/{}/{}/Females'.format(species_name, group_name) in lcl_cmp.keys()):
                        key = 'Natality/{}/{}/F:M Ratio'.format(species_name, group_name)
                        lcl_cmp[key] = da.where(
                            lcl_cmp['Population/{}/{}/Males'.format(species_name, group_name)] > 0,
                            lcl_cmp['Population/{}/{}/Females'.format(species_name, group_name)] /
                            lcl_cmp['Population/{}/{}/Males'.format(species_name, group_name)], np.inf
                        )

                    # Natality
                    # Offspring
                    ds = []
                    for time in model_times:
                        self.total_offspring(species, time, sex, gp)
                        ds.append(self.to_compute[-1].sum())
                    key = 'Natality/{}/{}/Total offspring'.format(species_name, group_name)
                    ds[0] = np.nan
                    lcl_cmp[key] = da.concatenate(map(da.atleast_1d, ds))
                    for __sex in ['male', 'female']:
                        offspring_sex = __sex[0].upper() + __sex[1:] + ' Offspring'
                        ds = []
                        for time in model_times:
                            self.total_offspring(species, time, sex, gp, offspring_sex=offspring_sex)
                            ds.append(self.to_compute[-1].sum())
                        key = 'Natality/{}/{}/{} offspring from {}s'.format(species_name, group_name,
                                                                            __sex[0].upper() + __sex[1:],
                                                                            sex_str[0].upper() + sex_str[1:])
                        ds[0] = np.nan
                        lcl_cmp[key] = da.concatenate(map(da.atleast_1d, ds))

                    if sex == 'female':
                        # Density coefficient
                        dd_fec_ds = []
                        for time in model_times:
                            self.fecundity(species, time, sex, gp, coeff=True)
                            dd_fec_ds.append(self.to_compute[-1].mean())
                        key = 'Natality/{}/{}/Density-Based Fecundity Reduction Rate'.format(species_name, group_name)
                        dd_fec_ds[0] = np.nan
                        lcl_cmp[key] = da.concatenate(map(da.atleast_1d, dd_fec_ds))

                        # Fecundity rate
                        ds = []
                        for time in model_times:
                            self.fecundity(species, time, sex, gp)
                            ds.append(self.to_compute[-1].mean())
                        key = 'Natality/{}/{}/{} mean fecundity'.format(species_name, group_name, sex_str)
                        ds[0] = np.nan
                        lcl_cmp[key] = da.concatenate(map(da.atleast_1d, ds))

                        # Offspring per female
                        key = 'Natality/{}/{}/offspring per female'.format(species_name, group_name)
                        lcl_cmp[key] = da.concatenate(
                            [np.array([np.nan]), da.where(
                                lcl_cmp['Population/{}/{}/Females'.format(species_name, group_name)][:-1] > 0,
                                lcl_cmp['Natality/{}/{}/Total offspring'.format(species_name, group_name)][1:] /
                                lcl_cmp['Population/{}/{}/Females'.format(species_name, group_name)][:-1], np.inf
                            )]
                        )

                    # Mortality
                    # Male/Female by group
                    mort_ds = []
                    for time in model_times:
                        self.total_mortality(species, time, sex, gp)
                        mort_ds.append(self.to_compute[-1].sum())
                    key = 'Mortality/{}/{}/{} deaths'.format(species_name, group_name,
                                                             sex_str[0].upper() + sex_str[1:])
                    mort_ds[0] = np.nan
                    lcl_cmp[key] = da.concatenate(map(da.atleast_1d, mort_ds))

                    # All for group
                    ds = []
                    for time in model_times:
                        self.total_mortality(species, time, None, gp)
                        ds.append(self.to_compute[-1].sum())
                    key = 'Mortality/{}/{}/Total deaths'.format(species_name, group_name)
                    ds[0] = np.nan
                    lcl_cmp[key] = da.concatenate(map(da.atleast_1d, ds))

                    mort_types = self.list_mortality_types(species, None, sex, gp)
                    for mort_type in mort_types:
                        if mort_type != 'Density Dependent Rate':
                            ds = []
                            for time in model_times:
                                self.total_mortality(species, time, sex, gp, mort_type)
                                ds.append(self.to_compute[-1].sum())

                            if 'Converted to ' in mort_type:
                                mort_str = '{} {}'.format(sex_str, mort_type)
                            else:
                                mort_str = '{} {} deaths'.format(sex_str, mort_type)
                            key = 'Mortality/{}/{}/{}'.format(species_name, group_name, mort_str)
                            ds[0] = np.nan
                            lcl_cmp[key] = da.concatenate(map(da.atleast_1d, ds))

                        # Skip the implicit mortality types, as they will not be included in the params
                        if mort_type in ['Old Age', 'Density Dependent'] or 'Converted to ' in mort_type:
                            continue

                        # Collect the parameter
                        ds = []
                        for time in model_times:
                            self.total_mortality(species, time, sex, gp, mort_type, False)
                            ds.append(self.to_compute[-1].mean())
                        key = 'Mortality/{}/{}/{} mean {} rate'.format(species_name, group_name,
                                                                       sex_str, mort_type)
                        if 'Density Dependent Rate' in key:
                            key = key[:-4]
                        ds[0] = np.nan
                        lcl_cmp[key] = da.concatenate(map(da.atleast_1d, ds))

            # Compute the summary
            keys, values = lcl_cmp.keys(), lcl_cmp.values()
            targets = [SummaryDataset(ds.shape, key=key) for key, ds in zip(keys, values)]

            # Optimize and compute the dask graph
            pd.store(values, targets)

            # Populate the species log
            for target in targets:
                _keys = target.key.split('/')
                try:
                    d = sp_log[_keys[0]]
                except KeyError:
                    sp_log[_keys[0]] = {}
                    d = sp_log[_keys[0]]

                for key in _keys[1:-1]:
                    try:
                        d = d[key]
                    except KeyError:
                        d[key] = {}
                        d = d[key]

                d[_keys[-1]] = target.value.tolist()

    @property
    def all_times(self):

        def _next(gp_cnt):
            group, cnt = gp_cnt
            # time is always 4 nodes down
            if cnt == 4:
                times.append(int(group.name.split('/')[-1]))
            else:
                for key in group.keys():
                    _next((group[key], cnt + 1))

        times = []

        for key in self.domain.file.keys():
            _next((self.domain.file[key], 1))

        return np.unique(times)

    def seek_instance(self, instance):
        types = ['species', 'mortality', 'carrying_capacity', 'fecundity']
        results = []

        def _next(d, _type):
            for key, val in d.items():
                if isinstance(val, dict):
                    _next(d, _type)
                elif instance is val:
                    results.append(val)

        for _type in types:
            _next(getattr(self.domain, _type), _type)

    def collect_iterables(self, species, time, sex, group, age='not provided'):

        def make_iter(obj):
            if any([isinstance(obj, o) for o in [tuple, list, set]]):
                return obj
            else:
                return [obj]

        # All times are used if time is None
        time = [t for t in make_iter(time) if t is not None]
        if len(time) == 0:
            time = self.all_times

        # All species are used if species is None
        species = [pd.name_key(sp) for sp in make_iter(species) if sp is not None]
        if len(species) == 0:
            species = [pd.name_key(name) for name in self.domain.species_names]

        # If sex is None, add both males and females
        sex = [s.lower() if s is not None else s for s in make_iter(sex)]
        if all([s is None for s in sex]):
            sex = [None, 'male', 'female']

        # Collect all groups if None
        group = [pd.name_key(gp) if gp is not None else gp for gp in make_iter(group)]
        if all([gp is None for gp in group]):
            group = []
            for sp in species:
                group += list(self.domain._group_keys(sp))

        if age == 'not provided':
            return species, time, sex, group
        else:
            return species, time, sex, group, [age for age in make_iter(age) if age is not None]


def total_population(domain, species=None, time=None, sex=None, group=None, age=None):
    ms = ModelSummary(domain)
    ms.total_population(species, time, sex, group, age)
    return ms.compute()[-1]


def average_age(domain, species=None, time=None, sex=None, group=None):
    ms = ModelSummary(domain)
    ms.average_age(species, time, sex, group)
    return ms.compute()[-1]


def total_carrying_capacity(domain, species=None, time=None, sex=None, group=None):
    ms = ModelSummary(domain)
    ms.total_carrying_capacity(species, time, sex, group)
    return ms.compute()[-1]


def total_mortality(domain, species=None, time=None, sex=None, group=None, mortality_name=None, as_population=True):
    ms = ModelSummary(domain)
    ms.total_mortality(species, time, sex, group, mortality_name, as_population)
    return ms.compute()[-1]


def total_offspring(domain, species=None, time=None, sex=None, group=None, offspring_sex=None):
    ms = ModelSummary(domain)
    ms.total_offspring(species, time, sex, group, offspring_sex)
    return ms.compute()[-1]


def fecundity(domain, species=None, time=None, sex=None, group=None, coeff=False):
    ms = ModelSummary(domain)
    ms.fecundity(species, time, sex, group, coeff)
    return ms.compute()[-1]


