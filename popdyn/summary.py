"""
Summarize data from a PopDyn simulation

Devin Cairns 2018
"""

import popdyn as pd
import dask.array as da
import numpy as np
from datetime import datetime
from dateutil.tz import tzlocal
from copy import deepcopy


def total_population(domain, species=None, time=None, sex=None, group=None, age=None):
    """
    Collect the sum of populations from a domain, filtering by sub-populations
    :param Domain domain: Domain object
    :param str species: Species name - may be an iterable
    :param time: time slice: int or iterable
    :param sex: Sex (one of 'male' or 'female') - may be an iterable
    :param group: group name - may be an iterable
    :param age: discrete Age - may be an iterable
    :return: ndarray with the total population
    """
    species, time, sex, group, ages = collect_iterables(domain, species, time, sex, group, age)

    populations = []
    for t in time:
        for sp in species:
            for s in sex:
                for gp in group:
                    # If ages are None, they must be collected for the group
                    if len(ages) == 0:
                        try:
                            _ages = domain.age_from_group(sp, s, gp)  # Returns a list
                        except pd.PopdynError:
                            # This group is not associated with the current species or sex in the iteration
                            continue
                    else:
                        _ages = ages
                    for age in _ages:
                        population = domain.get_population(sp, t, s, gp, age)
                        if population is not None:
                            populations.append(
                                da.from_array(domain[population], domain.chunks)
                            )

    if len(populations) == 0:
        return np.zeros(domain.shape, np.float32)
    else:
        return da.dstack(populations).sum(axis=-1).compute()


def total_carrying_capacity(domain, species=None, time=None, sex=None, group=None):
    """
    Collect the total carrying capacity in the domain for the given query
    :param Domain domain: Domain object
    :param str species: Species name
    :param time: time slice - may be an iterable or int
    :param str sex: Sex ('male' or 'female') - may be an iterable
    :param str group: Group name - may be an iterable
    :return: ndarray of total carrying capacity
    """
    species, time, sex, group = collect_iterables(domain, species, time, sex, group)

    carrying_capacity = []
    for t in time:
        for sp in species:
            for s in sex:
                for gp in group:
                    cc = '{}/{}/{}/{}/params/carrying capacity/Carrying Capacity'.format(sp, s, gp, t)
                    try:
                        carrying_capacity.append(da.from_array(domain[cc], domain.chunks))
                    except KeyError:
                        pass

    if len(carrying_capacity) == 0:
        return np.zeros(domain.shape, np.float32)
    else:
        return da.dstack(carrying_capacity).sum(axis=-1).compute()


def list_mortality_types(domain, species=None, time=None, sex=None, group=None):
    """
    List the mortality names in the domain for the given query
    :param Domain domain: Domain object
    :param str species: Species name
    :param time: time slice - may be an iterable or int
    :param str sex: Sex ('male' or 'female') - may be an iterable
    :param str group: Group name - may be an iterable
    :return: 1D str array of unique names
    """
    species, time, sex, group = collect_iterables(domain, species, time, sex, group)

    names = []
    for t in time:
        for sp in species:
            for s in sex:
                for gp in group:
                    try:
                        names += domain.file['{}/{}/{}/{}/params/mortality'.format(sp, s, gp, t)].keys()
                    except KeyError:
                        pass

    return np.unique(names)


def total_mortality(domain, species=None, time=None, sex=None, group=None, mortality_name=None, as_population=True):
    """
    Collect either the total population or the mortality parameter value for the given query
    :param Domain domain: Domain object
    :param str species: Species name
    :param time: time slice - may be an iterable or int
    :param str sex: Sex ('male' or 'female') - may be an iterable
    :param str group: Group name - may be an iterable
    :param bool as_population: Determines whether a population killed, or the mortality parameter is returned
    :return: ndarray of total population or the mortality parameter value depending on as_population
    """
    species, time, sex, group = collect_iterables(domain, species, time, sex, group)

    if as_population:
        _type = 'flux'
    else:
        _type = 'param'

    mortality = []
    for t in time:
        for sp in species:
            for s in sex:
                for gp in group:
                    base_key = '{}/{}/{}/{}/{}/mortality'.format(sp, s, gp, t, _type)
                    if mortality_name is None:
                        try:
                            names = domain.file[base_key].keys()
                        except KeyError:
                            continue
                    else:
                        names = [mortality_name]
                    for name in names:
                        try:
                            ds = domain['{}/{}'.format(base_key, name)]
                        except KeyError:
                            continue
                        mortality.append(da.from_array(ds, domain.chunks))

    if len(mortality) == 0:
        return np.zeros(domain.shape, np.float32)
    else:
        return da.dstack(mortality).sum(axis=-1).compute()


def total_offspring(domain, species=None, time=None, sex=None, group=None):
    """
    Collect the total offspring using the given query
    :param Domain domain: Domain object
    :param str species: Species name
    :param time: time slice - may be an iterable or int
    :param str sex: Sex ('male' or 'female') - may be an iterable
    :param str group: Group name - may be an iterable
    :return: ndarray of total offspring
    """
    species, time, sex, group = collect_iterables(domain, species, time, sex, group)

    offspring = []
    for t in time:
        for sp in species:
            for s in sex:
                for gp in group:
                    for _type in ['Male Offspring', 'Female Offspring']:
                        off = '{}/{}/{}/{}/flux/offspring/{}'.format(sp, s, gp, t, _type)
                        try:
                            offspring.append(da.from_array(domain[off], domain.chunks))
                        except KeyError:
                            pass

    if len(offspring) == 0:
        return np.zeros(domain.shape, np.float32)
    else:
        return da.dstack(offspring).sum(axis=-1).compute()


def fecundity(domain, species=None, time=None, sex=None, group=None):
    """
    Collect the total fecundity using the given query
    :param Domain domain: Domain object
    :param str species: Species name
    :param time: time slice - may be an iterable or int
    :param str sex: Sex ('male' or 'female') - may be an iterable
    :param str group: Group name - may be an iterable
    :return: ndarray of total offspring
    """
    species, time, sex, group = collect_iterables(domain, species, time, sex, group)

    _fecundity = []
    for t in time:
        for sp in species:
            for s in sex:
                for gp in group:
                    fec = '{}/{}/{}/{}/params/fecundity/Fecundity'.format(sp, s, gp, t)
                    try:
                        _fecundity.append(da.from_array(domain[fec], domain.chunks))
                    except KeyError:
                        pass

    if len(_fecundity) == 0:
        return np.zeros(domain.shape, np.float32)
    else:
        return da.dstack(_fecundity).sum(axis=-1).compute()


def model_summary(domain):
    """
    Summarize totals of each species and their parameters in the domain
    :param domain: Domain instance
    :return: dict of species and their parameters
    """
    model_times = all_times(domain)

    log = {'Time': model_times,
        'Habitat': pd.rec_dd(), 'Population': pd.rec_dd(), 'Natality': pd.rec_dd(), 'Mortality': pd.rec_dd(),
        'Parameterization': {'Domain size': str(domain.shape),
                             'Cell size (x)': domain.csx,
                             'Cell size (y)': domain.csy,
                             'Top corner': domain.top,
                             'Left corner': domain.left,
                             'Distributed scheduler chunk size': domain.chunks,
                             'Popdyn file': domain.path,
                             'Spatial Reference': domain.projection,
                             'Avoid Inheritance': domain.avoid_inheritance},
        'Solver': [datetime.now(tzlocal()).strftime('%A, %B %d, %Y %I:%M%p %Z')] + \
                  ['{},{:.4f}'.format(key, val) for key, val in domain.profiler.items()]
    }

    summary = {sp: deepcopy(log) for sp in domain.species.keys()}

    def add(top_key, log, key, name):
        try:
            ds = da.from_array(domain[key], chunks=domain.chunks)
        except KeyError:
            ds = ''

        # Take the mean if it's a mortality parameter
        if top_key == 'Mortality' and key.split('/')[-1] == 'param':
            ds = ds.mean()
        else:
            ds = ds.sum()
        try:
            log[top_key][species_name]['NA'][name].append(ds)
        except KeyError:
            log[top_key][species_name]['NA'][name] = [ds]

    for species in summary.keys():
        sp_log = summary[species]

        # Collect the species name
        try:
            species_name = [name for name in domain.species_names if species == pd.name_key(name)][0]
        except IndexError:
            raise pd.PopdynError('Unable to gather the species name from the key {}'.format(species))

        # Add total population for species
        total_pop = []
        for time in model_times:
            total_pop.append(total_population(domain, species, time).sum())
        sp_log['Population'][species_name]['NA']['Total'] = total_pop

        # Iterate groups and populate data
        for sex in ['male', 'female', None]:
            # Collect total population by sex
            sex_pop = []
            for time in model_times:
                sex_pop.append(total_population(domain, species, time, sex).sum())
            sp_log['Population'][species_name]['NA']['Total {}s'.format(sex[0].upper() + sex[1:])] = sex_pop

            for gp in domain.group_keys(species):
                if gp is None:
                    group_name = 'NA'
                else:
                    try:
                        group_name = [name for name in domain.group_names(species) if gp == pd.name_key(name)][0]
                    except IndexError:
                        raise pd.PopdynError('Unable to gather the group name from the key {}'.format(gp))

                name = '{} {}'.format(gp, sex)

                for param in ['param', 'flux']:
                    for time in model_times:
                        key = '{}/{}/{}/{}/{}'.format(species, sex, gp, time, param)

                        # Carrying Capacity
                        _key = '{}/carrying capacity/Carrying Capacity'.format(key)
                        add('Habitat', sp_log, _key, name + ' Carrying Capacity')

                        # Natality
                        nat_types = ['']
                        for nat_type in nat_types:
                            _key = '{}/fecundity/{}'.format(key, nat_type)
                            add('Natality', sp_log, _key, name + ' ' + nat_type)

                        # Mortality
                        mort_types = ['']
                        for mort_type in mort_types:
                            _key = '{}/mortality/{}'.format(key, mort_type)
                            add('Mortality', sp_log, _key, name + ' ' + mort_type + ' Deaths')

                # Total population for the group, only needed once
                if 'Total' not in sp_log['Population'][species_name][group_name].keys():
                    gp_pop = []
                    for time in model_times:
                        gp_pop.append(total_population(domain, species, time, group=gp).sum())
                    sp_log['Population'][species_name][group_name]['Total'] = gp_pop

                gp_sex_pop = []
                for time in model_times:
                    gp_sex_pop.append(total_population(domain, species, time, sex, gp).sum())
                sp_log['Population'][species_name][group_name]['{}s'.format(sex[0].upper() + sex[1:])] = gp_sex_pop

    return summary


def all_times(domain):

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


def seek_instance(domain, instance):
    types = ['species', 'mortality', 'carrying_capacity', 'fecundity']
    results = []

    def _next(d, _type):
        for key, val in d.items():
            if isinstance(val, dict):
                _next(d, _type)
            elif instance is val:
                results.append(val)

    for _type in types:
        _next(getattr(domain, _type), _type)


def collect_iterables(domain, species, time, sex, group, age='not provided'):

    def make_iter(obj):
        if any([isinstance(obj, o) for o in [tuple, list, set]]):
            return obj
        else:
            return [obj]

    if not isinstance(domain, pd.Domain):
        raise TypeError('The domain input must be a Domain instance')

    # All times are used if time is None
    time = [t for t in make_iter(time) if t is not None]
    if len(time) == 0:
        time = all_times(domain)

    # All species are used if species is None
    species = [pd.name_key(sp) for sp in make_iter(species) if sp is not None]
    if len(species) == 0:
        species = [pd.name_key(name) for name in domain.species_names]

    # If sex is None, add both males and females
    sex = [s.lower() if s is not None else s for s in make_iter(sex)]
    if all([s is None for s in sex]):
        sex = [None, 'male', 'female']

    # Collect all groups if None
    group = [pd.name_key(gp) if gp is not None else gp for gp in make_iter(group)]
    if all([gp is None for gp in group]):
        group = []
        for sp in species:
            group += list(domain.group_keys(sp))

    if age == 'not provided':
        return species, time, sex, group
    else:
        return species, time, sex, group, [age for age in make_iter(age) if age is not None]
