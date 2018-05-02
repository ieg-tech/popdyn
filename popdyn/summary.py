"""
Summarize data from a PopDyn simulation

Devin Cairns 2018
"""

from popdyn import *
import dask.array as da
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
                        ages = domain.age_from_group(sp, s, gp)
                    for age in ages:
                        population = domain.get_population(sp, t, s, gp, age)
                        if population is not None:
                            populations.append(
                                da.from_array(domain[populations], domain.chunks)
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
                        mortality.append(da.from_array(domain['{}/{}'.format(base_key, name)], domain.chunks))

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

    fecundity = []
    for t in time:
        for sp in species:
            for s in sex:
                for gp in group:
                    fec = '{}/{}/{}/{}/params/fecundity/Fecundity'.format(sp, s, gp, t)
                    try:
                        fecundity.append(da.from_array(domain[fec], domain.chunks))
                    except KeyError:
                        pass

    if len(fecundity) == 0:
        return np.zeros(domain.shape, np.float32)
    else:
        return da.dstack(fecundity).sum(axis=-1).compute()


def model_summary(domain):
    """
    Summarize totals of each species and their parameters in the domain
    :param domain: Domain instance
    :return: dict of species and their parameters
    """
    log = {
        'Habitat': {}, 'Population': {}, 'Natality': {}, 'Mortality': {},
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

    model_times = all_times(domain)

    for species in summary.keys():
        sp_log = summary[species]

        for time in model_times:
            # Carrying Capacity
            domain.file[species]

            # Natality


            # Mortality


            # Population


    return log


def all_times(domain):

    def _next(gp_cnt):
        group, cnt = gp_cnt
        # time is always 4 nodes down
        if cnt == 4:
            times.append(int(group[0].name))
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


def name_key(name):
    return name.strip().translate(None, punctuation + ' ').lower()


def collect_iterables(domain, species, time, sex, group, age='not provided'):

    def make_iter(obj):
        if any([isinstance(obj, o) for o in [tuple, list, set]]):
            return obj
        else:
            return [obj]

    if not isinstance(domain, Domain):
        raise TypeError('The domain input must be a Domain instance')

    # All times are used if time is None
    time = [t for t in make_iter(time) if t is not None]
    if len(time) == 0:
        time = all_times(domain)

    # All species are used if species is None
    species = [name_key(sp) for sp in make_iter(species) if sp is not None]
    if len(species) == 0:
        species = [name_key(name) for name in domain.species_names]

    # If sex is None, add both males and females
    sex = make_iter(sex)
    if all([s is None for s in sex]):
        sex = [None, 'male', 'female']

    # Collect all groups if None
    group = make_iter(group)
    if all([gp is None for gp in group]):
        group = [None]
        for sp in species:
            group += list(domain.group_keys(sp))

    if age == 'not provided':
        return species, time, sex, group
    else:
        return species, time, sex, group, [age for age in make_iter(age) if age is not None]
