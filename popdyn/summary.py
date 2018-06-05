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
        return np.zeros(shape=domain.shape, dtype=np.float32)
    else:
        return pd.dstack(populations).sum(axis=-1).compute()


def average_age(domain, species=None, time=None, sex=None, group=None):
    """
    Collect the average age from a domain, filtering by sub-populations

    :param Domain domain: Domain object
    :param str species: Species name - may be an iterable
    :param time: time slice: int or iterable
    :param sex: Sex (one of 'male' or 'female') - may be an iterable
    :param group: group name - may be an iterable\
    :return: ndarray with the total population
    """
    species, time, sex, group = collect_iterables(domain, species, time, sex, group)

    ages = {}
    for t in time:
        for sp in species:
            for s in sex:
                for gp in group:
                    try:
                        _ages = domain.age_from_group(sp, s, gp)  # Returns a list
                    except pd.PopdynError:
                        # This group is not associated with the current species or sex in the iteration
                        continue
                    for age in _ages:
                        population = domain.get_population(sp, t, s, gp, age)
                        if population is not None:
                            ds = da.from_array(domain[population], domain.chunks)
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
        return da.where(pops_sum > 0, ages.keys() * (pops / pops_sum), np.inf).sum(axis=-1).compute()


def total_carrying_capacity(domain, species=None, time=None, sex=None, group=None):
    """
    Collect the total carrying capacity in the domain for the given query.

    Note, carrying capacity for individual sexes or species may be collected for the species as a whole,

    # TODO: Raise exception if specified group or sex does not exist, as it will still return results with typos
    which will result in species-wide carrying capacity being returned.
    :param Domain domain: Domain object
    :param str species: Species name
    :param time: time slice - may be an iterable or int
    :param str sex: Sex ('male' or 'female') - may be an iterable
    :param str group: Group name - may be an iterable
    :return: ndarray of total carrying capacity
    """

    def retrieve(domain, species, time, sex, group):
        species, time, sex, group = collect_iterables(domain, species, time, sex, group)

        for t in time:
            for sp in species:
                for s in sex:
                    for gp in group:
                        cc = '{}/{}/{}/{}/params/carrying capacity/Carrying Capacity'.format(sp, s, gp, t)
                        try:
                            carrying_capacity.append(da.from_array(domain[cc], domain.chunks))
                        except KeyError:
                            pass

    carrying_capacity = []
    retrieve(domain, species, time, sex, group)

    # If the total species carrying capacity was used for density during the simulation,
    #  there will not be any carrying capacity associated with sexes/groups.
    if len(carrying_capacity) == 0 and (sex is not None or group is not None):
        # Try to collect the species-wide dataset
        sex, group = None, None
        carrying_capacity = []
        retrieve(domain, species, time, sex, group)

    if len(carrying_capacity) == 0:
        return np.zeros(shape=domain.shape, dtype=np.float32)
    else:
        return pd.dstack(carrying_capacity).sum(axis=-1).compute()


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
                        names += domain.file['{}/{}/{}/{}/flux/mortality'.format(sp, s, gp, t)].keys()
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
        _type = 'params'

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
        return np.zeros(shape=domain.shape, dtype=np.float32)
    else:
        return pd.dstack(mortality).sum(axis=-1).compute()


def total_offspring(domain, species=None, time=None, sex=None, group=None, offspring_sex=None):
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
                            offspring.append(da.from_array(domain[off], domain.chunks))
                        except KeyError:
                            pass

    if len(offspring) == 0:
        return np.zeros(shape=domain.shape, dtype=np.float32)
    else:
        return pd.dstack(offspring).sum(axis=-1).compute()


def fecundity(domain, species=None, time=None, sex=None, group=None, coeff=False):
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
                        _fecundity.append(da.from_array(domain[fec], domain.chunks))
                    except KeyError:
                        pass

    if len(_fecundity) == 0:
        return np.zeros(shape=domain.shape, dtype=np.float32)
    else:
        if coeff:
            return pd.dstack(_fecundity).mean(axis=-1).compute()
        else:
            return pd.dstack(_fecundity).sum(axis=-1).compute()


def model_summary(domain):
    """
    Summarize totals of each species and their parameters in the domain

    TODO: This is built to service the requirements of popdyn.logger.write_xlsx, and could use
    TODO: a substantial amount of optimization

    :param domain: Domain instance
    :return: dict of species and their parameters
    """
    model_times = all_times(domain)

    log = {'Time': model_times,
        'Habitat': {}, 'Population': pd.rec_dd(), 'Natality': pd.rec_dd(), 'Mortality': pd.rec_dd(),
        'Parameterization': {'Domain size': str(domain.shape),
                             'Cell size (x)': domain.csx,
                             'Cell size (y)': domain.csy,
                             'Top corner': domain.top,
                             'Left corner': domain.left,
                             'Distributed scheduler chunk size': domain.chunks,
                             'Popdyn file': domain.path,
                             'Spatial Reference': domain.projection,
                             'Avoid Inheritance': domain.avoid_inheritance,
                             'Age Groups': {}},
        'Solver': [datetime.now(tzlocal()).strftime('%A, %B %d, %Y %I:%M%p %Z')] + \
                  ['{},{:.4f}'.format(key, val) for key, val in domain.profiler.items()]
    }

    summary = {sp: deepcopy(log) for sp in domain.species.keys()}

    for species in summary.keys():
        # Collect the species name
        try:
            species_name = [name for name in domain.species_names if species == pd.name_key(name)][0]
        except IndexError:
            raise pd.PopdynError('Unable to gather the species name from the key {}'.format(species))

        sp_log = summary[species]

        # Carrying Capacity NOTE: This should be summarized by group in the future
        ds = []
        change_ds = []
        first_cc = None
        cc_mean_zero = ['']
        cc_mean_nonzero = ['']
        for time in model_times:
            cc_a = total_carrying_capacity(domain, species, time)
            total_cc = cc_a.sum()
            if time == model_times[0]:
                change_ds.append('')
            else:
                cc_a_mean = cc_a.mean() / (domain.csx * domain.csy * 1E6)  # Assuming metres
                cc_mean_zero.append(cc_a_mean)
                cc_mean_nonzero.append(cc_a[cc_a != 0].mean() / (domain.csx * domain.csy * 1E6)
                                       if not np.all(cc_a == 0) else 0)
                if first_cc is None:
                    first_cc = cc_a.mean() / (domain.csx * domain.csy * 1E6)
                if first_cc > 0:
                    change_ds.append(cc_a_mean / first_cc)
                else:
                    change_ds.append(0.)
            ds.append(total_cc)
        sp_log['Habitat'][species_name] = {'Total n': ds, 'Relative Change': change_ds,
                                           'Mean [including zeros] (n/km^2)': cc_mean_zero,
                                           'Mean [excluding zeros] (n/km^2)': cc_mean_nonzero}

        # Collect average ages
        ave_ages = []
        for time in model_times:
            m = average_age(domain, species, time)
            not_inf = ~np.isinf(m)
            if not_inf.sum() > 0:
                ave_ages.append(m[not_inf].mean())
            else:
                ave_ages.append(0)  # This should be something else, because age 0 could exist
        sp_log['Population'][species_name]['NA']['Average Age'] = ave_ages

        # Add total population and lambda population for species
        total_pop = []
        lambda_pop = []
        for time in model_times:
            pop_sum = total_population(domain, species, time).sum()
            total_pop.append(pop_sum)
            if time == model_times[0]:
                lambda_pop.append(1.)
            else:
                if prv_pop != 0:
                    lambda_pop.append(pop_sum / prv_pop)
                else:
                    lambda_pop.append('')
            prv_pop = pop_sum
        sp_log['Population'][species_name]['NA']['Total Population'] = total_pop
        sp_log['Population'][species_name]['NA']['Total Population Lambda'] = lambda_pop

        # Collect all offspring
        tot_new_off = []
        for time in model_times:
            tot_new_off.append(total_offspring(domain, species, time).sum())
        sp_log['Natality'][species_name]['NA']['Total new offspring'] = tot_new_off

        # Collect all deaths
        all_deaths = []
        for time in model_times:
            all_deaths.append(total_mortality(domain, species, time).sum())
        sp_log['Mortality'][species_name]['NA']['All deaths'] = all_deaths

        # Collect deaths by type
        mort_types = list_mortality_types(domain, species, None)
        for mort_type in mort_types:
            ds = []
            for time in model_times:
                ds.append(total_mortality(domain, species, time, mortality_name=mort_type).sum())
            sp_log['Mortality'][species_name]['NA']['Total deaths from {}'.format(mort_type)] = ds

        # Iterate groups and populate data
        for sex in ['male', 'female']:
            sp_log['Parameterization']['Age Groups'][sex] = []
            sex_str = sex

            # Collect total population by sex
            sex_pop = []
            for time in model_times:
                sex_pop.append(total_population(domain, species, time, sex).sum())
            sp_log['Population'][species_name]['NA']['Total {}s'.format(sex_str[0].upper() + sex_str[1:])] = sex_pop

            if sex == 'female':
                # Collect offspring / female
                sp_log['Natality'][species_name]['NA']['Total offspring per female'] = np.where(
                    np.array(sp_log['Population'][species_name]['NA']['Total Females']) > 0,
                    np.array(sp_log['Natality'][species_name]['NA']['Total new offspring']) /
                    np.array(sp_log['Population'][species_name]['NA']['Total Females']), np.inf
                )

            # Calculate the F:M ratio if both sexes present
            if all(['Total {}s'.format(_sex) in sp_log['Population'][species_name]['NA'].keys()
                    for _sex  in ['Male', 'Female']]):
                sp_log['Natality'][species_name]['NA']['F:M Ratio'] = \
                    np.where(sp_log['Population'][species_name]['NA']['Total Males'] > 0,
                             np.array(sp_log['Population'][species_name]['NA']['Total Females']) / \
                             np.array(sp_log['Population'][species_name]['NA']['Total Males']), np.inf).tolist()

            # Collect average ages by sex
            ave_ages = []
            for time in model_times:
                m = average_age(domain, species, time, sex)
                ave_ages.append(m[~np.isinf(m)].mean())
            sp_log['Population'][species_name]['NA'][
                'Average {} Age'.format(sex_str[0].upper() + sex_str[1:])
            ] = ave_ages

            # Offspring by sex
            offspring_sex = sex[0].upper() + sex[1:] + ' Offspring'
            ds = []
            for time in model_times:
                ds.append(total_offspring(domain, species, time, sex, offspring_sex=offspring_sex).sum())
            sp_log['Natality'][species_name]['NA']['{} offspring'.format(sex_str[0].upper() + sex_str[1:])] = ds

            # Collect deaths by sex
            sex_death = []
            for time in model_times:
                sex_death.append(total_mortality(domain, species, time, sex).sum())
            sp_log['Mortality'][species_name]['NA']['Total {} deaths'.format(sex)] = sex_death

            # Collect deaths by type/sex
            mort_types = list_mortality_types(domain, species, None, sex)
            for mort_type in mort_types:
                ds = []
                for time in model_times:
                    ds.append(total_mortality(domain, species, time, sex, mortality_name=mort_type).sum())
                sp_log['Mortality'][species_name]['NA']['{} deaths from {}'.format(sex_str[0].upper() + sex_str[1:],
                                                                                   mort_type)] = ds

            for gp in domain.group_keys(species):

                if gp is None:
                    continue

                try:
                    group_name = [name for name in domain.group_names(species) if gp == pd.name_key(name)][0]
                except IndexError:
                    raise pd.PopdynError('Unable to gather the group name from the key {}'.format(gp))
                if sex is not None:
                    sp_log['Parameterization']['Age Groups'][sex].append(group_name)

                # Collect the total population of the group, which is only needed once
                if 'Total' not in sp_log['Population'][species_name][group_name].keys():
                    gp_pop = []
                    lambda_pop = []
                    for time in model_times:
                        pop_sum = total_population(domain, species, time, group=gp).sum()
                        gp_pop.append(pop_sum)
                        if time == model_times[0]:
                            lambda_pop.append(1.)
                        else:
                            if prv_pop != 0:
                                lambda_pop.append(pop_sum / prv_pop)
                            else:
                                lambda_pop.append('')
                        prv_pop = pop_sum
                    sp_log['Population'][species_name][group_name]['Total'] = gp_pop
                    sp_log['Population'][species_name][group_name]['Lambda'] = lambda_pop

                # Collect average ages by gp
                ave_ages = []
                for time in model_times:
                    m = average_age(domain, species, time, sex, gp)
                    ave_ages.append(m[~np.isinf(m)].mean())
                sp_log['Population'][species_name][gp][
                    'Average {} Age'.format(sex_str[0].upper() + sex_str[1:])
                ] = ave_ages

                # Collect the population of this group and sex
                gp_sex_pop = []
                for time in model_times:
                    gp_sex_pop.append(total_population(domain, species, time, sex, gp).sum())
                sp_log['Population'][species_name][group_name][
                    '{}s'.format(sex_str[0].upper() + sex_str[1:])
                ] = gp_sex_pop

                # Calculate the F:M ratio if both sexes present
                if all(['{}s'.format(_sex) in sp_log['Population'][species_name][group_name].keys()
                        for _sex in ['Male', 'Female']]):
                    sp_log['Natality'][species_name][group_name]['F:M Ratio'] = \
                        np.where(sp_log['Population'][species_name][group_name]['Males'] > 0,
                                 np.array(sp_log['Population'][species_name][group_name]['Females']) / \
                                 np.array(sp_log['Population'][species_name][group_name]['Males']),
                                 np.inf).tolist()

                # Natality
                # Offspring
                offspring_sex = sex[0].upper() + sex[1:] + ' Offspring'
                ds = []
                for time in model_times:
                    ds.append(total_offspring(domain, species, time, sex, gp).sum())
                sp_log['Natality'][species_name][group_name]['Total offspring'] = ds
                ds = []
                for time in model_times:
                    ds.append(total_offspring(domain, species, time, sex, gp, offspring_sex=offspring_sex).sum())
                sp_log['Natality'][species_name][group_name]['{} offspring'.format(sex_str[0].upper() + sex_str[1:])] = ds

                if sex == 'female':
                    # Density coefficient
                    dd_fec_ds = []
                    for time in model_times:
                        dd_fec_ds.append(fecundity(domain, species, time, sex, gp, coeff=True).mean())
                    sp_log['Natality'][species_name][group_name]['Density-Based Fecundity Reduction Rate'] = dd_fec_ds

                    # Fecundity rate
                    ds = []
                    for time in model_times:
                        ds.append(fecundity(domain, species, time, sex, gp).mean())
                    sp_log['Natality'][species_name][group_name]['{} mean fecundity'.format(sex_str)] = ds

                    # Offspring per female
                    sp_log['Natality'][species_name][group_name]['offspring per female'] = np.where(
                        np.array(sp_log['Population'][species_name][group_name]['Females']) > 0,
                        np.array(sp_log['Natality'][species_name][group_name]['Total offspring']) /
                        np.array(sp_log['Population'][species_name][group_name]['Females']), np.inf
                    )

                # Mortality
                # Male/Female by group
                mort_ds = []
                for time in model_times:
                    mort_ds.append(total_mortality(domain, species, time, sex, gp).sum())
                sp_log['Mortality'][species_name][group_name][
                    '{} deaths'.format(sex_str[0].upper() + sex_str[1:])
                ] = mort_ds

                # All for group
                ds = []
                for time in model_times:
                    ds.append(total_mortality(domain, species, time, None, gp).sum())
                sp_log['Mortality'][species_name][group_name]['Total deaths'] = ds

                # Survivorship
                if 'Survivorship'.format(group_name) not in sp_log['Mortality'][species_name][group_name].keys():
                    srv_pop = []
                    for time_ind, time in enumerate(model_times):
                        if time == model_times[0]:
                            srv_pop.append(0)
                            continue
                        prv_pop = total_population(domain, species, time - 1, group=gp).sum()
                        if prv_pop == 0:
                            srv_pop.append(0)
                            continue
                        srv_pop.append((prv_pop - mort_ds[time_ind]) / prv_pop)
                    sp_log['Mortality'][species_name][group_name]['Survivorship'] = srv_pop

                mort_types = list_mortality_types(domain, species, None, sex, gp)
                for mort_type in mort_types:
                    ds = []
                    for time in model_times:
                        ds.append(total_mortality(domain, species, time, sex, gp, mort_type).sum())

                    if 'Converted to ' in mort_type:
                        mort_str = '{} {}'.format(sex_str, mort_type)
                    else:
                        mort_str = '{} {} deaths'.format(sex_str, mort_type)
                    sp_log['Mortality'][species_name][group_name][mort_str] = ds

                    # Skip the implicit mortality types, as they will not be included in the params
                    if mort_type in ['Old Age', 'Density Dependent'] or 'Converted to ' in mort_type:
                        continue

                    # Collect the parameter
                    if 'Converted to ' not in mort_type:
                        ds = []
                        for time in model_times:
                            ds.append(total_mortality(domain, species, time, sex, gp, mort_type, False).mean())
                        sp_log['Mortality'][species_name][group_name][
                            '{} mean {} rate'.format(sex_str, mort_type)] = ds

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