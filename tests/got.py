"""
Population dynamics module tests
"""


import os
import numpy as np
from scipy.ndimage import gaussian_filter
import popdyn as pd
from popdyn import summary
import shutil
import time


def some_random_k(shape, factor):
    k = np.zeros(shape=shape, dtype='float32')
    np.random.seed(2);
    i = np.random.randint(0, shape[0], 5)
    np.random.seed(2);
    j = np.random.randint(0, shape[1], 5)
    k[(i, j)] = 100000
    k = gaussian_filter(k, 10)
    np.random.seed(0);
    k += np.random.normal(0, 1000, shape)
    k = gaussian_filter(k, 10) * factor
    k[k < 0] = 0
    return k


# Domain
shape = (90, 160)


# Species
#================================================================================

# Stark
starks = pd.Species('Stark')

stark_male_infant = pd.AgeGroup('Stark', 'Infant', 'male', 0, 3)
stark_female_infant = pd.AgeGroup('Stark', 'Infant', 'female', 0, 3)

stark_male_adolescent = pd.AgeGroup('Stark', 'Adolescent', 'male', 4, 12)
stark_female_adolescent = pd.AgeGroup('Stark', 'Adolescent', 'female', 4, 12)

stark_male_adult = pd.AgeGroup('Stark', 'Adult', 'male', 13, 50)
stark_female_adult = pd.AgeGroup('Stark', 'Adult', 'female', 13, 50)

# Lannister
lannister = pd.Species('Lannister')

lannister_male_infant = pd.AgeGroup('Lannister', 'Infant', 'male', 0, 3)
lannister_female_infant = pd.AgeGroup('Lannister', 'Infant', 'female', 0, 3)

lannister_male_adolescent = pd.AgeGroup('Lannister', 'Adolescent', 'male', 4, 12)
lannister_female_adolescent = pd.AgeGroup('Lannister', 'Adolescent', 'female', 4, 12)

lannister_male_adult = pd.AgeGroup('Lannister', 'Adult', 'male', 13, 50)
lannister_female_adult = pd.AgeGroup('Lannister', 'Adult', 'female', 13, 50)

# White walker
white_walker = pd.Species('White Walker')

# Mortality
#================================================================================

# For humans
disease = pd.Mortality('Disease')
accident = pd.Mortality('Accident')
white_walker_death = pd.Mortality('White Walker')
white_walker_death.add_as_species(white_walker, [(0, 0), (0.1, 0.1), (0.5, 0.8), (1, 0.9)])

# Stark soldiers
lannister_male_soldier = pd.Mortality('Lannister Male')
lannister_male_soldier.add_as_species(lannister_male_adult, [(0, 0), (0.2, 0.05), (1, 0.1)])
lannister_female_soldier = pd.Mortality('Lannister Female')
lannister_female_soldier.add_as_species(lannister_female_adult, [(0, 0), (0.2, 0.01), (1, 0.05)])

# Lannister soldiers
stark_male_soldier = pd.Mortality('Stark Male')
stark_male_soldier.add_as_species(stark_male_adult, [(0, 0), (0.2, 0.05), (1, 0.1)])
stark_female_soldier = pd.Mortality('Stark Female')
stark_female_soldier.add_as_species(stark_male_adult, [(0, 0), (0.2, 0.05), (1, 0.1)])

# Carrying capacity
#================================================================================

stark_k_data = some_random_k(shape, 1.)
stark_k = pd.CarryingCapacity('Stark Habitat')
lannister_k_data = some_random_k(shape, 1.2)
lannister_k = pd.CarryingCapacity('Lannister Habitat')
white_walker_k = pd.CarryingCapacity('White Walker Habitat')
white_walker_k_data = some_random_k(shape, 2.)

# Fecundity
#================================================================================
male_fecundity = pd.Fecundity('males', multiplies=False)
female_fecundity = pd.Fecundity('females')

# Tests
#================================================================================

def no_species():
    """Should raise an exception in the solver error checker"""
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=1, left=0) as domain:
        pd.solvers.discrete_explicit(domain, 0, 1).execute()


def incorrect_ages():
    """Should raise an exception in the solver error checker"""
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=1, left=0) as domain:
        stark_female_infant = pd.AgeGroup('Stark', 'Infant', 'female', 0, 3)
        stark_female_adolescent = pd.AgeGroup('Stark', 'Adolescent', 'female', 3, 12)  # Age range overlaps infants
        domain.add_population(stark_female_infant, 1, 0)
        domain.add_population(stark_female_adolescent, 1, 0)

        pd.solvers.discrete_explicit(domain, 0, 1).execute()


def incorrect_species():
    """Should raise an exception in the solver error checker"""
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=1, left=0) as domain:
        domain.add_population(stark_female_adolescent, 1, 0)
        # Add white walker as a form of mortality, but purposefully do not add white walkers to the domain
        domain.add_mortality(stark_female_adolescent, white_walker_death, time=0)

        pd.solvers.discrete_explicit(domain, 0, 1).execute()


def single_species():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)
        domain.add_population(starks, 10000., 0, distribute_by_habitat=True)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()

        # Population should remain consistent at 10,000 for time steps 1 and 2
        for i in range(1, 3):
            tot_pop = summary.total_population(domain, 'stark', i).sum()
            if not np.isclose(tot_pop, 10000):
                raise Exception('The population has changed to {} at time {}'.format(tot_pop, i))


def single_species_emigration():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)
        domain.add_population(starks, 10000., 0, distribute_by_habitat=True)

        # Add population at time 1
        domain.add_population(starks, 10000., 1, distribute_by_habitat=True)

        pd.solvers.discrete_explicit(domain, 0, 2).execute()

        # Population should 20,000 at time step 1
        tot_pop = summary.total_population(domain, 'stark', 1).sum()
        if not np.isclose(tot_pop, 20000):
            raise Exception('The population should be {}, got {}'.format(20000, tot_pop))


def single_species_mvp():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        stark_with_mvp = pd.Species('stark', minimum_viable_population=100, minimum_viable_area=shape[0] * shape[1])

        domain.add_carrying_capacity(stark_with_mvp, stark_k, 0, shape[0] * shape[1] * 100.)
        domain.add_population(stark_with_mvp, 50., 0)

        pd.solvers.discrete_explicit(domain, 0, 10).execute()

        # Population should be reduced
        tp = []
        for i in range(0, 11):
            tp.append(summary.total_population(domain, 'stark', i).sum())
        if not any([t == 0 for t in tp]):
            raise Exception('The population did not succumb to MVP (populations: {})'.format(tp))


def single_species_random_k():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        rand_k = pd.CarryingCapacity('Stark Habitat')
        rand_k_data = some_random_k(shape, 1.)
        rand_k.random('normal', **{'scale': 10})
        domain.add_carrying_capacity(starks, rand_k, 0, rand_k_data, distribute=False)
        domain.add_population(starks, rand_k_data, 0, distribute=False)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()

        # Population is at k, so any k above the previous year should result in:
        #   1. density dependent mortality
        #   2. a reciprocal reduction in population
        prv_pop = rand_k_data
        for i in range(1, 3):
            # Make sure k is variable
            cc = summary.total_carrying_capacity(domain, 'stark', i)
            if np.all(cc == rand_k_data):
                raise Exception('The k is unchanged')
            tot_pop = summary.total_population(domain, 'stark', i)
            dd_mort = summary.total_mortality(domain, 'stark', i, mortality_name='Density Dependent')
            if not np.allclose(-dd_mort, (tot_pop - prv_pop), atol=1E-04):
                raise Exception('DD mortality not correct at time {}'.format(i))
            prv_pop = tot_pop


def single_species_sex():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)
        domain.add_population(starks, 10000., 0, distribute_by_habitat=True)
        domain.add_population(pd.Sex('Stark', 'male'), 5000., 0, distribute_by_habitat=True)
        domain.add_population(pd.Sex('Stark', 'female'), 5000., 0, distribute_by_habitat=True)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()

        # Population should remain consistent at 10,000 for time steps 1 and 2
        for i in range(1, 3):
            tot_pop = summary.total_population(domain, 'stark', i).sum()
            if not np.isclose(tot_pop, 20000):
                raise Exception('The total population has changed to {} at time {}'.format(tot_pop, i))
            tot_pop = summary.total_population(domain, 'stark', i, 'male').sum()
            if not np.isclose(tot_pop, 10000):
                raise Exception('The male population has changed to {} at time {}'.format(tot_pop, i))
            tot_pop = summary.total_population(domain, 'stark', i, 'female').sum()
            if not np.isclose(tot_pop, 10000):
                raise Exception('The female population has changed to {} at time {}'.format(tot_pop, i))
            cc = summary.total_carrying_capacity(domain, 'stark', i, 'male').sum()
            if not np.isclose(cc, (stark_k_data).sum()):
                raise Exception('The male carrying capacity has changed to {} at time {}'.format(cc, i))
            cc = summary.total_carrying_capacity(domain, 'stark', i, 'male').sum()
            if not np.isclose(cc, (stark_k_data).sum()):
                raise Exception('The female carrying capacity has changed to {} at time {}'.format(cc, i))


def single_species_fecundity():
    # F:M ratio allows offspring
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, 1000., distribute=False)
        males = pd.Sex('Stark', 'male')
        females = pd.Sex('Stark', 'female')
        domain.add_population(males, 50, 0)
        domain.add_population(females, 50, 0)
        domain.add_fecundity(males, male_fecundity, 0, 1.1)
        domain.add_fecundity(females, female_fecundity, 0, 1.1)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()

        # A constant set of offspring should result
        prv_pop = 50.
        for i in range(1, 3):
            tot_pop = summary.total_population(domain, 'stark', i, 'female').sum()
            tot_off = summary.total_offspring(domain, 'stark', i, 'female').sum()
            if not np.isclose(tot_off, prv_pop * 1.1):
                raise Exception(
                    'The offspring is {} and should be {} at time {}'.format(tot_off, prv_pop * 1.1, i)
                )
            if not np.isclose(tot_pop, prv_pop + (prv_pop * 1.1) / 2):
                raise Exception(
                    'The population is {} and should be {} at time {}'.format(
                        tot_pop, prv_pop + (prv_pop * 1.1) / 2, i
                    )
                )
            prv_pop = tot_pop

    if os.path.isfile('seven_kingdoms.popdyn'):
        os.remove('seven_kingdoms.popdyn')
    else:
        shutil.rmtree('seven_kingdoms.popdyn')

    # F:M ratio disallows offspring
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, 60)

        stark_male = pd.Sex('Stark', 'male')
        stark_female = pd.Sex('Stark', 'female')

        domain.add_population(stark_male, 20., 0)
        domain.add_population(stark_female, 40., 0)

        test_female_fecundity_1 = pd.Fecundity('females', fecundity_lookup=[(0., 1.), (2., 0.)])
        # Explore density-dependence
        test_female_fecundity_2 = pd.Fecundity('females',
                                               density_fecundity_threshold=0.9,
                                               density_fecundity_max=1.1,
                                               fecundity_lookup=[(0., 1.), (1.5, 1. / 1.1)])
        domain.add_fecundity(stark_male, male_fecundity, 0, 1.0)
        domain.add_fecundity(stark_female, test_female_fecundity_1, 0, 1.1)

        domain.add_fecundity(stark_female, test_female_fecundity_2, 2, 1.1)

        pd.solvers.discrete_explicit(domain, 0, 2).execute()

        male_pop = summary.total_offspring(domain, 'stark', 1, 'male').sum()
        if male_pop != 0:
            raise Exception('Offspring is {} when should be 0'.format(male_pop))

        female_pop = summary.total_population(domain, 'stark', 2, 'female').sum()
        # Fecundity should be 0.5
        if not np.isclose(female_pop, 40 + (40 * 0.5 / 2)):
            raise Exception(
                'The population is {} and should be {}'.format(female_pop, 40 + (40 * 0.5 / 2))
            )


def single_species_dispersion():
    for _iter, method in enumerate(['density-based dispersion', 'distance propagation']):
        with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
            starks = pd.Species('Stark')

            starks.add_dispersal(method, (10,))

            # Avoid density-dependent mortality
            domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data + 1000., distribute=False)

            seed_population = np.zeros(shape=shape, dtype='float32')
            seed_population[(np.random.randint(0, shape[0], 10), np.random.randint(0, shape[1], 10))] = 100
            domain.add_population(starks, seed_population, 0, distribute=False)

            try:
                pd.solvers.discrete_explicit(domain, 0, 2).execute()
            except NotImplementedError:
                domain.file.close()
                if os.path.isfile('seven_kingdoms.popdyn'):
                    os.remove('seven_kingdoms.popdyn')
                else:
                    shutil.rmtree('seven_kingdoms.popdyn')
                continue

            prv_pop = seed_population
            for i in range(1, 3):
                _pop = summary.total_population(domain, 'stark', i)
                if not np.isclose(_pop.sum(), prv_pop.sum()):
                    raise Exception('The population changed from {} to {} at time {}'.format(prv_pop.sum(), _pop.sum() ,i))
                if np.allclose(_pop, prv_pop):
                    raise Exception('The population did not disperse using {}'.format(method))
                prv_pop = _pop

        if _iter == 0:
            if os.path.isfile('seven_kingdoms.popdyn'):
                os.remove('seven_kingdoms.popdyn')
            else:
                shutil.rmtree('seven_kingdoms.popdyn')


def single_species_agegroups():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        pops = [250., 250., 600., 600., 3000., 3000.]
        tot_pop = np.sum(pops)

        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data + tot_pop, distribute=False)
        domain.add_population(stark_male_infant, 250., 0, distribute_by_habitat=True)
        domain.add_population(stark_female_infant, 250., 0, distribute_by_habitat=True)
        domain.add_population(stark_male_adolescent, 600., 0, distribute_by_habitat=True)
        domain.add_population(stark_female_adolescent, 600., 0, distribute_by_habitat=True)
        domain.add_population(stark_male_adult, 3000., 0, distribute_by_habitat=True)
        domain.add_population(stark_female_adult, 3000., 0, distribute_by_habitat=True)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()

        for i in range(1, 3):
            _pop = summary.total_population(domain, 'stark', i).sum()
            # Subtract old age mortality for the total population
            if not np.isclose(_pop, tot_pop - 3000. / (50 - 13. + 1) * 2 * i):
                raise Exception('The population should be {}, not {} at time {}'.format(
                    tot_pop - 3000. / (50 - 13. + 1) * 2, _pop, i)
                )
            # Check propagation
            _pop = summary.total_population(domain, 'stark', i, 'male', group=stark_male_infant.group_name).sum()
            if not np.isclose(_pop, 250 - ((250 / 4.) * i)):
                raise Exception('{} population should be {}, but is {} at time {}'.format(
                    stark_male_infant.group_name, 250 - ((250 / 4.) * i), _pop, i
                ))
            _pop = summary.total_population(domain, 'stark', i, 'male', group=stark_male_adolescent.group_name).sum()
            if not np.isclose(_pop, 600 - ((600 / 9.) * i) + ((250 / 4.) * i)):
                raise Exception('{} population should be {}, but is {} at time {}'.format(
                    stark_male_adolescent.group_name, 600 - ((600 / 9.) * i) + ((250 / 4.) * i), _pop, i
                ))
            _pop = summary.total_population(domain, 'stark', i, 'male', group=stark_male_adult.group_name).sum()
            if not np.isclose(_pop, 3000. - (3000. / (50 - 13. + 1) * i) + ((600 / 9.) * i)):
                raise Exception('{} population should be {}, but is {} at time {}'.format(
                    stark_male_adult.group_name, 3000. - (3000. / (50 - 13. + 1) * i) + ((600 / 9.) * i), _pop, i
                ))


def single_species_mask():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:

        starks = pd.Species('Stark')
        stark_male_adolescent = pd.AgeGroup('Stark', 'Adolescent', 'male', 4, 12)
        stark_male_adult = pd.AgeGroup('Stark', 'Adult', 'male', 13, 50)
        stark_male_adolescent.add_dispersal('masked density-based dispersion', (10,))

        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)
        domain.add_population(stark_male_adolescent, np.random.random(shape), 0, distribute=False)
        domain.add_population(stark_male_adult, np.random.random(shape), 0, distribute=False)
        domain.add_mask(starks, 0, np.random.random(shape), distribute=False)  # Should be inherited

        pd.solvers.discrete_explicit(domain, 0, 2).execute()

        for i in range(1, 2):
            _pop = summary.total_population(domain, 'stark', i).sum() + summary.total_mortality(domain, 'stark', i).sum()
            if not np.isclose(_pop, summary.total_population(domain, 'stark', 0).sum()):
                raise Exception('Population should be {}, got {}'.format(
                    summary.total_population(domain, 'stark', 0).sum(), _pop)
                )
            if np.allclose(summary.total_population(domain, 'stark', i) + summary.total_mortality(domain, 'stark', i),
                           summary.total_population(domain, 'stark', 0)):
                raise Exception('No dispersal occurred')


def single_species_mortality():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=shape[0], left=0) as domain:
        # Test old age propagation here
        stark_male_adolescent = pd.AgeGroup('Stark', 'Adolescent', 'male', 4, 12, live_past_max=True)
        stark_female_adolescent = pd.AgeGroup('Stark', 'Adolescent', 'female', 4, 12, live_past_max=True)

        domain.add_carrying_capacity(starks, stark_k, 0, 1000., distribute=False)
        domain.add_population(stark_male_adolescent, 10., 0)
        domain.add_population(stark_female_adolescent, 10., 0)
        domain.add_mortality(stark_male_adolescent, disease, 0, 0.2)
        domain.add_mortality(stark_female_adolescent, disease, 0, 0.2)
        domain.add_mortality(stark_female_adolescent, accident, 0, 0.1)
        domain.add_mortality(stark_male_adolescent, accident, 0, 0.1)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()

        for sex in ['male', 'female']:
            prv_pop = 10
            for i in range(1, 3):
                tot_pop = summary.total_population(domain, 'stark', i, sex, group='adolescent')
                mort1 = summary.total_mortality(domain, 'stark', i, sex, mortality_name='Accident')
                mort2 = summary.total_mortality(domain, 'stark', i, sex, mortality_name='Disease')
                if not np.isclose(tot_pop.sum(), prv_pop - (mort1.sum() + mort2.sum())):
                    raise Exception('{} adolescent mortality not correct at time {}'.format(sex, i))
                prv_pop = tot_pop


def single_species_recipient():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=shape[0], left=0) as domain:
        recipient = pd.Species('recipient')

        test_mortality = pd.Mortality('test recipient')
        test_mortality.add_recipient_species(recipient)

        domain.add_carrying_capacity(starks, stark_k, 0, 10000, distribute=False)
        domain.add_population(starks, 10000., 0, distribute_by_habitat=True)
        domain.add_mortality(starks, test_mortality, 0, 0.1)
        domain.add_population(recipient, 0, 10)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()

        tot_pop = summary.total_population(domain, 'recipient', 1).sum()
        if tot_pop != 10000 * .1:
            raise Exception('Recipient population should be {}, got {} at time 1'.format(
                10000 * .1, tot_pop)
            )
        tot_pop = summary.total_population(domain, 'recipient', 2).sum()
        if tot_pop != (10000 - (10000 * .1)) * .1:
            raise Exception('Recipient population should be {}, got {} at time 2'.format(
                (10000 - (10000 * .1)) * .1, tot_pop)
            )


def species_as_mortality():
    # White Walker mortality lookup: [(0, 0), (0.1, 0.1), (0.5, 0.8), (1, 0.9)]
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, 100)
        domain.add_population(starks, 100, 0)
        domain.add_mortality(starks, white_walker_death, 0)

        # 30% density, mortality should be 0.45
        domain.add_carrying_capacity(white_walker, white_walker_k, 0, 100)
        domain.add_population(white_walker, 30., 0)

        # 50% density, mortality should be 0.8
        domain.add_carrying_capacity(white_walker, white_walker_k, 2, 60)

        pd.solvers.discrete_explicit(domain, 0, 2).execute()

        tot_pop = summary.total_population(domain, 'stark', 1).sum()
        if not np.isclose(tot_pop, 100 - (100 * 0.45)):
            raise Exception('Population is {}, but should be {} at time step 1'.format(tot_pop, 100 - (100 * 0.45)))
        tot_pop = summary.total_population(domain, 'stark', 2).sum()
        if not np.isclose(tot_pop, 55. - (55 * 0.8)):
            raise Exception('Population is {}, but should be {} at time step 1'.format(tot_pop, 55. - (55 * 0.8)))


def species_as_carrying_capacity():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, 100)
        domain.add_carrying_capacity(lannister, lannister_k, 0, 100)
        domain.add_population(starks, 20, 0)  # Density should be 20%, increasing lannister habitat by 20%
        domain.add_population(lannister, 100, 0)

        stark_as_k = pd.CarryingCapacity('Starks')
        stark_as_k.add_as_species(starks, [(0., 1), (0.2, 1.2), (0.5, 0.9), (1., 0.1)])

        domain.add_carrying_capacity(lannister, stark_as_k, 0)

        pd.solvers.discrete_explicit(domain, 0, 2).execute()

        cc = summary.total_carrying_capacity(domain, 'lannister', 1).sum()
        if not np.isclose(cc, 120):
            raise Exception('Carrying capacity should be {}, but it is {}'.format(120, cc))


def circular_species():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, 100)
        domain.add_carrying_capacity(lannister, lannister_k, 0, 100)
        domain.add_population(starks, 20, 0)  # Density should be 20%, increasing lannister habitat by 20%
        domain.add_population(lannister, 100, 0)  # Density should be 100%, decreasing stark habitat by 90%

        stark_as_k = pd.CarryingCapacity('Starks')
        stark_as_k.add_as_species(starks, [(0., 1), (0.2, 1.2), (0.5, 0.9), (1., 0.1)])

        lannister_as_k = pd.CarryingCapacity('Lannister')
        lannister_as_k.add_as_species(lannister, [(0., 1), (0.2, 1.2), (0.5, 0.9), (1., 0.1)])

        domain.add_carrying_capacity(lannister, stark_as_k, 0)
        domain.add_carrying_capacity(starks, lannister_as_k, 0)

        pd.solvers.discrete_explicit(domain, 0, 2).execute()

        if not np.isclose(summary.total_carrying_capacity(domain, 'lannister', 1).sum(), 120):
            raise Exception('Lannister carrying capacity should be {}, but it is {}'.format(120, cc))
        if not np.isclose(summary.total_carrying_capacity(domain, 'stark', 1).sum(), 10):
            raise Exception('Stark carrying capacity should be {}, but it is {}'.format(10, cc))


def max_age():
    with pd.Domain('seven_kingdoms.popdyn', csx=1, csy=1, shape=(1, 1), top=1, left=0) as domain:
        gp = pd.AgeGroup('Stark', 'Infant', 'male', 0, 3)
        gp.live_past_max = True
        domain.add_carrying_capacity(gp, stark_k, 0, 100)
        domain.add_population(gp, 20, 0)

        pd.solvers.discrete_explicit(domain, 0, 10).execute()

        # The age range of the group should have changed
        if domain.species['stark']['male']['infant'].max_age != 13:
            raise Exception('The max age is {}, and it should be 13'.format(domain.species['stark']['male']['infant'].max_age))


def global_n_interspecies():
    # White Walker mortality lookup: [(0, 0), (0.1, 0.1), (0.5, 0.8), (1, 0.9)]
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, 100)
        domain.add_population(starks, 100, 0)

        white_walker_death = pd.Mortality('White Walker')
        white_walker_death.add_as_species(white_walker, [(0, 0), (1/3., 0.1), (0.35714, 0.5), (1, 1)],
                                          population_type='global ratio')

        domain.add_mortality(starks, white_walker_death, 0)

        # should be 1/3, making mortality 0.1
        domain.add_carrying_capacity(white_walker, white_walker_k, 0, 100)
        domain.add_population(white_walker, 50., 0)

        pd.solvers.discrete_explicit(domain, 0, 2, global_density=True).execute()

        tot_pop = summary.total_population(domain, 'stark', 1).sum()
        if not np.isclose(tot_pop, 90):
            raise Exception('Population is {}, but should be {} at time step 1'.format(tot_pop, 90))
        tot_pop = summary.total_population(domain, 'stark', 2).sum()
        if not np.isclose(tot_pop, 90 * .5):
            raise Exception('Population is {}, but should be {} at time step 2'.format(tot_pop, 90 * .5))


def rate_based_mortality():
    # Pasted from recipient species
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=shape[0], left=0) as domain:
        recipient = pd.Species('recipient')

        time_based_mortality = pd.Mortality('time based')
        time_based_mortality.add_time_based_mortality([0.2, 0.5, 1.])

        domain.add_mortality(recipient, time_based_mortality, 0)

        test_mortality = pd.Mortality('test recipient')
        test_mortality.add_recipient_species(recipient)

        domain.add_carrying_capacity(starks, stark_k, 0, 10000, distribute=False)
        domain.add_population(starks, 10000., 0, distribute_by_habitat=True)
        domain.add_mortality(starks, test_mortality, 0, 0.1)
        pd.solvers.discrete_explicit(domain, 0, 5).execute()

        tot_shouldbe = [1000, 900, 810, 729, 656.10004]
        mort_shouldbe = [0, 200, 450, 810, 0]

        tot_pop, mort = [], []
        for year in range(1, 6):
            tot_pop.append(summary.total_population(domain, 'recipient', year).sum())
            mort.append(np.squeeze(summary.total_mortality(domain, 'recipient', year, mortality_name='time based')))

        if not np.allclose(tot_pop, tot_shouldbe) or not np.allclose(mort, mort_shouldbe):
            raise Exception('Population should be {}, got {}\nMortality should be {}, got {}'.format(
                tot_shouldbe, tot_pop, mort_shouldbe, mort)
            )


if __name__ == '__main__':
    antitests = [no_species, incorrect_ages, incorrect_species]

    tests = [single_species, single_species_emigration, single_species_mvp, single_species_random_k,
             single_species_sex, single_species_fecundity, single_species_dispersion, single_species_agegroups,
             single_species_mask, single_species_mortality, single_species_recipient, species_as_mortality,
             species_as_carrying_capacity, circular_species, max_age, rate_based_mortality, global_n_interspecies]

    error_check = 0
    now = time.time()
    for test in antitests:
        try:
            test()
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                  "{}: FAIL:\nNo Exception raised\n"
                  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n".format(test.__name__))
        except Exception as e:
            if (isinstance(e, pd.solvers.SolverError) or
                isinstance(e, pd.PopdynError)):
                error_check += 1
                print("--------------------------------------\n"
                      "{}: PASS\n"
                      "--------------------------------------\n".format(test.__name__))
            else:
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                      "{}: FAIL:\n{}\n"
                      "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n".format(test.__name__, e))
        if os.path.isfile('seven_kingdoms.popdyn'):
            os.remove('seven_kingdoms.popdyn')
        else:
            shutil.rmtree('seven_kingdoms.popdyn')

    function_check = 0
    for test in tests:
        try:
            test()
            function_check += 1
            print("--------------------------------------\n"
                  "{}: PASS\n"
                  "--------------------------------------\n".format(test.__name__))
        except Exception as e:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                  "{}: FAIL:\n{}\n"
                  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n".format(test.__name__, e))
        if os.path.isfile('seven_kingdoms.popdyn'):
            os.remove('seven_kingdoms.popdyn')
        else:
            shutil.rmtree('seven_kingdoms.popdyn')

    stars = np.array([[' '] * 50 for i in range(10)])
    stars[(np.random.randint(0, 5, 40), np.random.randint(0, 49, 40))] = '*'
    stars[(np.random.randint(5, 10, 10), np.random.randint(0, 49, 10))] = '*'
    stars[:, -1] = '\n'
    stars = ''.join(stars.ravel())

    print("{}"
          "\n\n\n                Tests Completed in {} seconds\n\n"
          "".format(stars, time.time() - now))
    print("          {} of {} exception checks passed".format(error_check, len(antitests)))
    print("          {} of {} solver checks passed\n\n\n"
          "___wv_____wwWww____vW___|____wWv___|_|____w____wwvWWv".format(function_check, len(tests)))
