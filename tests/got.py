"""
Population dynamics module tests
"""


import os
import numpy as np
from scipy.ndimage import gaussian_filter
import popdyn as pd
import h5py
import dask.array as da


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
shape = (180, 320)


# Species
#================================================================================

# Stark
starks = pd.Species('Stark')

stark_male_infant = pd.AgeGroup('Stark', 'Infant', 'male', 0, 3)
stark_female_infant = pd.AgeGroup('Stark', 'Infant', 'female', 0, 3)

stark_male_adolescent = pd.AgeGroup('Stark', 'Adolescent', 'male', 4, 12, fecundity=1.)
stark_female_adolescent = pd.AgeGroup('Stark', 'Adolescent', 'female', 4, 12, fecundity=0.9)

stark_male_adult = pd.AgeGroup('Stark', 'Adult', 'male', 13, 50, fecundity=1.)
stark_female_adult = pd.AgeGroup('Stark', 'Adult', 'female', 13, 50, fecundity=0.6)

# Lannister
lannister = pd.Species('Lannister')

lannister_male_infant = pd.AgeGroup('Lannister', 'Infant', 'male', 0, 3)
lannister_female_infant = pd.AgeGroup('Lannister', 'Infant', 'female', 0, 3)

lannister_male_adolescent = pd.AgeGroup('Lannister', 'Adolescent', 'male', 4, 12, fecundity=1.)
lannister_female_adolescent = pd.AgeGroup('Lannister', 'Adolescent', 'female', 4, 12, fecundity=0.9)

lannister_male_adult = pd.AgeGroup('Lannister', 'Adult', 'male', 13, 50, fecundity=1.)
lannister_female_adult = pd.AgeGroup('Lannister', 'Adult', 'female', 13, 50, fecundity=0.6)

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
male_fecundity = pd.Fecundity('males')
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


def circular_carrying_capacity():
    """Should raise an exception in the solver error checker"""
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=1, left=0) as domain:
        cc_1 = pd.CarryingCapacity('Test')
        cc_1.add_as_species(starks, [(0., 1.), (1., 1.)])
        cc_2 = pd.CarryingCapacity('Test')
        cc_2.add_as_species(lannister, [(0., 1.), (1., 1.)])
        domain.add_carrying_capacity(starks, cc_2, 0)
        domain.add_carrying_capacity(lannister, cc_1, 0)
        pd.solvers.discrete_explicit(domain, 0, 1).execute()


def single_species():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)
        domain.add_population(starks, 10000., 0, distribute_by_habitat=True)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()

    with h5py.File('seven_kingdoms.popdyn', libver='latest') as f:
        for i in range(3):
            if np.sum(f['stark/None/None/{}/None'.format(i)][:]) != 10000.:
                raise Exception('Population not consistent')


def single_species_random_k():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        stark_k.random('normal', args=(10,))
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)
        domain.add_population(starks, stark_k_data, 0, distribute=False)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()

    with h5py.File('seven_kingdoms.popdyn', libver='latest') as f:
        pop = np.sum(f['stark/None/None/{}/None'.format(0)][:])
        for i in range(1, 3):
            if np.sum(f['stark/None/None/{}/None'.format(i)][:]) == pop:
                raise Exception('No variability in k')


def single_species_dispersion():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        starks.add_dispersal('density-based dispersion', (10,))
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)

        seed_population = np.zeros(shape=shape, dtype='float32')
        seed_population[(np.random.randint(0, shape[0], 10), np.random.randint(0, shape[1], 10))] = 100
        domain.add_population(starks, seed_population, 0, distribute=False)

        pd.solvers.discrete_explicit(domain, 0, 2).execute()

    with h5py.File('seven_kingdoms.popdyn', libver='latest') as f:
        pop = f['stark/None/None/{}/None'.format(0)][:]
        for i in range(1, 3):
            new_pop = f['stark/None/None/{}/None'.format(i)][:]
            if np.all(np.isclose(new_pop, pop)):
                raise Exception('No dispersal occurred')
            pop = new_pop


def single_species_sex():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)
        domain.add_population(starks, 10000., 0, distribute_by_habitat=True)
        domain.add_population(pd.Sex('Stark', 'male'), 5000., 0, distribute_by_habitat=True)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()

    with h5py.File('seven_kingdoms.popdyn', libver='latest') as f:
        for i in range(3):
            if not np.isclose(np.sum(f['stark/male/None/{}/None'.format(i)][:]), 15000.):
                raise Exception('Population for males changed')


def single_species_fecundity():
    # F:M ratio allows offspring
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, 100., distribute=False)
        males = pd.Sex('Stark', 'male')
        females = pd.Sex('Stark', 'female')
        domain.add_population(males, 50, 0)
        domain.add_population(females, 50, 0)
        domain.add_fecundity(males, male_fecundity, 0, 1.1)
        domain.add_fecundity(females, female_fecundity, 0, 1.1)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()

    with h5py.File('seven_kingdoms.popdyn', libver='latest') as f:
        fm = np.sum(f['stark/female/None/{}/flux/offspring/Female Offspring'.format(1)][:])
        ml = np.sum(f['stark/female/None/{}/flux/offspring/Male Offspring'.format(1)][:])
        if not np.isclose(ml, (50 * 1.1 / 2.)) or not np.isclose(fm, (50 * 1.1 / 2)):
            raise Exception('Incorrect offspring magnitude')
        fm = np.sum(f['stark/female/None/{}/flux/offspring/Female Offspring'.format(2)][:])
        ml = np.sum(f['stark/female/None/{}/flux/offspring/Male Offspring'.format(2)][:])
        if not np.isclose(ml, 77.5 * 1.1 / 2) or not np.isclose(fm, 77.5 * 1.1 / 2):
            raise Exception('Incorrect offspring magnitude')

    os.remove('seven_kingdoms.popdyn')

    # F:M ratio disallows offspring
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, 60)
        domain.add_population(pd.Sex('Stark', 'male', fecundity=1), 20., 0)
        domain.add_population(pd.Sex('Stark', 'female', fecundity=1.1,
                                     fecundity_lookup=[(0., 1.), (2., 0.)]), 40., 0)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()

    with h5py.File('seven_kingdoms.popdyn', libver='latest') as f:
        for i in range(1, 3):
            fm = np.sum(f['stark/female/None/{}/flux/offspring/Female Offspring'.format(i)][:])
            ml = np.sum(f['stark/female/None/{}/flux/offspring/Male Offspring'.format(i)][:])
            if ml != 0 or fm != 0:
                raise Exception('Offspring occurred when not allowable')


def single_species_agegroups():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)
        stark_female_adolescent.fecundity = 0.
        stark_female_adult.fecundity = 0.
        domain.add_population(stark_male_infant, 250., 0, distribute_by_habitat=True)
        domain.add_population(stark_female_infant, 250., 0, distribute_by_habitat=True)
        domain.add_population(stark_male_adolescent, 600., 0, distribute_by_habitat=True)
        domain.add_population(stark_female_adolescent, 600., 0, distribute_by_habitat=True)
        domain.add_population(stark_male_adult, 3000., 0, distribute_by_habitat=True)
        domain.add_population(stark_female_adult, 3000., 0, distribute_by_habitat=True)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()

        pop = da.stack(
            [da.from_array(ds, domain.chunks) for ds in domain.all_population('stark', 0).values()]
        ).sum().compute()
        for i in range(1, 3):
            _pop = da.stack(
                [da.from_array(ds, domain.chunks) for ds in domain.all_population('stark', i).values()]
            ).sum().compute()
            print(_pop)
            # if not np.isclose(pop, _pop):
            #     raise Exception('Population not consistent')


def single_species_mortality():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)
        stark_male_infant.add_dispersal('density-based dispersion', (5,))
        stark_female_infant.add_dispersal('density-based dispersion', (5,))
        stark_male_adolescent.add_dispersal('density-based dispersion', (5,))
        stark_female_adolescent.add_dispersal('density-based dispersion', (5,))
        stark_male_adult.add_dispersal('density-based dispersion', (5,))
        stark_female_adult.add_dispersal('density-based dispersion', (5,))
        domain.add_population(stark_male_infant, 250., 0, distribute_by_habitat=True)
        domain.add_population(stark_female_infant, 250., 0, distribute_by_habitat=True)
        domain.add_population(stark_male_adolescent, 600., 0, distribute_by_habitat=True)
        domain.add_population(stark_female_adolescent, 600., 0, distribute_by_habitat=True)
        domain.add_population(stark_male_adult, 3000., 0, distribute_by_habitat=True)
        domain.add_population(stark_female_adult, 3000., 0, distribute_by_habitat=True)
        for i in range(3):
            domain.add_mortality(starks, disease, i, 0.1, distribute=False)
            domain.add_mortality(starks, accident, i, 0.1, distribute=False)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()


def species_as_mortality():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)
        domain.add_population(starks, 10000., 0, distribute_by_habitat=True)
        for i in range(3):
            domain.add_mortality(starks, white_walker_death, i)

        domain.add_carrying_capacity(white_walker, white_walker_k, 0, white_walker_k_data, distribute=False)
        domain.add_population(white_walker, 1000., 0, distribute_by_habitat=True)

        pd.solvers.discrete_explicit(domain, 0, 2).execute()


def species_as_carrying_capacity():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)
        domain.add_carrying_capacity(lannister, lannister_k, 0, lannister_k_data, distribute=False)
        domain.add_population(starks, stark_k_data * 0.9, 0)
        domain.add_population(lannister, 10000., 0, distribute_by_habitat=True)

        stark_as_k = pd.CarryingCapacity('Starks')
        stark_as_k.add_as_species(starks, [(0., 0.), (0.2, .2), (0.5, -0.1), (1., -0.9)])

        domain.add_carrying_capacity(lannister, stark_as_k, 0)

        pd.solvers.discrete_explicit(domain, 0, 2).execute()


def test_simulation():
        # Create a rectangular domain
    #===============================================================================
    shape = (180, 320)
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=180, left=0) as domain:
        # Add starks
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)

        domain.add_population(starks, 10000., 0, distribute_by_habitat=True)

        domain.add_population(pd.Sex('Stark', 'female'), 10000., 0, distribute_by_habitat=True)

        domain.add_population(stark_male_infant, 10000., 0, distribute_by_habitat=True)
        domain.add_population(stark_female_infant, 10000., 0, distribute_by_habitat=True)
        domain.add_population(stark_male_adolescent, 200000., 0, distribute_by_habitat=True)
        domain.add_population(stark_female_adolescent, 200000., 0, distribute_by_habitat=True)
        domain.add_population(stark_male_adult, 500000., 0, distribute_by_habitat=True)
        domain.add_population(stark_female_adult, 500000., 0, distribute_by_habitat=True)
        domain.add_population(stark_male_elder, 10000., 0, distribute_by_habitat=True)
        domain.add_population(stark_female_elder, 10000., 0, distribute_by_habitat=True)

        # domain.add_mortality()

        # Add lannisters
        domain.add_carrying_capacity(lannister, lannister_k, 0, lannister_k_data, distribute=False)

        domain.add_population(lannister_male_infant, 10000, 0, distribute_by_habitat=True)
        domain.add_population(lannister_female_infant, 10000, 0, distribute_by_habitat=True)
        domain.add_population(lannister_male_adolescent, 200000, 0, distribute_by_habitat=True)
        domain.add_population(lannister_female_adolescent, 200000, 0, distribute_by_habitat=True)
        domain.add_population(lannister_male_adult, 500000, 0, distribute_by_habitat=True)
        domain.add_population(lannister_female_adult, 500000, 0, distribute_by_habitat=True)
        domain.add_population(lannister_male_elder, 10000, 0, distribute_by_habitat=True)
        domain.add_population(lannister_female_elder, 10000, 0, distribute_by_habitat=True)

        # Add white walkers
        domain.add_carrying_capacity(white_walker, stark_k, 0, stark_k_data, distribute=False)

        white_walker_pop0 = np.random.normal(100, 1000, shape)
        white_walker_pop0[white_walker_pop0 < 0] = 0
        white_walker_pop0[80:, :] = 0

        domain.add_population(white_walker, white_walker_pop0, 0)
        pd.solvers.discrete_explicit(domain, 0, 1)
        # pd.solvers.discrete_explicit(domain, 0, 1).execute()

        return domain.profiler


if __name__ == '__main__':
    antitests = [no_species, incorrect_ages, incorrect_species, circular_carrying_capacity]

    # tests = [single_species, single_species_random_k, single_species_dispersion, single_species_sex,
    #          single_species_fecundity, single_species_agegroups, single_species_mortality,
    #          species_as_mortality, species_as_carrying_capacity]

    tests = [single_species_sex]

    error_check = 0
    # for test in antitests:
    #     try:
    #         test()
    #         print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #               "{}: FAIL:\nNo Exception raised\n"
    #               "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n".format(test.__name__))
    #     except Exception as e:
    #         if (isinstance(e, pd.solvers.SolverError) or
    #             isinstance(e, pd.PopdynError)):
    #             error_check += 1
    #             print("--------------------------------------\n"
    #                   "{}: PASS\n"
    #                   "--------------------------------------\n".format(test.__name__))
    #         else:
    #             print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #                   "{}: FAIL:\n{}\n"
    #                   "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n".format(test.__name__, e))
    #     os.remove('seven_kingdoms.popdyn')

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
        os.remove('seven_kingdoms.popdyn')

    print("Completed tests\n")
    print("{} of {} exception checks passed".format(error_check, len(antitests)))
    print("{} of {} solver checks passed".format(function_check, len(tests)))

    # datasets = []
    # with pd.h5py.File('/Users/devin/PycharmProjects/popdyn/tests/seven_kingdoms.popdyn') as f:
    #     for i in range(3):
    #         datasets.append(f['stark']['male']['None'][str(i)]['None'][:])

    # ds = test_simulation()
