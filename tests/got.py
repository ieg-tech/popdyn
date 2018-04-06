"""
Population dynamics module tests
"""


import os
import numpy as np
from scipy.ndimage import gaussian_filter
import popdyn as pd


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

# TODO: Add a population check for these functions once summary.py is complete

def single_species():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)
        domain.add_population(starks, 10000., 0, distribute_by_habitat=True)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()


def single_species_random():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        stark_k.random('normal', args=(10,))
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)
        domain.add_population(starks, 10000., 0, distribute_by_habitat=True)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()


def single_species_dispersion():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        starks.add_dispersal('density-based dispersion', (10,))
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)

        seed_population = np.zeros(shape=shape, dtype='float32')
        seed_population[(np.random.randint(0, shape[0], 10), np.random.randint(0, shape[1], 10))] = 100
        domain.add_population(starks, seed_population, 0, distribute=False)

        pd.solvers.discrete_explicit(domain, 0, 2).execute()


def single_species_sex():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)
        domain.add_population(starks, 10000., 0, distribute_by_habitat=True)
        domain.add_population(pd.Sex('Stark', 'male'), 5000., 0, distribute_by_habitat=True)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()


def single_species_fecundity():
    with pd.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=shape[0], left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, 0, stark_k_data, distribute=False)
        domain.add_population(starks, 10000., 0, distribute_by_habitat=True)
        domain.add_population(pd.Sex('Stark', 'male', fecundity=1), 5000., 0, distribute_by_habitat=True)
        domain.add_population(pd.Sex('Stark', 'female', fecundity=1.1), 5000., 0, distribute_by_habitat=True)
        pd.solvers.discrete_explicit(domain, 0, 2).execute()


def single_species_agegroups():
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
        pd.solvers.discrete_explicit(domain, 0, 2).execute()


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
        domain.add_population(starks, stark_k_data * 0.3, 0)
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
    # Test error checking
    # ============================================================================
    # try:
    #     no_species()
    #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #           "Incorrect species addition test: FAIL:\nNo Exception raised\n"
    #           "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # except Exception as e:
    #     if isinstance(e, pd.solvers.SolverError):
    #         print("--------------------------------------\n"
    #               "Incorrect species addition test: PASS\n"
    #               "--------------------------------------")
    #     else:
    #         print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #               "Incorrect species addition test: FAIL:\n{}\n"
    #               "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    # print("")
    # os.remove('seven_kingdoms.popdyn')
    #
    # try:
    #     incorrect_ages()
    #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #           "Incorrect age group test: FAIL:\nNo Exception raised\n"
    #           "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # except Exception as e:
    #     if isinstance(e, pd.solvers.SolverError):
    #         print("--------------------------------------\n"
    #               "Incorrect age group test: PASS\n"
    #               "--------------------------------------")
    #     else:
    #         print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #               "Incorrect age group test: FAIL:\n{}\n"
    #               "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    # print("")
    # os.remove('seven_kingdoms.popdyn')
    #
    # try:
    #     incorrect_species()
    #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #           "Incorrect species available test: FAIL:\nNo Exception raised\n"
    #           "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # except Exception as e:
    #     if isinstance(e, pd.solvers.SolverError):
    #         print("--------------------------------------\n"
    #               "Incorrect species available test: PASS\n"
    #               "--------------------------------------")
    #     else:
    #         print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #               "Incorrect species available test: FAIL:\n{}\n"
    #               "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    # print("")
    # os.remove('seven_kingdoms.popdyn')

    # Test a single species without any sex, age groups, dispersion, or randomness
    # ============================================================================
    # try:
    #     single_species()
    #     print("--------------------------------------\n"
    #           "Single species test: PASS\n"
    #           "--------------------------------------")
    # except Exception as e:
    #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #           "Single species test: FAIL:\n{}\n"
    #           "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    # os.remove('seven_kingdoms.popdyn')

    #   ...add randomness
    # ============================================================================
    # try:
    #     single_species_random()
    #     print("--------------------------------------\n"
    #           "Single species, random k test: PASS\n"
    #           "--------------------------------------")
    # except Exception as e:
    #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #           "Single species, random k test: FAIL:\n{}\n"
    #           "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    # os.remove('seven_kingdoms.popdyn')

    #   ...add dispersion
    # ============================================================================
    # try:
    #     single_species_dispersion()
    #     print("--------------------------------------\n"
    #           "Single species, dispersion test: PASS\n"
    #           "--------------------------------------")
    # except Exception as e:
    #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #           "Single species, dispersion test: FAIL:\n{}\n"
    #           "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    # os.remove('seven_kingdoms.popdyn')
    #

    #   ...add males
    # ============================================================================
    # try:
    #     single_species_sex()
    #     print("--------------------------------------\n"
    #           "Single species, sex test: PASS\n"
    #           "--------------------------------------")
    # except Exception as e:
    #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #           "Single species, sex test: FAIL:\n{}\n"
    #           "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    # os.remove('seven_kingdoms.popdyn')

    #   ...add fecundity
    # ============================================================================
    # try:
    #     single_species_fecundity()
    #     print("--------------------------------------\n"
    #           "Single species, fecundity test: PASS\n"
    #           "--------------------------------------")
    # except Exception as e:
    #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #           "Single species, fecundity test: FAIL:\n{}\n"
    #           "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    # os.remove('seven_kingdoms.popdyn')

    #   ...add age groups
    # ============================================================================
    # try:
    #     single_species_agegroups()
    #     print("--------------------------------------\n"
    #           "Single species, age groups test: PASS\n"
    #           "--------------------------------------")
    # except Exception as e:
    #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #           "Single species, age groups test: FAIL:\n{}\n"
    #           "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    # os.remove('seven_kingdoms.popdyn')

    #   ...add mortality
    # ============================================================================
    # try:
    #     single_species_mortality()
    #     print("--------------------------------------\n"
    #           "Single species, mortality test: PASS\n"
    #           "--------------------------------------")
    # except Exception as e:
    #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #           "Single species, mortality test: FAIL:\n{}\n"
    #           "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    # os.remove('seven_kingdoms.popdyn')

    #   ...add species as mortality
    # ============================================================================
    # try:
    #     species_as_mortality()
    #     print("--------------------------------------\n"
    #           "Single species, species as mortality test: PASS\n"
    #           "--------------------------------------")
    # except Exception as e:
    #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    #           "Single species, species as mortality test: FAIL:\n{}\n"
    #           "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    # os.remove('seven_kingdoms.popdyn')

    #   ...add species as carrying capacity
    # ============================================================================
    try:
        species_as_carrying_capacity()
        print("--------------------------------------\n"
              "Single species, species as carrying capacity test: PASS\n"
              "--------------------------------------")
    except Exception as e:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
              "Single species, species as carrying capacity test: FAIL:\n{}\n"
              "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    # os.remove('seven_kingdoms.popdyn')

    # datasets = []
    # with pd.h5py.File('/Users/devin/PycharmProjects/popdyn/tests/seven_kingdoms.popdyn') as f:
    #     for i in range(3):
    #         datasets.append(f['stark']['male']['None'][str(i)]['None'][:])

    # ds = test_simulation()
