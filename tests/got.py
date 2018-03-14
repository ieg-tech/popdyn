"""
Population dynamics module tests
"""


import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import popdyn
from popdyn import solvers


def show(ds):
    out = ds.copy()
    out[out == 0] = np.nan
    plt.imshow(out, cmap='jet')
    plt.show()

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
shape = (720, 1280)


# Species
#================================================================================

# Stark
starks = popdyn.Species('Stark')

stark_male_infant = popdyn.AgeGroup('Stark', 'Infant', 'male', 0, 3)
stark_female_infant = popdyn.AgeGroup('Stark', 'Infant', 'female', 0, 3)

stark_male_adolescent = popdyn.AgeGroup('Stark', 'Adolescent', 'male', 4, 12)
stark_female_adolescent = popdyn.AgeGroup('Stark', 'Adolescent', 'female', 4, 12)

stark_male_adult = popdyn.AgeGroup('Stark', 'Adult', 'male', 13, 60)
stark_female_adult = popdyn.AgeGroup('Stark', 'Adult', 'female', 13, 60)

stark_male_elder = popdyn.AgeGroup('Stark', 'Elder', 'male', 61, 100)
stark_female_elder = popdyn.AgeGroup('Stark', 'Elder', 'female', 61, 100)

# Lannister
lannister = popdyn.Species('Lannister')

lannister_male_infant = popdyn.AgeGroup('Lannister', 'Infant', 'male', 0, 3)
lannister_female_infant = popdyn.AgeGroup('Lannister', 'Infant', 'female', 0, 3)

lannister_male_adolescent = popdyn.AgeGroup('Lannister', 'Adolescent', 'male', 4, 12)
lannister_female_adolescent = popdyn.AgeGroup('Lannister', 'Adolescent', 'female', 4, 12)

lannister_male_adult = popdyn.AgeGroup('Lannister', 'Adult', 'male', 13, 60)
lannister_female_adult = popdyn.AgeGroup('Lannister', 'Adult', 'female', 13, 60)

lannister_male_elder = popdyn.AgeGroup('Lannister', 'Elder', 'male', 61, 100)
lannister_female_elder = popdyn.AgeGroup('Lannister', 'Elder', 'female', 61, 100)

# White walker
white_walker = popdyn.Species('White Walker')

# Mortality
#================================================================================

# For humans
disease = popdyn.Mortality('Disease')
accident = popdyn.Mortality('Accident')
white_walker_death = popdyn.Mortality('White Walker')
white_walker_death.add_as_species(white_walker, [(0, 0), (0.1, 0.1), (0.5, 0.8), (1, 0.9)])

# Stark soldiers
lannister_male_soldier = popdyn.Mortality('Lannister Male')
lannister_male_soldier.add_as_species(lannister_male_adult, [(0, 0), (0.2, 0.05), (1, 0.1)])
lannister_female_soldier = popdyn.Mortality('Lannister Female')
lannister_female_soldier.add_as_species(lannister_female_adult, [(0, 0), (0.2, 0.01), (1, 0.05)])

# Lannister soldiers
stark_male_soldier = popdyn.Mortality('Stark Male')
stark_male_soldier.add_as_species(stark_male_adult, [(0, 0), (0.2, 0.05), (1, 0.1)])
stark_female_soldier = popdyn.Mortality('Stark Female')
stark_female_soldier.add_as_species(stark_male_adult, [(0, 0), (0.2, 0.05), (1, 0.1)])

# Carrying capacity
#================================================================================

stark_k_data = some_random_k(shape, 1.)
stark_k = popdyn.CarryingCapacity('Stark Habitat')
lannister_k_data = some_random_k(shape, 1.2)
lannister_k = popdyn.CarryingCapacity('Lannister Habitat')


# Tests
#================================================================================

def no_species():
    """Should raise an exception in the solver error checker"""
    with popdyn.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=1, left=0) as domain:
        solvers.discrete_explicit(domain, 0, 1).execute()


def incorrect_ages():
    """Should raise an exception in the solver error checker"""
    with popdyn.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=1, left=0) as domain:
        stark_female_infant = popdyn.AgeGroup('Stark', 'Infant', 'female', 0, 3)
        stark_female_adolescent = popdyn.AgeGroup('Stark', 'Adolescent', 'female', 3, 12)  # Age range overlaps infants
        domain.add_population(stark_female_infant, 1, 0)
        domain.add_population(stark_female_adolescent, 1, 0)

        solvers.discrete_explicit(domain, 0, 1).execute()


def incorrect_species():
    """Should raise an exception in the solver error checker"""
    with popdyn.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=(1, 1), top=1, left=0) as domain:
        domain.add_population(stark_female_adolescent, 1, 0)
        # Add white walker as a form of mortality, but purposefully do not add white walkers to the domain
        domain.add_mortality(stark_female_adolescent, white_walker_death, time=0)

        solvers.discrete_explicit(domain, 0, 1).execute()


def single_species():
    with popdyn.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=720, left=0) as domain:
        domain.add_carrying_capacity(starks, stark_k, stark_k_data, 0, distribute=False)
        domain.add_population(starks, 10000., 0, distribute_by_habitat=True)
        solvers.discrete_explicit(domain, 0, 10).execute()


def single_species_random():
    pass


def single_species_random_dispersion():
    pass


def test_simulation():
        # Create a rectangular domain
    #===============================================================================
    shape = (720, 1280)
    with popdyn.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=720, left=0) as domain:
        # Add starks
        domain.add_carrying_capacity(starks, stark_k, stark_k_data, 0, distribute=False)

        domain.add_population(starks, 10000., 0, distribute_by_habitat=True)

        domain.add_population(popdyn.Sex('Stark', 'female'), 10000., 0, distribute_by_habitat=True)

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
        domain.add_carrying_capacity(lannister, lannister_k, lannister_k_data, 0, distribute=False)

        domain.add_population(lannister_male_infant, 10000, 0, distribute_by_habitat=True)
        domain.add_population(lannister_female_infant, 10000, 0, distribute_by_habitat=True)
        domain.add_population(lannister_male_adolescent, 200000, 0, distribute_by_habitat=True)
        domain.add_population(lannister_female_adolescent, 200000, 0, distribute_by_habitat=True)
        domain.add_population(lannister_male_adult, 500000, 0, distribute_by_habitat=True)
        domain.add_population(lannister_female_adult, 500000, 0, distribute_by_habitat=True)
        domain.add_population(lannister_male_elder, 10000, 0, distribute_by_habitat=True)
        domain.add_population(lannister_female_elder, 10000, 0, distribute_by_habitat=True)

        # Add white walkers
        domain.add_carrying_capacity(white_walker, stark_k, stark_k_data, 0, distribute=False)

        white_walker_pop0 = np.random.normal(100, 1000, shape)
        white_walker_pop0[white_walker_pop0 < 0] = 0
        white_walker_pop0[80:, :] = 0

        domain.add_population(white_walker, white_walker_pop0, 0)
        solvers.discrete_explicit(domain, 0, 1).execute()

        import dask.array as da

        # return domain.all_population_keys('stark', 0)

        return da.dstack([da.from_array(domain[ds], chunks=domain[ds].chunks) for ds in domain.all_population_keys('stark', 0)]).sum().compute()


if __name__ == '__main__':
    # Test error checking
    # ============================================================================
    try:
        no_species()
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
              "Incorrect species addition test: FAIL:\nNo Exception raised\n"
              "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    except Exception as e:
        if isinstance(e, solvers.SolverError):
            print("--------------------------------------\n"
                  "Incorrect species addition test: PASS\n"
                  "--------------------------------------")
        else:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                  "Incorrect species addition test: FAIL:\n{}\n"
                  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    print("")
    os.remove('seven_kingdoms.popdyn')

    try:
        incorrect_ages()
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
              "Incorrect age group test: FAIL:\nNo Exception raised\n"
              "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    except Exception as e:
        if isinstance(e, solvers.SolverError):
            print("--------------------------------------\n"
                  "Incorrect age group test: PASS\n"
                  "--------------------------------------")
        else:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                  "Incorrect age group test: FAIL:\n{}\n"
                  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    print("")
    os.remove('seven_kingdoms.popdyn')

    try:
        incorrect_species()
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
              "Incorrect species available test: FAIL:\nNo Exception raised\n"
              "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    except Exception as e:
        if isinstance(e, solvers.SolverError):
            print("--------------------------------------\n"
                  "Incorrect species available test: PASS\n"
                  "--------------------------------------")
        else:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                  "Incorrect species available test: FAIL:\n{}\n"
                  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    print("")
    os.remove('seven_kingdoms.popdyn')

    # Test a single species without any sex, age groups, dispersion, or randomness
    # ============================================================================
    try:
        single_species()
        print("--------------------------------------\n"
              "Single species test: PASS\n"
              "--------------------------------------")
    except Exception as e:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
              "Single species test: FAIL:\n{}\n"
              "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".format(e))
    os.remove('seven_kingdoms.popdyn')

    # #   ...add randomness
    # # ============================================================================
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
    #
    # #   ...add dispersion
    # # ============================================================================
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
    # #
    # #   ...add age groups
    # # ============================================================================
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
    #
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
