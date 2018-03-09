import popdyn
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def show(ds):
    out = ds.copy()
    out[out == 0] = np.nan
    plt.imshow(out, cmap='jet')
    plt.show()

def some_random_k(shape):
    k = np.zeros(shape=shape, dtype='float32')
    np.random.seed(2);
    i = np.random.randint(0, shape[0], 5)
    np.random.seed(2);
    j = np.random.randint(0, shape[1], 5)
    k[(i, j)] = 100000
    k = gaussian_filter(k, 10)
    np.random.seed(0);
    k += np.random.normal(0, 1000, shape)
    k = gaussian_filter(k, 10)
    k[k < 0] = 0
    return k

def simple():
    pass

def complex():
    # Create some species - humans and white walkers
    #================================================================================
    humans = popdyn.Species('Human')

    human_male_infant = popdyn.AgeGroup('Human', 'Infant', 'male', 0, 3)
    human_female_infant = popdyn.AgeGroup('Human', 'Infant', 'female', 0, 3)

    human_male_adolescent = popdyn.AgeGroup('Human', 'Adolescent', 'male', 4, 12)
    human_female_adolescent = popdyn.AgeGroup('Human', 'Adolescent', 'female', 4, 12)

    human_male_adult = popdyn.AgeGroup('Human', 'Adult', 'male', 13, 60)
    human_female_adult = popdyn.AgeGroup('Human', 'Adult', 'female', 13, 60)

    # Should raise an error
    # human_male_elder = popdyn.AgeGroup('Human', 'Elder', 'male', 60, 100)

    human_male_elder = popdyn.AgeGroup('Human', 'Elder', 'male', 61, 100)
    human_female_elder = popdyn.AgeGroup('Human', 'Elder', 'female', 61, 100)

    white_walker = popdyn.Species('White Walker')


    # Create a rectangular domain
    #===============================================================================
    shape = (720, 1280)
    with popdyn.Domain('seven_kingdoms.popdyn', csx=1., csy=1., shape=shape, top=720, left=0) as domain:
        # Add human habitat
        k = some_random_k(shape)

        domain.add_k(humans, k, 0, distribute=False)

        # Add populations of each age group
        domain.add_population(human_male_infant, 10000, 0, distribute_by_habitat=True)
        domain.add_population(human_female_infant, 10000, 0, distribute_by_habitat=True)
        domain.add_population(human_male_adolescent, 200000, 0, distribute_by_habitat=True)
        domain.add_population(human_female_adolescent, 200000, 0, distribute_by_habitat=True)
        domain.add_population(human_male_adult, 500000, 0, distribute_by_habitat=True)
        domain.add_population(human_female_adult, 500000, 0, distribute_by_habitat=True)
        domain.add_population(human_male_elder, 10000, 0, distribute_by_habitat=True)
        domain.add_population(human_female_elder, 10000, 0, distribute_by_habitat=True)

        # Add a population of white walkers to the North behind a wall
        domain.add_k(white_walker, k, 0, distribute=False)

        white_walker_pop0 = np.random.normal(100, 1000, shape)
        white_walker_pop0[white_walker_pop0 < 0] = 0
        white_walker_pop0[80:, :] = 0

        domain.add_population(white_walker, white_walker_pop0, 0)


if __name__ == '__main__':
    ds = complex()
