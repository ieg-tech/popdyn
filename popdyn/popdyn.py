"""
Population dynamics simulation domain setup and model interface

Devin Cairns, 2018
"""

from __future__ import print_function
import os
from contextlib import contextmanager
from ast import literal_eval
import time as profile_time
from collections import defaultdict
import numpy as np
import numexpr as ne
import h5py
from osgeo import gdal, osr


class PopdynError(Exception):
    pass


class Domain(object):
    """
    Population dynamics simulation domain.

    Each domain instance is accompanied by a .popdyn file that is used to store 2-D data in HDF5 format.

    A domain is a study area with:
        -A specific spatial domain that is parameterized with a raster to inform the extent and resolution
        -A specific temporal resolution for discrete model time steps

    Add data to the model using the built in class methods.

    Examples:
    -----------
    >>> import popdyn

    Create an instance of the model domain, with a file path, and a raster to inform the study area

    >>> my_model = popdyn.Domain('~/usr/pd_prj/peregrine_falcon_v23.popdyn', '~/usr/pd_proj/sahtu.tif', 1)
    Popdyn domain ~/usr/pd_prj/peregrine_falcon_v23.popdyn created

    Check the resulting specs

    >>> my_model.csx  # Cell size in the x-direction
    1000.0
    >>> my_model.csy  # Cell size in the y-direction
    1000.0
    >>> my_model.shape  # Shape of the model domain (rows, cols)
    (1426, 5269)

    Add a male species to the model domain with three stage classes called 'Fledgling', 'Young', and 'Adult'.
    'Young' and 'Adult' may reproduce, but Fledgling cannot reproduce.
    Note that the age groups (years in the example) must match the units of the input time_step when
    instantiating the class. I.e. this model is solved annually.

    >>> my_model.add_species('Peregrine Falcon', ((0, 1), (2, 3), (4, 16)), ('Fledgling', 'Young', 'Adult'), 'male',
                             (False, True, True))

    Add a female species to the model domain with the same stage classes and reproduction parameters

    >>> my_model.add_species('Peregrine Falcon', ((0, 1), (2, 3), (4, 16)), ('Fledgling', 'Young', 'Adult'), 'female',
                             (False, True, True))

    Add a habitat raster to the domain. Habitat is functionally the spatially-distributed carrying capacity (k)
    of the species at a given time step.
    Alternatively, a np.ndarray may be added in the place of the k dataset, but it must be
    broadcastable to the domain.  For example. a scalar of 4 may be used, or an ndarray of shape (1426, 5269),
    which was specified when creating this model domain.
    When a raster is used, it will be transformed to match the model domain if it does not match. Where gaps, or
    raster no data values exist, the resulting k will be zero.

    >>> my_model.add_carry_capacity('Peregrine Falcon', '~/usr/pd_prj/peregrine_falcon_k_2018.tif', 2018)

    Add another habitat at a different time step to account for some change

    >>> my_model.add_carry_capacity('Peregrine Falcon', '~/usr/pd_prj/peregrine_falcon_k_2025.tif', 2025)
    Warning: the input raster has been transformed to match the model domain:
        Spatial Reference: NAD 1983 --> NAD 1983 NWT Lambert
        Cell Size (x):     0.0000659 --> 1000.0
        Cell Size (y):     0.0000742 --> 1000.0
        Top:               65.85 --> 8834724.40
        Left:              -124.95 --> -585644.0

    Add fecundity

    >>> my_model.

    Add two mortality types

    >>> my_model.
    """

    def __init__(self, popdyn_path, domain_raster=None, time_step=None):
        """
        PopDyn domain constructor.

        A raster file or an existing .popdyn file is used to construct the Domain class instance.
        All subsequent input data must be broadcastable to these specifications.
        The time step will be used to solved the model, and inform the time-based units. It is important
        to ensure Any additional time-based parameters (discrete or rate) added to the model are in
        the same units.

        :param str path: Path to a new or existing popdyn file on the disk
        :param str domain_raster: Path to an existing raster on the disk
        :param int[float] time_step: The temporal resolution of the domain
        """
        # If the path exists, parameterize the model based on the previous run
        if os.path.isfile(popdyn_path):
            self.path = popdyn_path  # Files are accessed through the file property
            self.update_from_file()
        else:
            # The rest can't be None
            if any([domain_raster is None, time_step is None]):
                raise PopdynError('If an input popdyn file is not specified, a domain raster and time step must be.')

            # Make sure the name suits the specifications
            self.path = self.create_file(popdyn_path)

            # Read the specifications of the spatial domain using the input raster
            spatial_params = self.data_from_raster(domain_raster)
            self.csx, self.csy,  self.shape, self.top, self.left, self.projection = spatial_params

        # Create some other attributes
        self.profiler = {}  # Used for profiling function times
        self.time_step = float(time_step)
        self.groups = defaultdict(dict)
        self.group_ages = defaultdict(dict)
        self.group_reproduction = defaultdict(dict)

    def time_this(func):
        """Decorator for profiling methods"""
        def inner(*args, **kwargs):
            self = args[0]
            start = profile_time.time()
            execution = func(*args, **kwargs)
            try:
                self.profiler[func.__name__] += profile_time.time() - start
            except KeyError:
                self.profiler[func.__name__] = profile_time.time() - start
            return execution
        return inner

    def data_from_raster(self, raster):
        """
        Collect data from an input raster.
        The raster will be transformed to match the  study domain
        using nearest neighbour interpolation if it doesn't match.
        Currently only the first band of rasters is supported.

        :param str raster: Path to a GDAL-supported raster dataset
        :return: The underlying raster data
        """
        file_source = gdal.Open(raster)
        if file_source is None:
            raise PopdynError('Unable to read the source raster {}'.format(raster))

        # Collect the spatial info
        gt = file_source.GetGeoTransform()
        csx, csy, top, left = float(gt[1]), float(abs(gt[5])), gt[3], gt[0]
        shape = (file_source.RasterYSize, file_source.RasterXSize)
        projection = file_source.GetProjectionRef()

        if not hasattr(self, 'shape'):
            # This is the first call, during construction, only collect raster specifications
            return csx, csy, shape, top, left, projection

        # Collecting data...see if a transform is required
        # Spatial References
        in_sr = osr.SpatialReference()
        domain_sr = osr.SpatialReference()
        in_sr.ImportFromWkt(projection)
        domain_sr.ImportFromWkt(projection)

        # Extent and position
        spatial_tests = zip([top, left, csx, csy, shape],
                            [self.top, self.left, self.csx, self.csy, self.shape])
        spatial_tests = [np.isclose(d_in, d_d) for d_in, d_d in spatial_tests]

        # They all must be true to avoid a transform
        if not all(spatial_tests + [in_sr.IsSame(domain_sr)]):
            # Transform the input dataset
            file_source = self.transform_ds(file_source)

            # Make a convenient printout
            print('Warning: the input raster has been transformed to match the model domain:\n'
                  '    Spatial Reference: {} {} --> {} {}\n'
                  '    Cell Size (x):     {:.2f} --> {:.2f}\n'
                  '    Cell Size (y):     {:.2f} --> {:.2f}\n'
                  '    Top:               {:.2f} --> {:.2f}\n'
                  '    Left:              {:.2f} --> {:.2f}'.format(
                in_sr.GetAttrValue('datum').replace('_', ' '), in_sr.GetAttrValue('projcs').replace('_', ' '),
                domain_sr.GetAttrValue('datum').replace('_', ' '), domain_sr.GetAttrValue('projcs').replace('_', ' '),
                csx, self.csx, csy, self.csy, top, self.top, left, self.left
            ))

        # Collect the raster data
        band = file_source.GetRasterBand(1)
        a = band.ReadAsArray()
        no_data = band.GetNoDataValue()

        # Destroy the swig
        file_source = None

        # Return a masked array
        return np.ma.masked_equal(a, no_data)

    def transform_ds(self, gdal_raster):
        """
        Transform a raster using nearest neighbour interpolation
        :param gdal_raster: Open gdal raster dataset
        :return: Open transformed gdal raster dataset
        """
        # Create an in-memory raster using the specs of the domain
        driver = gdal.GetDriverByName('MEM')
        outds = driver.Create('None', self.shape[1], self.shape[0], 1, gdal.GetDataTypeByName('Float32'))
        outds.SetGeoTransform((self.left, self.csx, 0, self.top, 0, self.csy * -1))
        outds.SetProjection(self.projection)

        # Load spatial references
        insrs = gdal_raster.GetProjectionRef()
        outsrs = self.projection

        # Check if spatial references are the same
        _insrs = osr.SpatialReference()
        _outsrs = osr.SpatialReference()
        _insrs.ImportFromWkt(insrs)
        _outsrs.ImportFromWkt(outsrs)
        if _insrs.IsSame(_outsrs):
            insrs, outsrs = None, None

        gdal.ReprojectImage(gdal_raster, outds, insrs, outsrs)

        # Return new in-memory raster
        return outds

    @staticmethod
    def create_file(path):
        """Create the domain HDF5 file.  The extension .popdyn is added if it isn't already."""
        if path.split('.')[-1] != 'popdyn':
            path = path + '.popdyn'
        try:
            with h5py.File(path, mode='w', libver='latest') as f:
                assert f  # Make sure all is well
                print("Popdyn domain %s created" % (path))
        except Exception as e:
            raise PopdynError('Unable to create the file {} because:\n{}'.format(path, e))
        return path

    def dump_attrs(self):
        """Dump all of the domain attributes to the file for future loading"""
        with self.file as f:
            f.attrs.update({key: (val if isinstance(val, np.ndarray) else str(val))
                            for key, val in self.__dict__.iteritems()})

    def update_from_file(self):
        """Get the model parameters from an existing popdyn file"""
        with self.file as f:
            for key, val in f.attrs.iteritems():
                if isinstance(val, basestring):
                    try:
                        self.__dict__[str(key)] = literal_eval(val)
                    except ValueError:
                        self.__dict__[str(key)] = val
                else:
                    self.__dict__[str(key)] = np.squeeze(val)

        print("Domain successfully populated from the file {}".format(self.path))

    @property
    @contextmanager
    def file(self):
        """Safely open the domain file using a with statement"""
        ds = h5py.File(self.path, libver='latest')
        yield ds
        ds.close()

    def _create_dataset(self, key, data, overwrite='replace'):
        """
        Internal method for writing to the disk. Overwrite behaviour supports some simple math.
        :param key: dataset key
        :param data: data to be written
        :param overwrite: Method to replace data.  Use one of ['replace', 'add', 'subtract', 'add_no_neg']
        :return: None
        """
        # Check the input
        overwrite = overwrite.lower()
        methods = ['replace', 'add', 'subtract', 'add_no_neg']
        if overwrite not in methods:
            raise PopdynError('Data may not be overwritten using the method "{}", '
                              'only one of the following:\n{}'.format(overwrite, '\n'.join(methods)))

        with self.file as f:
            try:
                ds = f[key]
                if not isinstance(ds, h5py.Dataset):
                    raise PopdynError('Unable to overwrite a group as a dataset using the key {}'.format(key))

                # Already exists- need to overwrite
                if overwrite == 'replace':
                    del ds
                    del f[key]
                    ds = f.create_dataset(key, data=data, compression='lzf')
                elif overwrite == 'add':
                    ds += data
                elif overwrite == 'subtract':
                    ds -= data
                elif overwrite == 'add_no_neg':
                    data -= ds
                    data[data < 0] = 0
                    ds[:] = data
            except KeyError:
                _ = f.create_dataset(key, data=data, compression='lzf')

    def get_dataset(self, key):
        """
        Get a dataset from a key, or create an empty one if it does not exist
        :param key: Dataset key
        :param file_instance:
        :return: array loaded into memory
        """
        with self.file as f:
            try:
                return f[key][:]
            except KeyError:
                return f.create_dataset(key, data=np.zeros(shape=self.shape, dtype='float32'))[:]

    def add_group(self, key, attrs={}):
        """
        Internal method to manage addition of groups in the .popdyn file
        :param key: key for group
        :param attrs: Attributes to append to the group
        :return:
        """

        def add_group(k, f):
            try:
                f[k]
            except KeyError:
                sk = '/'.join(k.split('/')[:-1])
                fk = k.split('/')[-1]
                add_group('/'.join(k.split('/')[:-1]), f)
                f[sk].create_group(fk)

        with self.file as f:
            add_group(key, f)

        f[key].attrs.update(attrs)

    def add_species(self, name, age_ranges, groups=None, sex=None, reproduces=None):
        """
        Add a species to the model domain
        :param name: Name of the species. Example "Moose".
        :param age_ranges: A tuple of age tuples (min, max) in a group of length corresponding to categories.
            Example ((0, 1), (2, 2), (3, 4), (5, 15)).
        :param groups: A tuple of age category names that matches the length of age_ranges. This may be left empty
            if the age_ranges is a single number, and it will be called "lifetime". Otherwise, it must be specified.
        :param sex: The sex of the species that is being added.  If not specified, hermaphrodite will be used.
        :param reproduces: A boolean tuple indicating with of the categories are able to reproduce. If not specified,
            all will not be capable of reproduction, unless a fecundity dataset is provided.
        :return: None
        """
        if sex is None:
            sex = 'hermaphrodite'

        # Limit species names to 25 chars
        if len(name) > 25:
            raise PopdynError('Species names must not exceed 25 characters.  Use something simple, like "Moose".')
        if '/' in name:
            raise PopdynError('The character "/" may not be in the species name. Use something simple, like "Moose".')

        if not hasattr(age_ranges[0], '__iter__'):
            self.group_ages[name][sex] = [map(float, age_ranges)]
        else:
            self.group_ages[name][sex] = [map(float, age) for age in age_ranges]

        if not hasattr(groups, '__iter__'):
            self.groups[name][sex] = [str(groups)]
        else:
            self.groups[name][sex] = list(map(float, age_ranges))

        if len(self.groups[name][sex]) != len(self.group_ages[name][sex]):
            raise PopdynError('The number of age ranges ({}) does not match the number of groups ({})'
                              ''.format(len(self.group_ages[name][sex]), len(self.groups[name][sex])))

        if reproduces is None:
            reproduces = [False for _ in range(len(self.groups[name][sex]))]

        if not hasattr(reproduces, '__iter__'):
            self.group_reproduction[name][sex] = [bool(reproduces)]
        else:
            self.group_reproduction[name][sex] = list(map(bool, reproduces))

    @property
    def species(self):
        """
        Collect a list of species in the model
        :return list: Species names
        """
        ret = self.groups.keys()
        if len(ret) == 0:
            print("No species in the model domain")
            return []
        else:
            return ret

    def reproduces(self, species, sex, age_gp):
        """Assert whether an age group reproduces"""
        try:
            index = [i for i, gp in enumerate(self.groups[species][sex]) if gp == age_gp][0]
        except (IndexError, KeyError):
            raise PopdynError('Cannot find the species, sex, or age group using {}-{}-{}'.format(species, sex, age_gp))
        return self.group_reproduction[species][sex][index]


    def get_population_keys(self, species, sex, age_group, time):
        """Return a list of population keys associated with an age group"""
        gps = self.get_age_groups(species, sex)
        keys = []
        for age, i in enumerate(self.get_age_ranges(species, sex)):
            if age_group == gps[i]:
                for _age in range(age[0], age[1] + 1):
                    keys.append('%s/%s/%s/%s' % (species, sex, time, _age))
                break
        return keys

    def add_population(self, species, sex, age_or_gp, population, time, distribute=True):
        """
        Add an array of population/cell for a given species
        :param species: Species to add the population to
        :param sex: Sex of the population
        :param age_or_gp: Age of the population, or age group (to split evenly over)
        :param population: Population data (ndarray)
        :param time: The time to insert the population into
        :param distribute: (optional) Distribute the population evenly, or using habitat if it exists in the model
        :return: None
        """
        def _distribute():
            # Try for k
            key = self.get_carry_capacity_key(species, sex, time)

            if len(key) == 0:
                # Simply redistribute evenly everywhere
                # TODO: Get a study domain mask from raster in constructor
                _pop = population.sum() / (self.shape[0] * self.shape[1])

            else:
                with self.file as f:
                    k = f[key][:]
                habitat_mask = k > 0
                total_habitat = k.sum()
                if type(population) == np.ndarray:
                    total_input_population = population[habitat_mask].sum()
                else:
                    total_input_population = population * habitat_mask.sum()
                _pop = ne.evaluate('(total_input_population/total_habitat)*k')

            return _pop

        # Check inputs
        try:
            population = np.broadcast_to(population, self.shape)
        except:
            raise PopdynError('The input population of shape %s does not match'
                              ' that of the domain: %s' % (population.shape,
                                                           self.shape))

        if not np.all(population == 0):
            sex = sex.lower()
            if sex not in ['male', 'female', 'hermaphrodite']:
                raise PopdynError('Sex must be male, female, or hermaphrodite, not %s' % sex)

            # Collect the key to the population dataset
            keys = self.get_population_keys(species, sex, age_or_gp, time)
            if len(keys) == 0:
                # Absolute age
                keys = ['%s/%s/%s/%s' % (species, sex, time, age_or_gp)]

            # Split population by number of ages in the group
            population /= len(keys)

            for key in keys:
                if distribute:
                    # Apply population selectively using k at the start time
                    self._create_dataset(key, _distribute())
                else:
                    self._create_dataset(key, population)

    def add_fecundity(self, species, age_gp, fecundity, time):
        """
        Update fecundity for a specific species, age group, and time. Sex is assumed to be female.
        It should be noted that fecundity is a rate, and must match the units of the domain time step.
        :param species: Species name
        :param age_gp: Age group name
        :param fecundity: (array-like or scalar) of the fecundity rate (offspring/time step)
        :param time: Time slice to insert the fecundity data, as it may be time-variant
        :return: None
        """
        # Check that the species as a female exists
        gps = self.get_age_groups(species, 'female')
        if age_gp not in gps:
            raise PopdynError('The age group {} does not exist for female {} in the model domain.'
                              ''.format(age_gp, species))

        # Generate a key to store the fecundity data at the specified time step
        key = '%s/%s/%s/%s/fecundity' % (species, 'female', time, age_gp)
        try:
            fecundity = np.broadcast_to(np.float32(fecundity), self.shape)
        except:
            raise PopdynError('Unable to broadcast the fecundity data to the domain shape.')

        self._create_dataset(key, fecundity)

        # Update the reproduces array to reflect the input fecundity
        reproduction = self.get_age_reproduction(species, 'female')
        for i, gp in enumerate(gps):
            if gp == age_gp:
                reproduction[i] = True
        with self.file as f:
            f['%s/%s' % (species, 'female')].attrs.update({'reproduces': reproduction})

    def add_mortality(self, species, sex, age_gp, time, mortality, name, proportion=False):
        """
        Update mortality for a specific species, age group, sex, and time.

        Mortality may be a scalar or an array, and can be within an tuple (and
        only a tuple!) for stochastic use.

        If mortality is a percentage of the population at a given time, use
        proportion=True.  If it is a fixed number, use False.
        """
        # Gather the groups
        gps = self.get_age_groups(species, sex)
        if age_gp not in gps:
            raise PopdynError('The age group {} does not exist for female {} in the model domain.'
                              ''.format(age_gp, species))

        # The name is used as the mortality key
        # Add new mortality as index
        key = '%s/%s/%s/%s/mortality/%s' % (species, sex, time, age_gp, name)
        try:
            mortality = np.broadcast_to(np.float32(mortality), self.shape)
        except:
            raise PopdynError('Unable to broadcast the mortality data to the domain shape.')
        self._create_dataset(key, np.float32(mortality))

    def add_lookup(self, key, x, y):
        """
        Add a lookup table into the data model
        Currently only supported by fecundity in solve
        """
        with self.file as f:
            f[key].attrs['lookup'] = np.vstack([x, y])

    def add_carry_capacity(self, species, carry_capacity, time=None):
        """
        Add a maximum density per unit area to a species. Note- ensure the
        units of the domain and the input raster are the same!
        """
        if '/' in species:
            raise PopdynError('"/" found in species name.')
        if time is None:
            time = self.start_time
        key = '%s/%s/k' % (species, time)

        # Convert density to population per cell
        carry_capacity = np.array(carry_capacity).astype('float32') * (self.csx * self.csy)
        if self.log is not False:
            message = 'add_carry_capacity,Adding k with a total capacity of %s at time %s\n' % (carry_capacity.sum(), time)
            self.log_item(message, time)
            with open(self.log, 'a') as logfile:
                logfile.write(message)

        self._create_dataset(key, carry_capacity)

    def get_carry_capacity_key(self, species, sex, time):
        """
        Find the closest carrying capacity in time and return the key
        """
        with self.file as f:
            # Get all times with k
            ts = np.array(
                [float(t) for t in f['%s/%s' % (species, sex)].keys()
                 if 'k' in f['%s/%s/%s' % (species, sex, t)].keys()]
            )
            if ts.size == 0:
                return ''  # Allow a return and handle exceptions elsewhere

            t = ts[np.argmin(np.abs(ts - float(time)))]
            return '%s/%s/%s/k' % (species, sex, t)

    def check_time_slice(self, species, group, s_time, file_instance=None):
        """
        Check that the time slice exists in the model, and populate with
        fecundity/mortality from a previous time slice if not.
        """
        if '/' in species:
            raise PopdynError('"/" found in species name.')

        def get_keys(f):
            # Ensure keys exist
            self.assert_group('%s/%s/%s' % (species, s_time, 'male'))
            self.assert_group('%s/%s/%s' % (species, s_time, 'female'))
            groups = f['%s/%s' % (species, s_time)].keys()

            if group in groups:
                fec = '%s/%s/%s' % (species, s_time, group)
            else:
                # Use previous
                _time = np.sort(
                    [int(t) for t in range(self.start_time,
                                           s_time + self.time_step,
                                           self.time_step)
                     if group in f['%s/%s' % (species, t)].keys()]
                )[-1]
                fec = '%s/%s/%s' % (species, _time, group)
            if group in f['%s/%s/%s' % (species, s_time, 'male')].keys():
                male_mrt = '%s/%s/%s/%s' % (species, s_time, 'male', group)
            else:
                # Use previous
                _time = np.sort(
                    [int(t) for t in range(self.start_time,
                                           s_time + self.time_step,
                                           self.time_step)
                     if group in f['%s/%s/%s' % (species, t, 'male')].keys()]
                )[-1]
                male_mrt = '%s/%s/%s/%s' % (species, _time, 'male', group)
            if group in f['%s/%s/%s' % (species, s_time, 'female')].keys():
                female_mrt = '%s/%s/%s/%s' % (species, s_time, 'female', group)
            else:
                # Use previous
                _time = np.sort(
                    [int(t) for t in range(self.start_time,
                                           s_time + self.time_step,
                                           self.time_step)
                     if group in f['%s/%s/%s' % (species, t, 'female')].keys()]
                )[-1]
                female_mrt = '%s/%s/%s/%s' % (species, _time, 'female', group)
            return fec, male_mrt, female_mrt

        if file_instance is not None:
            fec, male_mrt, female_mrt = get_keys(file_instance)
        else:
            with self.file as f:
                fec, male_mrt, female_mrt = get_keys(f)

        return fec, male_mrt, female_mrt

    def get_ages(self, species, t, file_instance=None):
        """
        Get all existing ages at current time step
        """

        def _get_ages(f):
            ages = []
            for sex in ['male', 'female']:
                for key in f['%s/%s/%s' % (species, t, sex)].keys():
                    try:
                        ages.append(int(key))
                    except ValueError:
                        continue
            return np.unique(ages)

        if file_instance is not None:
            return _get_ages(file_instance)
        else:
            with self.file as f:
                return _get_ages(f)

# Species [1]
#    |
#    sex [2]
#      |
#      if female: (fecundity,
#      |
#      age group [gp, n]
#          |
#          mortality [n]
#          |
#          dispersal [n]
#          |
#          population [range(gp.min(), gp.max() + 1)]
#          |
#          reproduces [1]
class Species(object):

    def __init__(self, name):
        pass
