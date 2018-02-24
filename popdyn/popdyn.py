"""
Population dynamics simulation domain setup and model interface

Devin Cairns, 2018
"""

from __future__ import print_function
import os
from contextlib import contextmanager
import time as profile_time
import pickle
from string import punctuation
import numpy as np
import h5py
from tempfile import _get_candidate_names as uid
from osgeo import gdal, osr
import dispersal


class PopdynError(Exception):
    pass


class Domain(object):
    """
    Population dynamics simulation domain. A domain has all of the physical and temporal characteristics of an
    ecological system. Conceptualizing and creating a domain is the first step when making a popdyn model, and is
    required before any species may be created (the domain instance is a required input for species).

    Each domain instance is accompanied by a .popdyn file that is used to store 2-D data in HDF5 format.

    A domain is a study area with:
        -A specific spatial domain that is parameterized with a raster to inform the extent and resolution
        -A specific temporal resolution for discrete model time steps

    Add data to the model using the built in class methods.
    """

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

    def save(func):
        """Decorator for saving attributes to the popdyn file"""
        def inner(*args, **kwargs):
            self = args[0]
            execution = func(*args, **kwargs)

            self.dump()

            return execution
        return inner

    def __init__(self, popdyn_path, domain_raster=None, **kwargs):
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
            self.load()
        else:
            self.build_new(domain_raster, kwargs)

    @save
    def build_new(self, domain_raster, **kwargs):
        """Used in the constructor to build the domain using input arguments"""
        # Make sure the name suits the specifications
        self.path = self.create_file(popdyn_path)

        if domain_raster is None:
            # Try and build the domain using keyword arguments
            self.__dict__.update(kwargs)

            try:
                if not hasattr(self, 'mask'):
                    self.mask = np.ones(self.shape)
                if not hasattr(self, 'projection'):
                    self.projection = ''
                else:
                    # Projection must be a SRID
                    sr = osr.SpatialReference()
                    try:
                        sr.ImportFromEPSG(self.projection)
                    except TypeError:
                        raise PopdynError('Only a SRID is an accepted input projection argument')
                    self.projection = sr.ExportToWkt()
            except AttributeError:
                raise PopdynError('If an input popdyn file and domain raster are not specified, '
                                  'the following keyword args must be included (at a minimum):\n'
                                  'csx, csy, shape, top, left')

        else:
            # Read the specifications of the spatial domain using the input raster
            spatial_params = self.data_from_raster(domain_raster)
            self.csx, self.csy, self.shape, self.top, self.left, self.projection, self.mask = spatial_params

        # Create some other attributes
        self.profiler = {}  # Used for profiling function times
        self.species = {}

    @staticmethod
    def raster_as_array(raster):
        """Collect the underlying array from a raster data source"""
        # Open the file
        file_source = gdal.Open(raster)
        if file_source is None:
            raise PopdynError('Unable to read the source raster {}'.format(raster))

        # Collect the raster data
        band = file_source.GetRasterBand(1)
        a = band.ReadAsArray()
        no_data = band.GetNoDataValue()

        # Destroy the swig
        file_source = None

        # Return a masked array
        return np.ma.masked_equal(a, no_data)

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
            # This is the first call, during construction, only collect raster specifications and a mask
            mask = ~self.raster_as_array(raster).mask
            return csx, csy, shape, top, left, projection, mask

        # Collecting data...see if a transform is required
        # Extent and position
        spatial_tests = zip([top, left, csx, csy, shape],
                            [self.top, self.left, self.csx, self.csy, self.shape])
        spatial_tests = [np.isclose(d_in, d_d) for d_in, d_d in spatial_tests]

        # Spatial References
        if self.projection == '':
            # Cannot change the spatial reference if it is not known
            spatial_tests += [True]
        else:
            in_sr = osr.SpatialReference()
            domain_sr = osr.SpatialReference()
            in_sr.ImportFromWkt(projection)
            domain_sr.ImportFromWkt(self.projection)
            spatial_tests += [in_sr.IsSame(domain_sr)]

        # They all must be true to avoid a transform
        if not all(spatial_tests):
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

        return self.raster_as_array(file_source)

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
        if self.projection == '' or _insrs.IsSame(_outsrs):
            insrs, outsrs = None, None

        gdal.ReprojectImage(gdal_raster, outds, insrs, outsrs)

        # Return new in-memory raster
        return outds

    @staticmethod
    def create_file(path):
        """Create the domain HDF5 file.  The extension .popdyn is appended if it isn't already"""
        if path.split('.')[-1] != 'popdyn':
            path = path + '.popdyn'
        try:
            with h5py.File(path, mode='w', libver='latest') as f:
                assert f  # Make sure all is well
                print("Popdyn domain %s created" % (path))
        except Exception as e:
            raise PopdynError('Unable to create the file {} because:\n{}'.format(path, e))
        return path

    def dump(self):
        """Dump all of the domain instance to the file"""
        with self.file as f:
            f.attrs['instance'] = pickle.dumps(self.__dict__)

    def load(self):
        """Get the model parameters from an existing popdyn file"""
        with self.file as f:
            self.__dict__.update(pickle.loads(f.attrs['instance']))

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

    def get_dataset(self, key, fill_if_missing=False, fill_value=0):
        """
        Get a dataset from a key, and create an empty one if it does not exist
        :param str key: Dataset key
        :param bool fill_if_missing: Create a new dataset if the query doesn't exist
        :param float fill_value: Value to fill the new dataset if it is created
        :return: A view into the on-disk array
        """
        with self.file as f:
            try:
                return f[key]
            except KeyError:
                if fill_if_missing:
                    ds = f.create_dataset(key, shape=self.shape, dtype='float32')
                    ds[:] = fill_value
                    return ds
                else:
                    raise PopdynError('The dataset "{}" does not exist'.format(key))

    def add_directory(self, key):
        """
        Internal method to manage addition of HDF5 groups in the .popdyn file
        Groups are added recursively in order to add the entire directory tree
        :param str key: key for group
        :return: None
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

    @save
    def add_population(self, species, data, time, **kwargs):
        """
        Add population data for a given species at a specific time slice in the domain
        :param Species species: A Species instance
        :param type data: Population data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)
        :param time: The time slice to insert the population data into

        :param dict kwargs:
            :distribute: Divide the input population data evenly among the domain elements
            :distribute_by_habitat: Divide the input population among domain elements using
                carrying capacity as a covariate

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


class Species(object):

    def __init__(self, name):
        """

        :param str name: Name of the species. Example "Moose"
        """
        # Limit species names to 25 chars
        if len(name) > 25:
            raise PopdynError('Species names must not exceed 25 characters. '
                              'Use something simple, like "Moose".')

        self.name = name
        self.name_key = name.strip().translate(None, punctuation + ' ').lower()

        # No dispersal by default
        self.dispersal = {}

        # Reproduces by default
        self.reproduces = True

    def add_dispersal(self, dispersal_type, args):
        if dispersal_type not in dispersal.METHODS.keys():
            raise PopdynError('The dispersal method {} has not been implemented'.format(dispersal_type))

        self.dispersal[dispersal_type] = args


class Sex(Species):

    def __init__(self, name, sex):

        self.sex = sex

        super(Sex, self).__init__(name)


class AgeGroup(Sex):

    def __init__(self, species_name, group_name, sex, min_age, max_age):

        self.group_name = group_name

        # Make the ages a range array
        self.ages = np.arange(min_age, max_age + 1)

        super(AgeGroup, self).__init__(species_name, sex)
