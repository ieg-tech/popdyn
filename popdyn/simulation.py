"""
Population dynamics simulation domain setup and solving

ALCES 2018
"""

import os
import numpy
import time as profile_time
from datetime import datetime
from dateutil.tz import tzlocal
from contextlib import contextmanager
import numexpr as ne
import h5py
from ast import literal_eval
from osgeo import gdal


class PopdynError(Exception):
    pass


class domain(object):
    """
    Population dynamics simulation domain.

    Each domain instance is accompanied by a .popdyn file that is used to store 2-D data in HDF5 format.

    A domain is a study area with:
        -A specific spatial domain that is parameterized with a raster to inform the extent and resolution
        -A specific temporal resolution for discrete model time steps

    Add data to the model using the built in class methods, and query the output file using popdyn.summary.
    """

    def __init__(self, path, domain_raster=None, time_step=None):
        """
        Popdyn domain constructor
        :param path: path to output popdyn file
        :param domain_raster: raster used to parameterize the spatial domain if the path does not exist
        :param kwargs:
        """
        # If the path exists, parameterize the model based on the previous run
        if os.path.isfile(path):
            self.path = path  # Files are accessed through the file property
            self.update_from_file()
        else:
            # Make sure the name suits the specifications
            self.path = self.create_file(path)

            # Read the specifications of the spatial domain using the input raster
            self.gather_domain(domain_raster)

        # Create some other attributes
        self.profiler = {}  # Used for profiling function times
        self.time_step = float(time_step)

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

    def gather_domain(self, domain_raster):
        """Parameterize the spatial domain using a raster"""
        file_source = gdal.Open(domain_raster)
        if file_source is None:
            raise PopdynError('Unable to read the source raster {}'.format(domain_raster))

        gt = file_source.GetGeoTransform()
        self.csx, self.csy = map(float, (gt[1], abs(gt[5])))
        self.shape = (file_source.RasterYSize, file_source.RasterXSize)
        del file_source

    @staticmethod
    def create_file(path):
        """Create the domain HDF5 file.  The extension .popdyn is added if it isn't already."""
        if path.split('.')[-1] != 'popdyn':
            path = path + '.popdyn'
        try:
            with h5py.File(path, mode='w', libver='latest') as f:
                assert f  # Make sure all is well
                print "Popdyn domain %s created" % (path)
        except Exception as e:
            raise PopdynError('Unable to create the file {} because:\n{}'.format(path, e))
        return path

    def dump_attrs(self):
        """Dump all of the domain attributes to the file for future loading"""
        with self.file as f:
            f.attrs.update({key: (val if isinstance(val, numpy.ndarray) else str(val))
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
                    self.__dict__[str(key)] = numpy.squeeze(val)

    @property
    @contextmanager
    def file(self):
        """Safely open the domain file using a with statement"""
        ds = h5py.File(self.path, libver='latest')
        yield ds
        ds.close()

    #============== TODO: Remove AO-Specific Django DB calls
    # def log_item(self, message, period):
    #     message = (message[:250] + '...') if len(message) > 250 else message #truncate really long messages
    #     ScenarioLog.objects.create(
    #         scenario_run=self.scenario_run,
    #         message = message,
    #         period = period)

    def _create_dataset(self, key, data, overwrite='replace'):
        """
        Internal method for writing to the disk. Overwirte behaviour supports some simple math.
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
                return f.create_dataset(key, data=numpy.zeros(shape=self.shape, dtype='float32'))[:]

    def add_group(self, key, attrs={}):
        """
        Add or update a group in the model
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
        # Limit species names to 25 chars
        if len(name) > 25:
            raise PopdynError('Species names must not exceed 25 characters.  Use something simple, like "Moose".')
        if '/' in name:
            raise PopdynError('The character "/" may not be in the species name. Use something simple, like "Moose".')

        attrs = {}

        if not hasattr(age_ranges[0], '__iter__'):
            attrs['age_ranges'] = [map(float, age_ranges)]
        else:
            attrs['age_ranges'] = [map(float, age) for age in age_ranges]

        if not hasattr(groups, '__iter__'):
            attrs['groups'] = [str(groups)]
        else:
            attrs['categories'] = list(map(float, age_ranges))

        if len(attrs['categories']) != len(attrs['age_ranges']):
            raise PopdynError('The number of age ranges ({}) does not match the number of groups ({})'
                              ''.format(len(attrs['age_ranges']), len(attrs['categories'])))

        if sex is None:
            sex = 'hermaphrodite'
        if reproduces is None:
            reproduces = [False for _ in range(len(attrs['categories']))]

        if not hasattr(reproduces, '__iter__'):
            attrs['reproduces'] = [bool(reproduces)]
        else:
            attrs['reproduces'] = list(map(bool, reproduces))

        key = '%s/%s' % (name, sex)
        self.add_group(key, attrs)

    def get_age_ranges(self, species, sex):
        """
        Retrieve a list of age ranges for the given species and sex
        :param species:
        :param sex:
        :return:
        """
        with self.file as f:
            try:
                return f['%s/%s' % (species, sex)].attrs['age_ranges']
            except KeyError:
                raise PopdynError('The specified species ({}) and/or sex ({}) '
                                  'has not been introduced into the domain'.format(species, sex))

    def get_age_groups(self, species, sex):
        """
        Retrieve a list of age group names for the given species and sex
        :param species:
        :param sex:
        :return:
        """
        with self.file as f:
            try:
                return f['%s/%s' % (species, sex)].attrs['groups']
            except KeyError:
                raise PopdynError('The specified species ({}) and/or sex ({}) '
                                  'has not been introduced into the domain'.format(species, sex))

    def get_age_reproduction(self, species, sex):
        """
        Retrieve a list of reproduction booleans for the given species and sex
        :param species:
        :param sex:
        :return:
        """
        with self.file as f:
            try:
                return f['%s/%s' % (species, sex)].attrs['reproduces']
            except KeyError:
                raise PopdynError('The specified species ({}) and/or sex ({}) '
                                  'has not been introduced into the domain'.format(species, sex))

    def reproduces(self, species, sex, age_gp):
        """Assert whether an age group reproduces"""
        gps = self.get_age_groups(species, sex)
        reproduction = self.get_age_reproduction(species, sex)
        for i, gp in enumerate(gps):
            if gp == age_gp:
                return reproduction[i]
        raise PopdynError('The age group does not exist in the model domain for {} {}'.format(species, sex))

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
                if type(population) == numpy.ndarray:
                    total_input_population = population[habitat_mask].sum()
                else:
                    total_input_population = population * habitat_mask.sum()
                _pop = ne.evaluate('(total_input_population/total_habitat)*k')

            return _pop

        # Check inputs
        try:
            population = numpy.broadcast_to(population, self.shape)
        except:
            raise PopdynError('The input population of shape %s does not match'
                              ' that of the domain: %s' % (population.shape,
                                                           self.shape))

        if not numpy.all(population == 0):
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
            fecundity = numpy.broadcast_to(numpy.float32(fecundity), self.shape)
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
            mortality = numpy.broadcast_to(numpy.float32(mortality), self.shape)
        except:
            raise PopdynError('Unable to broadcast the mortality data to the domain shape.')
        self._create_dataset(key, numpy.float32(mortality))

    # Need to re-incorporate all of this stuff!

    self.mortality_names = {}
    self.log = kwargs.get('log', False)
    self.scenario_run = kwargs.get('scenario_run')
    # Create a formal log
    self.formalLog = {'Parameterization': {'Domain size': str(self.shape),
                                           'Cell size (x)': self.csx,
                                           'Cell size (y)': self.csy,
                                           'Start time': self.start_time,
                                           'Time step': self.time_step,
                                           'Age groups': str(zip(self.ages, self.durations))},
                      'Habitat': {},
                      'Population': {},
                      'Natality': {},
                      'Mortality': {},
                      'Time': [],
                      'Solver': [datetime.now(tzlocal()).strftime('%A, %B %d, %Y %I:%M%p %Z')]}


    def add_lookup(self, key, x, y):
        """
        Add a lookup table into the data model
        Currently only supported by fecundity in solve
        """
        with self.file as f:
            f[key].attrs['lookup'] = numpy.vstack([x, y])

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
        carry_capacity = numpy.array(carry_capacity).astype('float32') * (self.csx * self.csy)
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
            ts = numpy.array(
                [float(t) for t in f['%s/%s' % (species, sex)].keys()
                 if 'k' in f['%s/%s/%s' % (species, sex, t)].keys()]
            )
            if ts.size == 0:
                return ''  # Allow a return and handle exceptions elsewhere

            t = ts[numpy.argmin(numpy.abs(ts - float(time)))]
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
                _time = numpy.sort(
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
                _time = numpy.sort(
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
                _time = numpy.sort(
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

    def collect_from_lookup(self, key, a, file_instance=None):
        """
        Return a modified array based on lookup tables from a GID
        """

        def lookup(f):
            x, y = f[key].attrs['lookup'][0, :], f[key].attrs['lookup'][1, :]
            return numpy.pad(y, 1, 'edge')[1:][numpy.digitize(a, x)]

        if file_instance is not None:
            return lookup(file_instance)
        else:
            with self.file as f:
                return lookup(f)

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
            return numpy.unique(ages)

        if file_instance is not None:
            return _get_ages(file_instance)
        else:
            with self.file as f:
                return _get_ages(f)
