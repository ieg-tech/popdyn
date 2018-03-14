"""
Population dynamics simulation domain setup and model interface

Devin Cairns, 2018
"""

from __future__ import print_function
import os
import time as profile_time
import pickle
from ast import literal_eval
from string import punctuation
import numpy as np
from collections import defaultdict
import dask.array as da
import h5py
from osgeo import gdal, osr
import dispersal
import dynamic


class PopdynError(Exception):
    pass


def rec_dd():
    """Recursively update defaultdicts to avoid key errors"""
    return defaultdict(rec_dd)


class Domain(object):
    """
    Population dynamics simulation domain. A domain instance is used to:
        -Manage a .popdyn file to store 2-D data in HDF5 format
        -Provide a study area extent and resolution
        -Prepare and save spatially-distributed data
        -Relate spatially-distributed data to species instances (Species, Sex, AgeGroup)
        -Call code to solve the population over time
    Species are added to a model domain through use of any of the these methods,
    which coincide with the respective datasets.
        add_population
        add_mortality
        add_carrying_capacity
    Once added to a domain, species become stratified by their name, sex, and age groups, and are
    able to inherit mortality and carrying capacity based on a species - sex - age group hierarchy
    """
    # Decorators
    #-------------------------------------------------------------------------
    def time_this(func):
        """Decorator for profiling methods"""
        def inner(*args, **kwargs):
            instance = args[0]
            start = profile_time.time()
            execution = func(*args, **kwargs)
            try:
                instance.profiler[func.__name__] += profile_time.time() - start
            except KeyError:
                instance.profiler[func.__name__] = profile_time.time() - start
            return execution
        return inner

    def save(func):
        """Decorator for saving attributes to the popdyn file"""
        def inner(*args, **kwargs):
            instance = args[0]
            execution = func(*args, **kwargs)

            instance.dump()

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
            self.file = h5py.File(self.path, libver='latest')
            self.load()
        else:
            self.build_new(popdyn_path, domain_raster, **kwargs)

    # File management and builtins
    # ==================================================================================================
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

    def add_directory(self, key):
        """
        Internal method to manage addition of HDF5 groups in the .popdyn file
        Groups are added recursively to create the entire directory tree
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

        add_group(key, self.file)

    def dump(self):
        """Dump all of the domain instance to the file"""
        self.file.attrs['self'] = np.void(pickle.dumps({key: val for key, val in
                                                        self.__dict__.items() if key not in ['file', 'path']}))

    def load(self):
        """Get the model parameters from an existing popdyn file"""
        self.__dict__.update(pickle.loads(self.file.attrs['self']))

        print("Domain successfully populated from the file {}".format(self.path))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.file.close()

    def __del__(self):
        # As a precaution
        try:
            self.file.close()
        except:
            pass

    def __getitem__(self, item):
        return self.file[item]

    def __setitem__(self, key, data):
        try:
            del self.file[key]
        except KeyError:
            pass
        _ = self.file.create_dataset(key, data=np.broadcast_to(data, self.shape),
                                     compression='lzf', chunks=self.chunks)

    def __repr__(self):
        return 'Popdyn model domain of shape {} with\n{}'.format(self.shape, '\n'.join(self.species_names))

    # Data parsing and checking methods
    #==================================================================================================
    @save
    def build_new(self, popdyn_path, domain_raster, **kwargs):
        """Used in the constructor to build the domain using input arguments"""
        # Make sure the name suits the specifications
        self.path = self.create_file(popdyn_path)
        self.file = h5py.File(self.path, libver='latest')

        if domain_raster is None:
            # Try and build the domain using keyword arguments
            self.__dict__.update(kwargs)

            try:
                if not hasattr(self, 'mask'):
                    self.mask = np.ones(self.shape, dtype='bool')
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

            # Domain cannot be empty
            if self.mask.sum() == 0:
                raise PopdynError('The domain cannot be empty. Check the input raster "{}"'.format(domain_raster))

        # Create some other attributes
        self.profiler = {}  # Used for profiling function times
        self.species = rec_dd()
        self.population = rec_dd()
        self.mortality = rec_dd()
        self.carrying_capacity = rec_dd()
        # Inheritance may be turned off completely
        if not hasattr(self, 'avoid_inheritance'):
            self.avoid_inheritance = False
        # Chunk size is specified for data storage and dask scheduling
        if not hasattr(self, 'chunks'):
            chunks = [2000, 2000]  # ~16MB chunks should ensure large domains will not overflow RAM
            if chunks[0] > self.shape[0]:
                chunks[0] = self.shape[0]
            if chunks[1] > self.shape[1]:
                chunks[1] = self.shape[1]
            self.chunks = tuple(chunks)

    def raster_as_array(self, raster, as_mask=False):
        """
        Collect the underlying array from a raster data source
        :param raster: input raster file
        :param as_mask: return only a mask of where data values exist in the raster
        :return: numpy.ndarray
        """
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

        if as_mask:
            return a != no_data
        else:
            # Return an array with no data converted to 0
            a[(a == no_data) | ~self.mask] = 0
            return a

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
            return csx, csy, shape, top, left, projection, self.raster_as_array(raster, True)

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
    def get_time_input(time):
        """Make sure an input time is an integer"""
        try:
            _time = int(time)
        except:
            raise PopdynError("Unable to parse the input time of type {}".format(type(time).__name__))

        if not np.isclose(_time, float(time)):
            raise PopdynError('Input time may only be an integer (whole number)')

        return _time

    @time_this
    def get_data_type(self, data):
        """Parse a data argument to retrieve a domain-shaped matrix"""
        # First, try for a file
        if isinstance(data, basestring) and os.path.isfile(data):
            return self.data_from_raster(data)

        # Try and construct a matrix from data
        else:
            try:
                return np.asarray(data).astype('float32')
            except:
                raise PopdynError('Unable to parse the input data {} into the domain'.format(data))

    # Domain-species interaction methods
    #==================================================================================================

    # These are the methods to call for primary purposes
    #---------------------------------------------------
    @save
    def add_population(self, species, data, time, **kwargs):
        """
        Add population data for a given species at a specific time slice in the domain
        :param Species species: A Species instance
        :param data: Population data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)
        :param time: The time slice to insert the population data into

        :param kwargs:
            :distribute: Divide the input population data evenly among the domain elements (default is True)
            :distribute_by_habitat: Divide the input population linearly among domain elements using
                the covariate carrying capacity (Default is False)
            :overwrite: one of ['replace', 'add', 'subtract'] to send to self._add_data() (Default is replace)
            :discrete_age: Apply the population to a single discrete age in the age group
        :return: None
        """
        self.introduce_species(species)  # Introduce the species
        time = self.get_time_input(time)  # Gather time
        data = self.get_data_type(data)  # Parse input data

        discrete_age = kwargs.get('discrete_age', None)


        ages = species.age_range
        if ages is None:
            # No ages are used, this is simply a population tied to a species or sex
            ages = [None]

        if discrete_age is not None:
            if discrete_age not in ages:
                raise PopdynError('The input discrete age {} does not exist for the'
                                  ' species {} {}s of (key) {}'.format(
                    discrete_age, species.name, species.sex, species.group_key
                ))
            ages = [discrete_age]


        distribute_by_habitat = kwargs.get('distribute_by_habitat', False)
        if distribute_by_habitat:
            distribute_by_habitat = 'carrying_capacity'
        else:
            distribute_by_habitat = None

        # Divide the input data by the number of ages
        data /= len(ages)

        for age in ages:
            key = '{}/{}/{}/{}/{}'.format(species.name_key, species.sex, species.group_key, time, age)
            self.population[species.name_key][species.sex][species.group_key][time][age] = key

            self._add_data(key, data,
                distribute=kwargs.get('distribute', True),
                distribute_by_co=distribute_by_habitat,
                overwrite=kwargs.get('overwrite', 'replace')
            )

    @save
    def add_carrying_capacity(self, species, carrying_capacity, data, time, **kwargs):
        """
        Carrying capacity is added to the domain with species and CarryingCapacity objects in tandem.
        Multiple carrying capacity datasets may added to a single species,
            as they are stacked in the domain for each species (or sex/age group).

        :param Species species: Species instance
        :param CarryingCapacity carrying_capacity: Carrying capacity instance.
        :param data: Carrying Capacity data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)
        :param time: The time slice to insert the carrying capacity
        :kwargs:
            :distribute: Divide the input data evenly over the domain
            :is_density: The input data are a density, and not an absolute population (Default is False)
            :overwrite: Overwrite any existing data (Default 'replace')
        """
        if not isinstance(carrying_capacity, CarryingCapacity):
            raise PopdynError('Expected a CarryingCapacity instance, not "{}"'.format(
                type(carrying_capacity).__name__)
            )

        self.introduce_species(species)  # Introduce the species
        time = self.get_time_input(time)  # Gather time
        data = self.get_data_type(data)  # Parse input data

        if kwargs.get('is_density', False):
            # Convert density to population per cell
            data *= self.csx * self.csy

        k_key = carrying_capacity.name_key

        key = '{}/{}/{}/{}/{}'.format(species.name_key, species.sex, species.group_key, time, k_key)
        # The instance and the key are added
        self.carrying_capacity[species.name_key][species.sex][species.group_key][time][k_key] = (carrying_capacity,
                                                                                                  key)

        # Add the data to the file
        self._add_data(key, data,
            distribute=kwargs.get('distribute', True),
            overwrite=kwargs.get('overwrite', 'replace')
        )

    @save
    def add_mortality(self, species, mortality, time, data=None, **kwargs):
        """
        Mortality is added to the domain with species and Mortality objects in tandem.
        Multiple mortality datasets may added to a single species,
            as they are stacked in the domain for each species (or sex/age group).

        :param Species species: Species instance
        :param Morality mortality: Mortality instance.
        :param data: Mortality data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)
        :param time: The time slice to insert mortality
        :kwargs:
            :distribute: Divide the input data evenly over the domain
            :overwrite: Overwrite any existing data (Default 'replace')
        """
        if not isinstance(mortality, Mortality):
            raise PopdynError('Expected a mortality instance, not "{}"'.format(type(mortality).__name__))

        self.introduce_species(species)  # Introduce the species
        time = self.get_time_input(time)  # Gather time
        if data is not None:
            data = self.get_data_type(data)  # Parse input data
        else:
            # Mortality must be tied to a species if there are no input data
            if not hasattr(mortality, 'species'):
                raise PopdynError('Mortality data must be provided with {}, as it is not attached to a species'.format(
                    mortality.name
                ))

        m_key = mortality.name_key

        # If mortality is defined only by a lookup table, the key is None
        if data is None:
            key = None
        else:
            key = '{}/{}/{}/{}/{}'.format(species.name_key, species.sex, species.group_key, time, m_key)

            # Add the data to the file
            self._add_data(key, data,
                distribute=kwargs.get('distribute', True),
                overwrite=kwargs.get('overwrite', 'replace')
            )

        self.mortality[species.name_key][species.sex][species.group_key][time][m_key] = (mortality, key)

    @save
    def remove_species(self, species):
        """Remove the species from the domain, including all respective data"""
        # Remove all datasets
        try:
            del self.file[species.name_key]
        except KeyError:
            print("Warning: the species {} was not in the domain".format(species.name))
            pass

        # Remove species from instance
        del self.species[species.name_key]
        del self.mortality[species.name_key]
        del self.population[species.name_key]
        del self.carrying_capacity[species.name_key]

        print("Successfully removed {} from the domain".format(species.name))

    # These are factories
    # ---------------------------------------------------
    # Recursive functions for traversing species in the domain
    @property
    def species_names(self):
        def is_instance(d):
            for key, val in d.items():
                if isinstance(val, dict):
                    is_instance(val)
                elif any([isinstance(val, obj) for obj in [Species, Sex, AgeGroup]]):
                    names.append(val.name)

        names = []
        is_instance(self.species)
        return np.unique(names)

    @property
    def species_instances(self):
        """Collect all species instances added to the model domain"""
        def is_instance(d):
            for key, val in d.items():
                if isinstance(val, dict):
                    is_instance(val)
                elif any([isinstance(val, obj) for obj in [Species, Sex, AgeGroup]]):
                    instances.append(val)

        instances = []
        is_instance(self.species)
        return instances

    @staticmethod
    def deconstruct_key(key):
        """Deconstruct a dataset key into hashes that will work with the Domain instance"""
        return [val if isinstance(val, basestring) else literal_eval(val) for val in key.split('/')]

    def discrete_ages(self, species_name, sex):
        """Return all discrete ages for a species in the model domain"""
        def is_age(d):
            for key, val in d.items():
                if isinstance(val, dict):
                    is_age(val)
                elif any([isinstance(val, obj) for obj in [Species, Sex, AgeGroup]]):
                    ages.append(val.age_range)

        ages = []
        is_age(self.species[species_name][sex])
        age_ret = []
        for age in ages:
            age_ret += age
        return np.sort(age_ret)

    def introduce_species(self, species):
        """INTERNAL> Add a species to the model domain"""
        # Notify console if this species is entirely new
        if species.name_key not in self.species.keys():
            print("Species {} introduced into the domain".format(species.name))

        self.species[species.name_key][species.sex][species.group_key] = species

    @time_this
    def _add_data(self, key, data, **kwargs):
        """
        INTERNAL method for writing to the disk.
        Overwrite behaviour supports some simple math.

        :param key: key of the dataset in the popdyn file
        :param np.ndarray data: Population data to save. This must be filtered through Domain.get_data_type in advance

        :param kwargs:
            :distribute: Divide the input data evenly among the domain elements
            :distribute_by_co: Divide the input data linearly among domain elements using
                a covariate dataset (population, mortality, carrying_capacity, [or None])
            :overwrite: Use one of ['replace', 'add', 'subtract'] for data override behaviour
        :return: None
        """
        # Make sure overwrite is valid
        overwrite_methods = ['replace', 'add', 'subtract']
        overwrite = kwargs.get('overwrite', 'replace').lower()
        if overwrite not in overwrite_methods:
            raise PopdynError('Data may not be overwritten using the method "{}", '
                              'only one of the following:\n{}'.format(overwrite, '\n'.join(overwrite_methods)))

        # Check if an input covariate name is valid
        distribute_by_co = kwargs.get('distribute_by_co', None)
        if distribute_by_co not in ['population', 'mortality', 'carrying_capacity', None]:
            raise PopdynError('Unsupported covariate "{}"'.format(distribute_by_co))

        # Distribute if necessary- evenly or using a covariate
        if distribute_by_co is not None:
            # Change the key into arguments that may be used to collect the covariate
            species_key, sex, group, time, ds_type = self.deconstruct_key(key)[:5]
            # The covariate is collected using a function call that is constructed using the data type name
            co = getattr(self, 'get_{}'.format(distribute_by_co))(species_key, sex, group, time)
            # More than one dataset may have been returned
            co = [self[_co[1]] if isinstance(_co, tuple) else self[_co] for _co in co]

            # Use dask to aggregate data because this could be memory-heavy
            co = da.dstack([da.from_array(ds, ds.chunks) for ds in co]).sum(axis=-1).compute()

            # Compute the sum. If no data are available or the dataset is empty, the sum will be 0
            co_sum = co.sum()
            if co_sum == 0:
                raise PopdynError('The closest {} covariate backwards in time is empty and cannot be used to'
                                  ' distribute {} at time {}'.format(distribute_by_co, ds_type, time))

            data = np.sum(data) * (co / co_sum)

        elif kwargs.get('distribute', True):
            # Split all data evenly among active cells
            data = (np.sum(data) / np.sum(self.mask)) * self.mask.astype('float32')  # The mask is used to broadcast

        # Try to collect the dataset directly
        try:
            ds = self[key]
        except KeyError:
            self[key] = data
            return

        # Activate a replacement method because the dataset exists
        if overwrite == 'replace':
            self[key] = data
        elif overwrite == 'add':
            ds[:] = np.add(ds, data)
        elif overwrite == 'subtract':
            ds[:] = np.subtract(ds, data)

    def get_mortality(self, species_key, sex, group_key, time, avoid_inheritance=False):
        """
        Collect the mortality instance - key pairs
        :param str species_key:
        :param str sex:
        :param str group_key:
        :param int time:
        :param avoid_inheritance: Do not inherit a dataset from the sex or species
        :return: list of instance - key pairs
        """
        time = self.get_time_input(time)

        # Collect the dataset keys using inheritance
        if avoid_inheritance or self.avoid_inheritance:
            groups = [group_key]
            sexes = [sex]
        else:
            groups = [None, group_key]
            sexes = [None, sex]

        datasets = {}
        for group in groups:
            for _sex in sexes:
                datasets.update(self.mortality[species_key][_sex][group][time])

        return datasets.values()

    def get_carrying_capacity(self, species_key, sex, group_key, time, snap_to_time=True, avoid_inheritance=False):
        """
        Collect the carrying capacity instance - key pairs
        :param str species_key:
        :param str sex:
        :param str group_key:
        :param int time:
        :param snap_to_time: If the dataset queried does not exist, it can be snapped backwards in time to the nearest available dataset
        :param avoid_inheritance: Do not inherit a dataset from the sex or species
        :return: list of instance - key pairs
        """
        time = self.get_time_input(time)

        if avoid_inheritance or self.avoid_inheritance:
            groups = [group_key]
            sexes = [sex]
        else:
            groups = [None, group_key]
            sexes = [None, sex]

        if snap_to_time:
            # Modify time to the location closest backwards in time with data
            times = []
            for group in groups:
                for _sex in sexes:
                    times += self.carrying_capacity[species_key][_sex][group].keys()

            times = np.unique(times)
            delta = time - times
            backwards = delta >= 0
            # If no times are available, time is not updated and an empty list will be returned
            if backwards.sum() > 0:
                times = times[backwards]
                delta = delta[backwards]
                i = np.argmin(delta)
                time = times[i]

        # Collect the instance - key pairs
        datasets = {}
        for group in groups:
            for _sex in sexes:
                datasets.update(self.carrying_capacity[species_key][_sex][group][time])

        return datasets.values()

    def get_population(self, species_key, time, sex=None, group_key=None, age=None):
        """
        Collect the population key of a species/sex/group at a given time if it exists
        :param species_key:
        :param time:
        :param sex:
        :param group_key:
        :param age:
        :return: key or None
        """
        time = self.get_time_input(time)

        key = self.population[species_key][sex][group_key][time][age]
        if len(key) > 0:
            return key
        else:
            return None

    def all_population_keys(self, species_key, time, sex=None, group_key=None):
        """
        Collect the population keys of a species at a given time. All children keys are returned.
        :param species_key:
        :param time:
        :param sex:
        :param group_key:
        :param age:
        :return: list of keys
        """
        time = self.get_time_input(time)

        # sex, group and age may be None, which enables return of all keys
        sexes = [sex]
        if sex is None:
            sexes += ['male', 'female']
        groups = [group_key]
        if group_key is None:
            for _sex in sexes:
                groups += [key for key in self.population[species_key][_sex].keys() if key is not None]

        keys = []
        for _sex in sexes:
            for group in groups:
                age_keys = self.population[species_key][_sex][group][time].keys()
                for age in age_keys:
                    val = self.population[species_key][_sex][group][time][age]
                    if len(val) > 0:  # Could be a defaultdict, or string
                        keys.append(val)

        return np.unique(keys)


# Species classes
#   Species: No age or sex information.
#      |
#       Sex: Species with sex specified. Inherits all carrying capacity and
#            mortality from the same species when added to the domain.
#        |
#         AgeGroup: Species with a sex and a range of ages. Inherits all carrying capacity, mortality,
#                   and fecundity from the same species/sex when added to the domain.
# ========================================================================================================


class Species(object):

    def __init__(self, name, **kwargs):
        """

        :param str name: Name of the species. Example "Moose"
        :param kwargs:
            :param bool contributes_to_density: This species contributes to density calculations (default is True)
        """
        # Limit species names to 25 chars
        if len(name) > 25:
            raise PopdynError('Species names must not exceed 25 characters. '
                              'Use something simple, like "Moose".')

        self.name = name
        self.name_key = name.strip().translate(None, punctuation + ' ').lower()

        # No dispersal by default
        self.dispersal = {}

        # sex and age group are none
        self.sex = self.group_key = None

    def add_dispersal(self, dispersal_type, args):
        if dispersal_type not in dispersal.METHODS.keys():
            raise PopdynError('The dispersal method {} has not been implemented'.format(dispersal_type))

        self.dispersal[dispersal_type] = args

    @property
    def age_range(self):
        try:
            return range(self.min_age, self.max_age + 1)
        except AttributeError:
            return None


class Sex(Species):

    def __init__(self, name, sex, **kwargs):
        """
        Species --> Sex constructor which adds reproductive attributes
        :param str name: Name of the species. Example "Moose"
        :param sex: Name of sex, must be one of 'male', 'female'
        :param kwargs: Attributes related to fecundity:
            :param float fecundity: Rate of fecundity (default is 0)
            :param float birth_ratio: Ratio of females to males born (default is 0.5)
            :param float density_fecundity_threshold: Density at which fecundity becomes affected by density (default is 0)
            :param iterable fecundity_lookup: Lookup iterable of x, y pairs to define a relationship between density
                and fecundity. The x-values are density, and the y-values are a coefficient to modify fecundity.
        """

        super(Sex, self).__init__(name, **kwargs)

        if sex not in ['male', 'female']:
            raise PopdynError('Sex must be one of "male" or "female"')

        self.sex = sex.lower()

        self.fecundity = kwargs.get('fecundity', 0)
        self.birth_ratio = kwargs.get('birth_ratio', 0.5)
        self.density_fecundity_threshold = kwargs.get('density_fecundity_threshold', 0)

        self.fecundity_lookup = dynamic.collect_lookup(kwargs.get('fecundity_lookup', None))\
            if kwargs.get('fecundity_lookup', None) is not None else None

        self.__dict__.update(kwargs)

    def random_fecundity(self, **kwargs):
        """
        Apply a type of random variability to fecundity
        :param type: Variance type. Use one of the random distribution generator methods in popdyn.dynamic.
        :param kwargs: :param tuple args: parameters for the variance algorithm (variable)
        :return: None
        """
        if type not in dynamic.RANDOM_METHODS.keys():
            raise PopdynError('Unsupported random distribution generator "{}". Choose from:\n{}'.format(
                type, '\n'.join(dynamic.RANDOM_METHODS.keys()))
            )

        self.fecundity_random = type
        self.fecundity_random_args = kwargs.get('args', None)

    def random_birth_ratio(self, **kwargs):
        """
        Apply a type of random variability to the birth ratio
        :param type: Variance type. Use one of the random distribution generator methods in popdyn.dynamic.
        :param kwargs: :param tuple args: parameters for the variance algorithm (variable)
        :return: None
        """
        if type not in dynamic.RANDOM_METHODS.keys():
            raise PopdynError('Unsupported random distribution generator "{}". Choose from:\n{}'.format(
                type, '\n'.join(dynamic.RANDOM_METHODS.keys()))
            )

        self.birth_random = type
        self.birth_random_args = kwargs.get('args', None)


class AgeGroup(Sex):

    def __init__(self, species_name, group_name, sex, min_age, max_age, **kwargs):
        if len(group_name) > 25:
            raise PopdynError('Group names must not exceed 25 characters. '
                              'Use something simple, like "Yearling".')

        super(AgeGroup, self).__init__(species_name, sex, **kwargs)

        self.group_name = group_name
        self.group_key = group_name.strip().translate(None, punctuation + ' ').lower()

        self.min_age = min_age
        self.max_age = max_age


# Carrying capacity, mortality drivers, and fecundity.
# These classes parameterize specific information, and enable
# inter-species relationships and randomness in the domain
# =======================================================================================================


class CarryingCapacity(object):
    """
    Carrying capacity is a template that may be applied to any species in a model domain. Multiple carrying capacity
        instances may be added to a given species.

    Carrying capacity instances contain relevant attributes, and also serve to use species to modify carrying capacity.
    Carrying capacity is added to a Domain along with a species that utilizes the carrying capacity,
        and a dataset that defines the carrying capacity.
    """

    def __init__(self, name, **kwargs):
        # Limit carrying capacity names to 25 chars
        if len(name) > 25:
            raise PopdynError('Carrying capacity names must not exceed 25 characters. '
                              'Use something simple, like "Thicket".')

        self.name = name
        self.name_key = name.strip().translate(None, punctuation + ' ').lower()

        self.species = self.species_table = None

        self.__dict__.update(kwargs)

    def add_as_species(self, species, lookup_table):
        """
        Add carrying capacity using another species. Inter-species relationships are specified using a lookup table.
        :param Species species: Species instance
        :param iterable lookup_table: A table to define the relationship between the input species density and
            carrying capacity. The lookup table x-values define the density of the input species, and the
            y-values define a coefficient that is applied to the carrying capacity data when added to another
            species in the domain.
        :return: None
        """
        if all([not isinstance(species, obj) for obj in [Species, Sex, AgeGroup]]):
            raise PopdynError('Input carrying capacity is not a species')

        self.species = species
        self.species_table = dynamic.collect_lookup(lookup_table)

    def random(self, type, **kwargs):
        """
        Apply a type of random variability to this carrying capacity
        :param type: Variance type. Use one of the random distribution generator methods in popdyn.dynamic.
        :param kwargs: :param tuple args: parameters for the variance algorithm (variable)
        :return: None
        """
        if type not in dynamic.RANDOM_METHODS.keys():
            raise PopdynError('Unsupported random distribution generator "{}". Choose from:\n{}'.format(
                type, '\n'.join(dynamic.RANDOM_METHODS.keys()))
            )

        self.random = type
        self.random_args = kwargs.get('args', None)


class Mortality(object):
    """
    Mortality drivers are templates that may be applied to any species in a model domain. Multiple mortality
        instances may be added to a given species.

    Mortality instances contain relevant attributes, and also serve to use species as a mortality driver.
    Mortality is added to a Domain along with a species on which to impart the mortality, and a dataset that
        defines the rates.
    """

    def __init__(self, name, **kwargs):
        """
        Mortality driver instance
        :param str name: Name of the mortality
        :param kwargs:
            :param is_rate: This mortality is defined as a rate, as opposed to an absolute number (default is True)
        """
        # Limit mortality names to 25 chars
        if len(name) > 25:
            raise PopdynError('Mortality names must not exceed 25 characters. '
                              'Use something simple, like "Starvation".')

        self.name = name
        self.name_key = name.strip().translate(None, punctuation + ' ').lower()

        self.species = self.species_table = None

        self.__dict__.update(kwargs)

    def add_as_species(self, species, lookup_table):
        """
        Add mortality using another species. Inter-species relationships are specified using a lookup table.
        :param Species species: Species instance
        :param iterable lookup_table: A table to define the relationship between the input species density and mortality.
            The lookup table x-values define the density of the input species, and the y-values define a mortality
            rate that overrides the mortality data when added to another species in the domain.
        :return: None
        """
        if all([not isinstance(species, obj) for obj in [Species, Sex, AgeGroup]]):
            raise PopdynError('Input mortality is not a species')

        self.species = species
        self.species_table = dynamic.collect_lookup(lookup_table)

    def random(self, type, **kwargs):
        """
        Apply a type of random variability to this mortality
        :param type: Variance type. Use one of the random distribution generator methods in popdyn.dynamic.
        :param kwargs: :param tuple args: parameters for the variance algorithm (variable)
        :return: None
        """
        if type not in dynamic.RANDOM_METHODS.keys():
            raise PopdynError('Unsupported random distribution generator "{}". Choose from:\n{}'.format(
                type, '\n'.join(dynamic.RANDOM_METHODS.keys()))
            )

        self.random = type
        self.random_args = kwargs.get('args', None)
