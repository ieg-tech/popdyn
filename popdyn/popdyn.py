"""
Population dynamics simulation domain setup and model interface

Devin Cairns, 2018
"""

from __future__ import print_function
import os
import pickle
from string import punctuation
import numpy as np
from collections import defaultdict
import dask.array as da
import h5py
from osgeo import gdal, osr
import dispersal
import dynamic
from logger import time_this


class PopdynError(Exception):
    pass


def rec_dd():
    """Recursively update defaultdicts to avoid key errors"""
    return defaultdict(rec_dd)


class Domain(object):
    """
    Population dynamics simulation domain. A domain instance is used to:
        -Manage a .popdyn file to store 2-D data in HDF5 format, and act as a quasi-database for the model domain
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
    able to inherit mortality and carrying capacity based on a species - sex - age group hierarchy.
    """
    # Decorators
    #-------------------------------------------------------------------------
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
                print("Popdyn domain %s created" % path)
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
        self.file.attrs.update(
            {key: np.void(pickle.dumps(val)) for key, val in self.__dict__.items() if key not in ['file', 'path']}
        )

    def load(self):
        """Get the model parameters from an existing popdyn file"""
        self.__dict__.update({key: pickle.loads(val.tostring()) for key, val in self.file.attrs.items()})

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
                    # Projection must be a SRID or WKT
                    sr = osr.SpatialReference()
                    try:
                        if isinstance(self.projection, basestring):
                            sr.ImportFromWkt(self.projection)
                        else:
                            # Assume EPSG
                            sr.ImportFromEPSG(self.projection)
                    except TypeError:
                        raise PopdynError('Only a SRID or WKT are accepted input projection arguments')
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
        self.fecundity = rec_dd()
        self.carrying_capacity = rec_dd()
        # Inheritance may be turned off completely
        if not hasattr(self, 'avoid_inheritance'):
            self.avoid_inheritance = False
        # Chunk size is specified for data storage and dask scheduling
        if not hasattr(self, 'chunks'):
            chunks = [5000, 5000]  # ~100MB chunks. This should be sufficient for most modern computers.
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
            return a.astype(np.float32)

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
    @save
    def domain_compute(self, datasets):
        """
        Takes dask arrays and dataset pointers and computes/writes to the file

        NOTE: dask.store optimization will not allow multiple writes of the same output from the graph

        :param dict datasets: dataset pointers (keys) and respective dask arrays
        :return: None
        """
        # Force all data to float32...
        datasets = {key: val.astype(np.float32) for key, val in datasets.items()}

        # Compute and dump the datasets (adapted from da.to_hdf5 to avoid opening file again)
        dsets = [self.file.require_dataset(dp, shape=x.shape, dtype=x.dtype,
                                           chunks=tuple([c[0] for c in x.chunks]), **{'compression': 'lzf'})
                 for dp, x in datasets.items()]
        da.store(list(datasets.values()), dsets)

        # Add population keys to domain
        for key in datasets.keys():
            species, sex, group, time, age = self.deconstruct_key(key)[:5]
            if age not in ['params', 'flux']:  # Avoid offspring, carrying capacity, and mortality
                self.population[species][sex][group][time][age] = key

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
        Add population data for a given species at a specific time slice in the domain.
        :param Species species: A Species instance
        :param data: Population data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)
        :param time: The time slice to insert the population data into

        :param kwargs:
            :distribute: Divide the input population data evenly among the domain elements (default is True)
            :distribute_by_habitat: Divide the input population linearly among domain elements using
                the covariate carrying capacity (Default is False)
            :overwrite: one of ['replace', 'add', 'subtract'] to send to self.add_data() (Default is replace)
            :discrete_age: Apply the population to a single discrete age in the age group
        :return: None
        """
        self.introduce_species(species)  # Introduce the species
        time = self.get_time_input(time)  # Gather time
        data = self.get_data_type(data)  # Parse input data

        discrete_age = kwargs.get('discrete_age', None)

        ages = species.age_range

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

            self.add_data(key, data,
                distribute=kwargs.get('distribute', True),
                distribute_by_co=distribute_by_habitat,
                overwrite=kwargs.get('overwrite', 'replace')
            )

    @save
    def add_carrying_capacity(self, species, carrying_capacity, time, data=None, **kwargs):
        """
        Carrying capacity is added to the domain with species and CarryingCapacity objects in tandem.
        Multiple carrying capacity datasets may added to a single species,
            as they are stacked in the domain for each species (or sex/age group).

        :param Species species: Species instance
        :param CarryingCapacity carrying_capacity: Carrying capacity instance.
        :param time: The time slice to insert the carrying capacity
        :param data: Carrying Capacity data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)
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
        if data is not None:
            data = self.get_data_type(data)  # Parse input data
            if kwargs.get('is_density', False):
                # Convert density to population per cell
                data *= self.csx * self.csy
        else:
            # Mortality must be tied to a species if there are no input data
            if not hasattr(carrying_capacity, 'species'):
                raise PopdynError('Carrying capacity data must be provided with {}, '
                                  'as it is not attached to a species'.format(
                    carrying_capacity.name
                ))

            # The species may not be itself (this would create an infinite loop)
            if carrying_capacity.species.name_key == species.name_key:
                raise PopdynError('A species may not dynamically change carrying capacity for itself.')

        k_key = carrying_capacity.name_key

        if data is None:
            key = None
        else:
            key = '{}/{}/{}/{}/{}'.format(species.name_key, species.sex, species.group_key, time, k_key)

            # Add the data to the file
            self.add_data(key, data,
                distribute=kwargs.get('distribute', True),
                overwrite=kwargs.get('overwrite', 'replace')
            )

        # The instance and the key are added
        self.carrying_capacity[species.name_key][species.sex][species.group_key][time][k_key] = (
            carrying_capacity, key
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
            self.add_data(key, data,
                distribute=kwargs.get('distribute', True),
                overwrite=kwargs.get('overwrite', 'replace'),
                distribute_by_co=kwargs.get('distribute_by_co', None)
            )

        self.mortality[species.name_key][species.sex][species.group_key][time][m_key] = (mortality, key)

    @save
    def add_fecundity(self, species, fecundity, time, data=None, **kwargs):
        """
        Fecundity is added to the domain with species and Fecundity objects in tandem.
        Multiple fecundity datasets may added to a single species,
            as they are stacked in the domain for each species (or sex/age group).

        :param Species species: Species instance
        :param Fecundity fecundity: Fecundity instance.
        :param data: Fecundity data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)
        :param time: The time slice to insert fecundity
        :kwargs:
            :distribute: Divide the input data evenly over the domain
            :overwrite: Overwrite any existing data (Default 'replace')
        """
        # Species must be a Sex or AgeGroup to have fecundity
        if not any([isinstance(species, o) for o in [Sex, AgeGroup]]):
            raise PopdynError('Species with fecundity must have a sex')

        if not isinstance(fecundity, Fecundity):
            raise PopdynError('Expected a Fecundity instance, not "{}"'.format(type(fecundity).__name__))

        self.introduce_species(species)  # Introduce the species
        time = self.get_time_input(time)  # Gather time
        if data is not None:
            data = self.get_data_type(data)  # Parse input data
        else:
            # Fecundity must be tied to a species if there are no input data
            if not hasattr(fecundity, 'species'):
                raise PopdynError('Fecundity data must be provided if there are no attached species')

        f_key = fecundity.name_key

        # If fecundity is defined only by a lookup table, the key is None
        if data is None:
            key = None
        else:
            key = '{}/{}/{}/{}/{}'.format(species.name_key, species.sex, species.group_key, time, f_key)

            # Add the data to the file
            self.add_data(key, data,
                distribute=kwargs.get('distribute', True),
                overwrite=kwargs.get('overwrite', 'replace'),
                distribute_by_co=kwargs.get('distribute_by_co', None)
            )

        self.fecundity[species.name_key][species.sex][species.group_key][time][f_key] = (fecundity, key)

    @save
    def remove_dataset(self, ds_type, species_key, sex, group, time, name):
        """Remove a dataset and corresponding key [and instance] from the domain"""
        ds_type = ds_type.lower()
        if ds_type not in ['population', 'mortality', 'carrying_capacity']:
            raise PopdynError('No such dataset type "{}"'.format(ds_type))

        time = self.get_time_input(time)

        del getattr(self, ds_type)[species_key][sex][group][time][name]
        key = '{}/{}/{}/{}/{}'.format(species_key, sex, group, time, name)
        try:
            del self.file[key]
        except KeyError:
            pass

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

    def group_names(self, species_key):
        """Collect all groups under a specific species"""

        def is_instance(d):
            for key, val in d.items():
                if isinstance(val, dict):
                    is_instance(val)
                elif any([isinstance(val, obj) for obj in [Species, Sex, AgeGroup]]):
                    try:
                        groups.append(val.group_name)
                    except AttributeError:
                        pass

        groups = []
        is_instance(self.species[species_key])
        return np.unique(groups)

    def group_keys(self, species_key):
        """Collect all groups under a specific species"""
        def is_instance(d):
            for key, val in d.items():
                if isinstance(val, dict):
                    is_instance(val)
                elif any([isinstance(val, obj) for obj in [Species, Sex, AgeGroup]]):
                    groups.append(val.group_key)

        groups = []
        is_instance(self.species[species_key])
        return np.unique(groups)

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

    def discrete_ages(self, species_key, sex):
        """Return all discrete ages for a species in the model domain"""
        def is_age(d):
            for key, val in d.items():
                if isinstance(val, dict):
                    is_age(val)
                elif any([isinstance(val, obj) for obj in [Species, Sex, AgeGroup]]):
                    ages.append(val.age_range)

        ages = []
        is_age(self.species[species_key][sex])
        age_ret = []
        for age in ages:
            age_ret += age
        return np.sort(age_ret)

    @staticmethod
    def deconstruct_key(key):
        """Deconstruct a dataset key into hashes that will work with the Domain instance"""
        values = []

        for val in key.split('/'):
            if val == 'None':
                val = None
            try:
                val = int(val)  # Times or ages may be integers
            except:
                pass
            values.append(val)

        return values

    def youngest_group(self, species, sex):
        """Collect the youngest group instance"""
        age = np.finfo(np.float32).max
        min_gp = None
        for group in self.species[species][sex].values():
            if isinstance(group, Species):
                try:
                    _age = group.min_age
                except AttributeError:
                    continue
                if _age < age:
                    age = _age
                    min_gp = group

        return min_gp

    def group_from_age(self, species, sex, age):
        """Collect the group of a discrete age"""
        groups = self.species[species][sex].keys()
        for group in groups:
            val = self.species[species][sex][group]  # Could be a dict or an instance
            if isinstance(val, Species):
                if any([age == val for val in val.age_range]):
                    return group

    def get_species_instance(self, species_key, sex, gp):
        if gp in self.species[species_key][sex].keys():
            instance = self.species[species_key][sex][gp]
        else:
            raise PopdynError('There appears to be no species associated with the query:\n'
                              'Species Key: {}\n'
                              'Sex: {}\n'
                              'Group: {}'.format(species_key, sex, gp))
        return instance

    def age_from_group(self, species_key, sex, gp):
        """Collect all ages from a group"""
        instance = self.get_species_instance(species_key, sex, gp)
        return instance.age_range

    def instance_from_key(self, key):
        """Return a species instance associated with the input key"""
        species_key, sex, group = self.deconstruct_key(key)[:3]
        return self.get_species_instance(species_key, sex, group)

    def introduce_species(self, species):
        """INTERNAL> Add a species to the model domain"""
        # Notify console if this species is entirely new
        if species.name_key not in self.species.keys():
            print("Species {} introduced into the domain".format(species.name))

        self.species[species.name_key][species.sex][species.group_key] = species

    @time_this
    def add_data(self, key, data, **kwargs):
        """
        Prepare and write data to the popdyn file
        Overwrite behaviour supports some simple math

        This is not meant to be called directly.

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
            co = getattr(self, 'get_{}'.format(distribute_by_co))(species_key, time, sex, group, inherit=True)
            # Only collect HDF5 data
            co = [_co[1] for _co in co if _co[1] is not None]

            # Use dask to aggregate data because this could be memory-heavy
            if len(co) > 0:
                co = da.dstack([da.from_array(ds, ds.chunks) for ds in co]).sum(axis=-1).compute()

            # Compute the sum. If no data are available or the dataset is empty, the sum will be 0
            co_sum = np.sum(co)
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

    def get_mortality(self, species_key, time, sex, group_key, snap_to_time=True, inherit=True):
        """
        Collect the mortality instance - key pairs
        :param str species_key:
        :param str sex:
        :param str group_key:
        :param int time:
        :param snap_to_time: If the dataset queried does not exist, it can be snapped backwards in time to the nearest available dataset
        :param avoid_inheritance: Do not inherit a dataset from the sex or species
        :return: list of instance - HDF5 dataset pairs
        """
        time = self.get_time_input(time)

        if snap_to_time:
            times = self.mortality[species_key][sex][group_key].keys()
            times = [t for t in times if len(self.mortality[species_key][sex][group_key][t]) > 0]

            times = np.unique(times)
            delta = time - times
            backwards = delta >= 0
            # If no times are available, time is not updated
            if backwards.sum() > 0:
                times = times[backwards]
                delta = delta[backwards]
                i = np.argmin(delta)
                time = times[np.squeeze(i)]

        # Collect the dataset keys using inheritance
        if not inherit or self.avoid_inheritance:
            groups = [group_key]
            sexes = [sex]
        else:
            groups = [None, group_key]
            sexes = [None, sex]

        datasets = {}
        for group in groups:
            for _sex in sexes:
                datasets.update(self.mortality[species_key][_sex][group][time])

        # If there are data in the domain, return the HDF5 dataset, else None
        return [(ds[0], None) if ds[1] is None else (ds[0], self[ds[1]]) for ds in datasets.values()]

    def get_fecundity(self, species_key, time, sex, group_key, snap_to_time=True, inherit=True):
        """
        Collect the fecundity instance - key pairs
        :param str species_key:
        :param int time:
        :param str sex:
        :param str group_key:
        :param snap_to_time: If the dataset queried does not exist, it can be snapped backwards in time to the nearest available dataset
        :param avoid_inheritance: Do not inherit a dataset from the sex or species
        :return: list of instance - HDF5 dataset pairs
        """
        time = self.get_time_input(time)

        if snap_to_time:
            times = self.fecundity[species_key][sex][group_key].keys()
            times = [t for t in times if len(self.fecundity[species_key][sex][group_key][t]) > 0]

            times = np.unique(times)
            delta = time - times
            backwards = delta >= 0
            # If no times are available, time is not updated
            if backwards.sum() > 0:
                times = times[backwards]
                delta = delta[backwards]
                i = np.argmin(delta)
                time = times[i]

        # Collect the dataset keys using inheritance
        if not inherit or self.avoid_inheritance:
            groups = [group_key]
            sexes = [sex]
        else:
            groups = [None, group_key]
            sexes = [None, sex]

        datasets = {}
        for group in groups:
            for _sex in sexes:
                datasets.update(self.fecundity[species_key][_sex][group][time])

        # If there are data in the domain, return the HDF5 dataset, else None
        return [(ds[0], None) if ds[1] is None else (ds[0], self[ds[1]]) for ds in datasets.values()]

    def get_carrying_capacity(self, species_key, time, sex=None, group_key=None, snap_to_time=True, inherit=False):
        """
        Collect the carrying capacity instance - key pairs
        :param str species_key:
        :param str sex:
        :param str group_key:
        :param int time:
        :param snap_to_time: If the dataset queried does not exist, it can be snapped backwards in time to the nearest available dataset
        :param bool inherit: Collect data from parent species if they do not exist for the input. Used primarily for distributing by a covariate,
            and should not be used during simulations (the values may change during pre-solving inheritance)
        :return: list of instance - HDF5 dataset pairs
        """
        time = self.get_time_input(time)

        def collect_ds(species_key, time, sex, group_key, snap_to_time):
            """Factory so looping can occur if inherit is True"""
            if snap_to_time:
                # Modify time to the location closest backwards in time with data
                times = self.carrying_capacity[species_key][sex][group_key].keys()
                times = [t for t in times if len(self.carrying_capacity[species_key][sex][group_key][t]) > 0]

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
            datasets = self.carrying_capacity[species_key][sex][group_key][time].keys()
            name_dict = self.carrying_capacity[species_key][sex][group_key][time]

            # If there are data in the domain, return the HDF5 dataset, else None
            return [(name_dict[key][0], None)
                    if name_dict[key][1] is None
                    else (name_dict[key][0], self[name_dict[key][1]])
                    for key in datasets]

        if inherit:
            for _sex in [sex, None]:
                for gp in [group_key, None]:
                    ds = collect_ds(species_key, time, _sex, gp, snap_to_time)
                    if len(ds) > 0:
                        return ds
            return []
        else:
            return collect_ds(species_key, time, sex, group_key, snap_to_time)

    def all_carrying_capacity(self, species_key, time, sex=None, group_key=None, snap_to_time=True):
        """
        Collect all carrying capacity for all children in the given query
        :param species_key:
        :param time:
        :param sex:
        :param group_key:
        :return:
        """
        time = self.get_time_input(time)

        # sex, group and age may be None, which enables return of all keys
        sexes = [sex]
        if sex is None:
            sexes += ['male', 'female']
        groups = [group_key]
        if group_key is None:
            for _sex in sexes:
                groups += [key for key in self.carrying_capacity[species_key][_sex].keys() if key is not None]

        cc = []
        for _sex in sexes:
            for group in groups:
                cc += self.get_carrying_capacity(species_key, time, _sex, group, snap_to_time)

        return cc

    def get_population(self, species_key, time, sex=None, group_key=None, age=None, inherit=False):
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

        def collect(species_key, time, sex, group_key, age):
            key = self.population[species_key][sex][group_key][time][age]
            if len(key) > 0:
                return key
            else:
                return None

        if inherit:
            for _sex in [sex, None]:
                for gp in [group_key, None]:
                    for _age in [age, None]:
                        key = collect(species_key, time, _sex, gp, _age)
                        if key is not None:
                            return key
            return None
        else:
            return collect(species_key, time, sex, group_key, age)

    def all_population(self, species_key, time, sex=None, group_key=None):
        """
        Collect the population keys of a species at a given time. Datasets from all children keys are returned.
        :param species_key:
        :param time:
        :param sex:
        :param group_key:
        :param age:
        :return: dict of key - HDF5 dataset pairs
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

        return {key: self[key] for key in np.unique(keys)}


# Species classes
#   Species: No age or sex information.
#      |
#       Sex: Species with sex specified. Inherits all carrying capacity and
#            mortality from the same species when added to the domain.
#        |
#         AgeGroup: Species with a sex and a range of ages. Inherits all carrying capacity, mortality,
#                   and fecundity from the same species/sex when added to the domain.
# ========================================================================================================

def name_key(name):
    """Map a given name to a stripped alphanumeric hash"""
    # Remove white space and make lower-case
    name = name.strip().replace(' ', '').lower()

    try:
        # String
        return name.translate(None, punctuation)
    except:
        # Unicode
        return name.translate(dict.fromkeys(punctuation))


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
        self.name_key = name_key(name)

        # Does this species get included in species-wide density calculations?
        self.contributes_to_density = kwargs.get('contributes_to_density', True)
        # Point at which density-dependent mortality is effective
        self.density_threshold = np.float32(kwargs.get('density_threshold', 1.))
        # Rate of density-dependent mortality
        self.density_scale = np.float32(kwargs.get('density_scale', 1.))

        # Does this species live past the maximum specified age (if it is an age group)?
        self.live_past_max = kwargs.get('live_past_max', False)

        # No dispersal by default
        self.dispersal = []

        # sex and age group are none
        self.sex = self.group_key = None

    def add_dispersal(self, dispersal_type, args=()):
        """
        Sequentially add disperal methods to the species.
        Dispersal will be applied in the order that it is added to the species.
        :param dispersal_type: One of the dispersal.METHODS keywords
        :param tuple args: Arguments (*args) to accompany the dispersal method
        :return:
        """
        if dispersal_type not in dispersal.METHODS.keys():
            raise PopdynError('The dispersal method {} has not been implemented'.format(dispersal_type))

        self.dispersal.append((dispersal_type, args))

    @property
    def age_range(self):
        try:
            return range(self.min_age, self.max_age + 1)
        except AttributeError:
            return [None]


class Sex(Species):

    def __init__(self, name, sex, **kwargs):
        """
        Species --> Sex constructor
        :param str name: Name of the species. Example "Moose"
        :param sex: Name of sex, must be one of 'male', 'female'
        """

        super(Sex, self).__init__(name, **kwargs)

        if sex not in ['male', 'female']:
            raise PopdynError('Sex must be one of "male" or "female"')

        self.sex = sex.lower()


class AgeGroup(Sex):

    def __init__(self, species_name, group_name, sex, min_age, max_age, **kwargs):
        if len(group_name) > 25:
            raise PopdynError('Group names must not exceed 25 characters. '
                              'Use something simple, like "Yearling".')

        super(AgeGroup, self).__init__(species_name, sex, **kwargs)

        self.group_name = group_name
        self.group_key = name_key(group_name)

        self.min_age = min_age
        self.max_age = max_age


# Carrying capacity, mortality drivers, and fecundity.
# These classes parameterize specific information, and enable
# inter-species relationships and randomness in the domain
# =======================================================================================================


class Parameter(object):

    def __init__(self, name, **kwargs):
        # Limit names to 25 chars
        if len(name) > 25:
            raise PopdynError('Parameter names must not exceed 25 characters. '
                              'Use something simple.')

        self.name = name
        self.name_key = name_key(name)

        self.species = self.species_table = None

        self.__dict__.update(kwargs)

    def add_as_species(self, species, lookup_table):
        """
        Add a parameter using another species. Inter-species relationships are specified using a lookup table.
        :param Species species: Species instance
        :param iterable lookup_table: A table to define the relationship between the input species density and
            the parameter. The lookup table x-values define the density of the input species, and the y-values
            define this parameter.
        :return: None
        """
        if all([not isinstance(species, obj) for obj in [Species, Sex, AgeGroup]]):
            raise PopdynError('Input mortality is not a species')

        self.species = species
        self.species_table = dynamic.collect_lookup(lookup_table)

    def random(self, type, **kwargs):
        """
        Apply a type of random variability to this parameter
        :param type: Variance type. Use one of the random distribution generator methods in popdyn.dynamic.
        :param kwargs: :param tuple args: parameters for the variance algorithm (variable)
        :return: None
        """
        if type not in dynamic.RANDOM_METHODS.keys():
            raise PopdynError('Unsupported random distribution generator "{}". Choose from:\n{}'.format(
                type, '\n'.join(dynamic.RANDOM_METHODS.keys()))
            )

        self.random_method = type
        self.random_args = kwargs


class CarryingCapacity(Parameter):
    """
    Carrying capacity is a template that may be applied to any species in a model domain. Multiple carrying capacity
        instances may be added to a given species.

    Carrying capacity instances contain relevant attributes, and also serve to use species to modify carrying capacity.
    Carrying capacity is added to a Domain along with a species that utilizes the carrying capacity,
        and a dataset that defines the carrying capacity.
    """

    def __init__(self, name, **kwargs):
        super(CarryingCapacity, self).__init__(name, **kwargs)


class Fecundity(Parameter):
    """
    Fecundity can be derived from a dataset or a species and a lookup table. It is added to a species and a domain
    using the Domain.add_fecundity method, along with respective data if necessary.
    """

    def __init__(self, name, **kwargs):

        super(Fecundity, self).__init__(name, **kwargs)

        self.birth_ratio = np.float32(kwargs.get('birth_ratio', 0.5))  # May be 'random' to use a random uniform query
        self.density_fecundity_threshold = np.float32(kwargs.get('density_fecundity_threshold', 1.))
        self.fecundity_reduction_rate = np.float32(kwargs.get('fecundity_reduction_rate', 1.))
        self.density_fecundity_max = np.float32(kwargs.get('density_fecundity_max', 1.))

        # This dictates whether offspring spawn from species tied to this fecundity
        self.multiplies = kwargs.get('multiplies', True)

        # The default fecundity lookup is inf (no males): 0., less: 1.
        self.fecundity_lookup = dynamic.collect_lookup(kwargs.get('fecundity_lookup', None)) \
            if kwargs.get('fecundity_lookup', False) else dynamic.collect_lookup([(0, 1.), (np.inf, 0.)])


class Mortality(Parameter):
    """
    Mortality drivers are templates that may be applied to any species in a model domain. Multiple mortality
        instances may be added to a given species.
    The mortality name cannot be any of "mortality", "density dependent", or "old age" because of solver defaults.

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
        super(Mortality, self).__init__(name, **kwargs)

        forbidden_names = ['mortality', 'density dependent', 'old age']
        if name.strip().lower() in forbidden_names:
            raise PopdynError('The mortality name may not be any of: {}'.format(', '.join(forbidden_names)))
