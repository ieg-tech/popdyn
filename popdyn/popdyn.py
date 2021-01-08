"""
Population dynamics simulation domain setup and model interface

Devin Cairns, 2018
"""

from __future__ import print_function
import os
import pickle
import numpy as np
from osgeo import gdal, osr
from logger import Timer
import dispersal
import dynamic
from util import *

# import h5fake as h5py
import h5py


class PopdynError(Exception):
    pass


class Domain(object):
    """
    Population dynamics simulation domain, which is used to define a study area for a popdyn simulation

    Species, Parameters, and datasets are added to the model domain, which creates a file to _save and store all information
    """
    def _save(func):
        """Decorator for saving attributes to the popdyn file"""
        def inner(*args, **kwargs):
            instance = args[0]
            execution = func(*args, **kwargs)

            instance._dump()

            return execution
        return inner

    def __init__(self, popdyn_path, domain_raster=None, **kwargs):
        """
        To create a Model Domain, one of either an existing ``.popdyn`` file, a raster, or construct the domain
        using the keyword arguments

        :param str path: Path to a new (new Domain) or existing (existing Domain) ``.popdyn`` file on the disk
        :param str domain_raster: Path to an existing raster on the disk that defines the domain extent and resolution

        :Keyword Arguments:
            **shape** (*tuple*) --
                Number of rows and columns of the study area [``(rows, cols)``]
            **csx** (*float*) --
                Grid spacing (cell size) in the x-direction
            **csy** (*float*) --
                Grid spacing (cell size) in the y-direction
            **top** (*float*) --
                The northernmost coordinate of the study area
            **left** (*float*) --
                The westernmost coordinate of the study area
            **projection** (*int*) --
                Spatial Reference of the study area, which must be an EPSG code
        """
        # If the path exists, parameterize the model based on the previous run
        if os.path.isfile(popdyn_path) or os.path.isdir(popdyn_path):
            self.path = popdyn_path  # Files are accessed through the file attribute
            self.file = h5py.File(self.path, libver='latest', mode='r+')
            self._load()
        else:

            self._build_new(popdyn_path, domain_raster, **kwargs)

        if not hasattr(self, 'timer'):
            self.timer = Timer()
    # File management and builtins
    # ==================================================================================================

    @staticmethod
    def _create_file(path):
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

    def _add_directory(self, key):
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

    def _dump(self):
        """Dump all of the domain instance to the file"""
        if h5py.__name__ == 'h5py':
            self.file.attrs.update(
                {key: np.void(pickle.dumps(val)) for key, val in self.__dict__.items() if key not in ['file', 'path']}
            )
        else:
            self.file.attrs.update(
                {key: val for key, val in self.__dict__.items() if key not in ['file', 'path']}
            )

    def _load(self):
        """Get the model parameters from an existing popdyn file"""
        if h5py.__name__ == 'h5py':
            self.__dict__.update({key: pickle.loads(val.tostring()) for key, val in self.file.attrs.items()})
        else:
            self.__dict__.update({key: val for key, val in self.file.attrs.items()})

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
        try:
            return self.file[item]
        except KeyError:
            raise KeyError('Item {} not found'.format(item))

    def __setitem__(self, key, data):
        try:
            del self.file[key]
        except KeyError:
            pass
        if h5py.__name__ == 'h5py':
            kwargs = {}
        else:
            # h5fake
            kwargs = {
                'sr': self.projection,
                'gt': (self.left, self.csx, 0, self.top, 0, self.csy * -1),
                'nd': [0]
            }
        _ = self.file.create_dataset(key, data=np.broadcast_to(data, self.shape),
                                     compression='lzf', chunks=self.chunks, **kwargs)

    def __repr__(self):
        return 'Popdyn domain of shape {} with\n{}'.format(self.shape, '\n'.join(self.species_names))

    # Data parsing and checking methods
    # ==================================================================================================
    @_save
    def _build_new(self, popdyn_path, domain_raster, **kwargs):
        """Used in the constructor to build the domain using input arguments"""
        # Make sure the name suits the specifications
        self.path = self._create_file(popdyn_path)
        self.file = h5py.File(self.path, libver='latest', mode='w')

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
            spatial_params = self._data_from_raster(domain_raster)
            self.csx, self.csy, self.shape, self.top, self.left, self.projection, self.mask = spatial_params

            # Domain cannot be empty
            if self.mask.sum() == 0:
                raise PopdynError('The domain cannot be empty. Check the input raster "{}"'.format(domain_raster))

        # Create some other attributes
        self.timer = Timer()  # Used for profiling function times
        self.species = rec_dd()
        self.population = rec_dd()
        self.mortality = rec_dd()
        self.fecundity = rec_dd()
        self.dispersal = rec_dd()
        self.masks = rec_dd()
        self.carrying_capacity = rec_dd()
        # Inheritance may be turned off completely
        if not hasattr(self, 'avoid_inheritance'):
            self.avoid_inheritance = False
        # Chunk size is specified for data storage and dask scheduling
        if not hasattr(self, 'chunks'):
            if h5py.__name__ == 'h5py':
                chunks = [5000, 5000]
                if chunks[0] > self.shape[0]:
                    chunks[0] = self.shape[0]
                if chunks[1] > self.shape[1]:
                    chunks[1] = self.shape[1]
            else:
                chunks = [256, 256]
            self.chunks = tuple(chunks)

    def _raster_as_array(self, raster, as_mask=False):
        """
        Collect the underlying array from a raster data source
        :param raster: input raster file
        :param as_mask: return only a mask of where data values exist in the raster
        :return: numpy.ndarray
        """
        # Expects a gdal raster object
        file_source = raster

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

    def _data_from_raster(self, raster):
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
            return csx, csy, shape, top, left, projection, self._raster_as_array(file_source, True)

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
            file_source = self._transform_ds(file_source)

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

        return self._raster_as_array(file_source)

    def _transform_ds(self, gdal_raster):
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
    def _get_time_input(time):
        """Make sure an input time is an integer"""
        try:
            _time = int(time)
        except:
            raise PopdynError("Unable to parse the input time of type {}".format(type(time).__name__))

        if not np.isclose(_time, float(time)):
            raise PopdynError('Input time may only be an integer (whole number)')

        return _time

    def _get_data_type(self, data):
        """Parse a data argument to retrieve a domain-shaped matrix"""
        # First, try for a file
        if isinstance(data, basestring) and os.path.isfile(data):
            return self._data_from_raster(data)

        # Try and construct a matrix from data
        else:
            try:
                return np.asarray(data).astype('float32')
            except:
                raise PopdynError('Unable to parse the input data {} into the domain'.format(data))

    # Domain-species interaction methods
    # ==================================================================================================

    # These are the methods to call for primary purposes
    # ---------------------------------------------------
    @_save
    def add_population(self, species, data, time, **kwargs):
        """
        Add population data for a given species at a specific time slice in the domain.

        :param Species species: A Species instance
        :param data: Population data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)
        :param time: The time slice to insert the population data into

        :Keyword Arguments:
            **distribute_by_habitat** (*bool*) --
                Divide the sum of the input population linearly among domain elements using the covariate
                carrying capacity (Default: False)
            **discrete_age** (*int*) --
                Apply the input population to a single discrete age in the age group (Default: None)
            **distribute** (*bool*) --
                Divide the sum of the input data evenly among all domain nodes (grid cells) (default: True)
            **overwrite** (*str*) --
                Overwrite method for replacing existing data. Use one of ``['replace', 'add']`` (Default 'replace')
        """
        self._introduce_species(species)  # Introduce the species
        time = self._get_time_input(time)  # Gather time
        data = self._get_data_type(data)  # Parse input data

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

            self._add_data(key, data,
                           distribute=kwargs.get('distribute', True),
                           distribute_by_co=distribute_by_habitat,
                           overwrite=kwargs.get('overwrite', 'replace')
                           )

    @_save
    def add_carrying_capacity(self, species, carrying_capacity, time, data=None, **kwargs):
        """
        Carrying capacity is added to the domain with species and CarryingCapacity objects in tandem.

        Multiple carrying capacity datasets may added to a single species,
            as they are stacked in the domain for each species (or sex/age group).

        :param Species species: Species instance
        :param CarryingCapacity carrying_capacity: Carrying capacity instance.
        :param time: The time slice to insert the carrying capacity
        :param data: Carrying Capacity data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)

        :Keyword Arguments:
            **is_density** (*bool*) --
                The input data are a density, and not an absolute population (Default: False)
            **distribute** (*bool*) --
                Divide the sum of the input data evenly among all domain nodes (grid cells) (default: True)
            **overwrite** (*str*) --
                Overwrite method for replacing existing data. Use one of ``['replace', 'add']`` (Default 'replace')
        """
        if not isinstance(carrying_capacity, CarryingCapacity):
            raise PopdynError('Expected a CarryingCapacity instance, not "{}"'.format(
                type(carrying_capacity).__name__)
            )

        self._introduce_species(species)  # Introduce the species
        time = self._get_time_input(time)  # Gather time
        if data is not None:
            data = self._get_data_type(data)  # Parse input data
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
            self._add_data(key, data,
                           distribute=kwargs.get('distribute', True),
                           overwrite=kwargs.get('overwrite', 'replace')
                           )

        # The instance and the key are added
        self.carrying_capacity[species.name_key][species.sex][species.group_key][time][k_key] = (
            carrying_capacity, key
        )

    @_save
    def add_mortality(self, species, mortality, time, data=None, **kwargs):
        """
        Mortality is added to the domain with species and Mortality objects in tandem.

        Multiple mortality datasets may added to a single species,
        as they are stacked in the domain for each species (or sex/age group). If the mortality being added is
        derived from another species, or is time-based, data does not need to be provided.

        :param Species species: Species instance
        :param Morality mortality: Mortality instance.
        :param data: Mortality data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)
        :param time: The time slice to insert mortality

        :Keyword Arguments:
            **distribute** (*bool*) --
                Divide the sum of the input data evenly among all domain nodes (grid cells) (default: True)
            **overwrite** (*str*) --
                Overwrite method for replacing existing data. Use one of ``['replace', 'add']`` (Default 'replace')
        """
        if not isinstance(mortality, Mortality):
            raise PopdynError('Expected a mortality instance, not "{}"'.format(type(mortality).__name__))

        self._introduce_species(species)  # Introduce the species
        time = self._get_time_input(time)  # Gather time
        if data is not None:
            data = self._get_data_type(data)  # Parse input data
        else:
            # Mortality must be tied to a species or be time-based if there are no input data
            if not hasattr(mortality, 'species') and not hasattr(mortality, 'time_based_rates'):
                raise PopdynError(
                    'Mortality data must be provided with {}, as it is not attached to a species and does not have '
                    'time-based rate dependence'.format(mortality.name)
                )

        m_key = mortality.name_key

        # If mortality is defined only by a lookup table or a time-based rate, the key is None
        if data is None:
            key = None
        else:
            key = '{}/{}/{}/{}/{}'.format(species.name_key, species.sex, species.group_key, time, m_key)

            # Add the data to the file
            self._add_data(key, data,
                           distribute=kwargs.get('distribute', True),
                           overwrite=kwargs.get('overwrite', 'replace'),
                           distribute_by_co=kwargs.get('distribute_by_co', None)
                           )

        self.mortality[species.name_key][species.sex][species.group_key][time][m_key] = (mortality, key)

    @_save
    def add_fecundity(self, species, fecundity, time, data=None, **kwargs):
        """
        Fecundity is added to the domain with species and Fecundity objects in tandem.

        Multiple fecundity datasets may added to a single species, as they are stacked in the domain for each
        species (or sex/age group).

        :param Species species: Species instance
        :param Fecundity fecundity: Fecundity instance.
        :param data: Fecundity data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)
        :param time: The time slice to insert fecundity

        :Keyword Arguments:
            **distribute** (*bool*) --
                Divide the sum of the input data evenly among all domain nodes (grid cells) (default: True)
            **overwrite** (*str*) --
                Overwrite method for replacing existing data. Use one of ``['replace', 'add']`` (Default 'replace')
        """
        # Species must be a Sex or AgeGroup to have fecundity
        if not any([isinstance(species, o) for o in [Sex, AgeGroup]]):
            raise PopdynError('Species with fecundity must have a sex')

        if not isinstance(fecundity, Fecundity):
            raise PopdynError('Expected a Fecundity instance, not "{}"'.format(type(fecundity).__name__))

        self._introduce_species(species)  # Introduce the species
        time = self._get_time_input(time)  # Gather time
        if data is not None:
            data = self._get_data_type(data)  # Parse input data
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
            self._add_data(key, data,
                           distribute=kwargs.get('distribute', True),
                           overwrite=kwargs.get('overwrite', 'replace'),
                           distribute_by_co=kwargs.get('distribute_by_co', None)
                           )

        self.fecundity[species.name_key][species.sex][species.group_key][time][f_key] = (fecundity, key)

    @_save
    def add_time_based_dispersal(self, species, dispersal_type, time, args=()):
        """
        Add a disperal method that is time variant to a specific species

        .. attention:: Time-variant dispersal methods are applied prior to other dispersal methods during a time step

        :param Species species: Species instance
        :param time: The time slice to insert fecundity
        :param str dispersal_type: One of the ``dispersal.METHODS`` keywords
        :param tuple args: Arguments to accompany the dispersal method
        """
        if dispersal_type not in dispersal.METHODS.keys():
            raise PopdynError('The dispersal method {} has not been implemented'.format(dispersal_type))

        self._introduce_species(species)  # Introduce the species
        time = self._get_time_input(time)  # Gather time

        self.dispersal[species.name_key][species.sex][species.group_key][time][dispersal_type] = args

    def add_disease(self, input_file, **kwargs):
        direct, env = read_cwd_input(input_file)
        if kwargs.get('direct_transmission'):
            self.direct_transmission = direct
        if kwargs.get('environmental_transmission'):
            # TODO: E data should be added in the domain, as it has a specific discretization
            self.environmental_transmission = {'C': env, 'E': kwargs.get('E_data')}

    @_save
    def add_mask(self, species, time, data=None, function='masked dispersal', **kwargs):
        """
        A mask is a general-use dataset associated with a species, and must be associated with a function

        :param Species species: Species instance
        :param time: The time slice to insert fecundity
        :param data: Mask data. This may be a scalar, raster, or vector/matrix (broadcastable to the domain)
        :param str function: The purpose of the mask (where it is used in solvers)

        :Keyword Arguments:
            **distribute** (*bool*) --
                Divide the sum of the input data evenly among all domain nodes (grid cells) (default: True)
            **overwrite** (*str*) --
                Overwrite method for replacing existing data. Use one of ``['replace', 'add']`` (Default 'replace')
        """
        self._introduce_species(species)  # Introduce the species
        time = self._get_time_input(time)  # Gather time
        data = self._get_data_type(data)  # Parse input data

        key = '{}/{}/{}/{}/masks/{}'.format(species.name_key, species.sex, species.group_key, time, function)

        # Add the data to the file
        self._add_data(key, data,
                       distribute=kwargs.get('distribute', True),
                       overwrite=kwargs.get('overwrite', 'replace'),
                       distribute_by_co=kwargs.get('distribute_by_co', None)
                       )

        self.masks[species.name_key][species.sex][species.group_key][time][function] = key

    @_save
    def remove_dataset(self, ds_type, species_key, sex, group, time, name):
        """
        Remove a dataset and corresponding key (and instance) from the Domain

        :param ds_type: use one of ``('population', 'mortality', 'fecundity', 'carrying_capacity')``
        :param species_key: Species.name_key
        :param sex: sex ('male' or 'female')
        :param group: AgeGroup.group_key
        :param int time: Time slice
        :param name: Name of the dataset. For populations, use an age integer, for other types, use the instance name
        """
        ds_type = ds_type.lower()
        if ds_type not in ['population', 'mortality', 'carrying_capacity']:
            raise PopdynError('No such dataset type "{}"'.format(ds_type))

        time = self._get_time_input(time)

        del getattr(self, ds_type)[species_key][sex][group][time][name]

        # Remove loose ends
        if len(getattr(self, ds_type)[species_key][sex][group][time]) == 0:
            del getattr(self, ds_type)[species_key][sex][group][time]
            if len(getattr(self, ds_type)[species_key][sex][group]) == 0:
                del getattr(self, ds_type)[species_key][sex][group]
                if len(getattr(self, ds_type)[species_key][sex]) == 0:
                    del getattr(self, ds_type)[species_key][sex]
                    if len(getattr(self, ds_type)[species_key]) == 0:
                        del getattr(self, ds_type)[species_key]

        key = '{}/{}/{}/{}/{}'.format(species_key, sex, group, time, name)
        try:
            del self.file[key]
        except KeyError:
            pass

    @_save
    def remove_species(self, species):
        """
        Remove the species from the domain, including all respective data

        :param species: Species.name_key
        """
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
        del self.dispersal[species.name_key]

        print("Successfully removed {} from the domain".format(species.name))

    # These are factories
    # ---------------------------------------------------
    # Recursive functions for traversing species in the domain
    @property
    def species_names(self):
        """
        Property of all species names in the Domain

        :return list: Sorted strings of all species names
        """

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
        """
        Collect all group names within a specific species

        :param str species_key: Species.name_key
        :return list: Strings of group names
        """

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

    def _group_keys(self, species_key):
        """
        Collect all group dataset keys within a specific species

        :param str species_key: Species.name_key
        :return list: Strings of group dataset keys
        """
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
        """
        Collect all species instances added to the model domain

        :return list: Species instances
        """
        def is_instance(d):
            for key, val in d.items():
                if isinstance(val, dict):
                    is_instance(val)
                elif any([isinstance(val, obj) for obj in [Species, Sex, AgeGroup]]):
                    instances.append(val)

        instances = []
        is_instance(self.species)
        return instances

    @property
    def mortality_instances(self):
        """
        Collect all mortality instances in the model domain

        :return list: Mortality instances
        """
        def is_instance(d):
            for key, val in d.items():
                if isinstance(val, dict):
                    is_instance(val)
                elif isinstance(val[0], Parameter):
                    instances.append(val[0])

        instances = []
        is_instance(self.mortality)
        return np.unique(instances).tolist()

    @property
    def fecundity_instances(self):
        """
        Collect all fecundity instances in the model domain

        :return list: Fecundity instances
        """

        def is_instance(d):
            for key, val in d.items():
                if isinstance(val, dict):
                    is_instance(val)
                elif isinstance(val[0], Parameter):
                    instances.append(val[0])

        instances = []
        is_instance(self.fecundity)
        return np.unique(instances).tolist()

    @property
    def carrying_capacity_instances(self):
        """
        Collect all k instances in the model domain

        :return list: CarryingCapacity instances
        """

        def is_instance(d):
            for key, val in d.items():
                if isinstance(val, dict):
                    is_instance(val)
                elif isinstance(val[0], Parameter):
                    instances.append(val[0])

        instances = []
        is_instance(self.carrying_capacity)
        return np.unique(instances).tolist()

    def discrete_ages(self, species_key, sex):
        """
        Collect all discrete ages for a species in the model domain

        :param str species_key: Species.name_key
        :param sex: sex ('male' or 'female')
        :return list: Ordered sequence of ages
        """
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

    @property
    def discrete_times(self):
        """Collect all times with data in the Domain"""
        attrs = ['population',
                 'mortality',
                 'fecundity',
                 'masks',
                 'carrying_capacity']
        times = []
        for attr in attrs:
            species = getattr(self, attr).keys()
            for sp in species:
                sexes = getattr(self, attr)[sp].keys()
                for sex in sexes:
                    groups = getattr(self, attr)[sp][sex].keys()
                    for gp in groups:
                        times += getattr(self, attr)[sp][sex][gp].keys()

        return np.unique(times)

    @staticmethod
    def _deconstruct_key(key):
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

    def all_species_with_mortality(self, mortality):
        """
        Collect all of the species tied to a specific mortality instance
        :param Mortality mortality:
        :return: List of species name keys
        """
        species = []

        def next_key(d, sp):
            if isinstance(d, dict):
                for key, val in d.items():
                    next_key(val, sp)
            elif isinstance(d[0], Mortality) and mortality.name_key == d[0].name_key:
                species.append(sp)

        for current_species in self.mortality.keys():
            next_key(self.mortality[current_species], current_species)

        return species

    def youngest_group(self, species, sex):
        """
        Collect the group with the youngest age in the domain

        :param str species_key: Species.name_key
        :param sex: sex ('male' or 'female')
        :return str: group name
        """
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
        """
        Collect the group of a discrete age

        :param str species: Species.name_key
        :param sex: sex ('male' or 'female')
        :param int age: Age within a group age range
        :return str: group name
        """
        groups = self.species[species][sex].keys()
        for group in groups:
            val = self.species[species][sex][group]  # Could be a dict or an instance
            if isinstance(val, Species):
                if any([age == val for val in val.age_range]):
                    return group

    def get_species_instance(self, species_key, sex, gp):
        """
        Collect a species instance

        :param str species_key: Species.name_key
        :param sex: sex ('male' or 'female')
        :param gp: group name
        :return Species: instance
        """
        if gp in self.species[species_key][sex].keys():
            instance = self.species[species_key][sex][gp]
        else:
            raise PopdynError('There appears to be no species associated with the query:\n'
                              'Species Key: {}\n'
                              'Sex: {}\n'
                              'Group: {}'.format(species_key, sex, gp))
        return instance

    def age_from_group(self, species_key, sex, gp):
        """
        Collect all ages from a group

        :param str species_key: Species.name_key
        :param sex: sex ('male' or 'female')
        :param gp: group name
        :return list: All ages
        """
        instance = self.get_species_instance(species_key, sex, gp)
        return instance.age_range

    def _instance_from_key(self, key):
        """Return a species instance associated with the input key"""
        species_key, sex, group = self._deconstruct_key(key)[:3]
        return self.get_species_instance(species_key, sex, group)

    def _introduce_species(self, species):
        """INTERNAL> Add a species to the model domain"""
        # Notify console if this species is entirely new
        if species.name_key not in self.species.keys():
            print("Species {} introduced into the domain".format(species.name))

        self.species[species.name_key][species.sex][species.group_key] = species

    def _add_data(self, key, data, **kwargs):
        """
        Prepare and write data to the popdyn file
        Overwrite behaviour supports some simple math

        This is not meant to be called directly.

        :param key: key of the dataset in the popdyn file
        :param np.ndarray data: Population data to _save. This must be filtered through Domain._get_data_type in advance

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
            species_key, sex, group, time, ds_type = self._deconstruct_key(key)[:5]
            # The covariate is collected using a function call that is constructed using the data type name
            co = getattr(self, 'get_{}'.format(distribute_by_co))(species_key, time, sex, group, inherit=True)
            # Only collect HDF5 data
            co = [self[_co[1]] for _co in co if _co[1] is not None]

            # Use dask to aggregate data because this could be memory-heavy
            if len(co) > 0:
                co = dsum([da.from_array(ds, ds.chunks) for ds in co]).compute()

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
            ds[:] = np.add(ds[:], data)
        elif overwrite == 'subtract':
            ds[:] = np.subtract(ds[:], data)

    def get_mortality(self, species_key, time, sex, group_key, snap_to_time=True, inherit=True):
        """
        Collect the mortality instance - dataset pairs

        :param str species_key: Species.name_key
        :param str sex: sex ('male', or 'female')
        :param str group_key: AgeGroup.group_key
        :param int time: Time slice
        :param snap_to_time: If the dataset queried does not exist, it can be snapped backwards in time to the nearest available dataset
        :param avoid_inheritance: Do not inherit a dataset from the sex or species
        :return: list of instance - :class:`h5py.dataset` key pairs
        """
        time = self._get_time_input(time)

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
        return [ds for ds in datasets.values()]

    def get_fecundity(self, species_key, time, sex, group_key, snap_to_time=True, inherit=True):
        """
        Collect the fecundity instance - HDF5 Dataset pairs

        :param str species_key: Species.name_key
        :param str sex: sex ('male', or 'female')
        :param str group_key: AgeGroup.group_key
        :param int time: Time slice
        :param snap_to_time: If the dataset queried does not exist, it can be snapped backwards in time to the nearest available dataset
        :param avoid_inheritance: Do not inherit a dataset from the sex or species
        :return: list of instance - :class:`h5py.dataset` key pairs
        """
        time = self._get_time_input(time)

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
        return [ds for ds in datasets.values()]

    def get_dispersal(self, species_key, time, sex, group_key, snap_to_time=True, inherit=True):
        """
        Collect the name and args associated with a time-variant dispersal

        :param str species_key: Species.name_key
        :param str sex: sex ('male', or 'female')
        :param str group_key: AgeGroup.group_key
        :param int time: Time slice
        :param snap_to_time: If the data queried does not exist, it can be snapped backwards in time to the
            nearest available data
        :param inherit: Do not inherit a data from the sex or species
        :return: tuple (dispersal method name, args)
        """
        time = self._get_time_input(time)

        # Collect the dataset keys using inheritance
        def collect(species_key, time, sex, group_key):
            if snap_to_time:
                times = np.unique(self.dispersal[species_key][sex][group_key].keys())

                delta = time - times
                backwards = delta >= 0
                # If no times are available, time is not updated
                if backwards.sum() > 0:
                    times = times[backwards]
                    delta = delta[backwards]
                    i = np.argmin(delta)
                    time = times[i]
            keys = [
                (key, self.dispersal[species_key][sex][group_key][time][key])
                for key in self.dispersal[species_key][sex][group_key][time].keys()
            ]
            if len(keys) > 0:
                return keys
            else:
                return None

        if inherit:
            for _sex in [sex, None]:
                for gp in [group_key, None]:
                    keys = collect(species_key, time, _sex, gp)
                    if keys is not None:
                        return keys
            return None
        else:
            return collect(species_key, time, sex, group_key)

    def get_mask(self, species_key, time, sex, group_key, function='masked dispersal', snap_to_time=True, inherit=True):
        """
        Collect the key associated with the mask query

        :param str species_key: Species.name_key
        :param str sex: sex ('male', or 'female')
        :param str group_key: AgeGroup.group_key
        :param int time: Time slice
        :param str function: The purposed of the mask being retrieved
        :param snap_to_time: If the dataset queried does not exist, it can be snapped backwards in time to the
            nearest available dataset
        :param inherit: Do not inherit a dataset from the sex or species
        :return: Dataset key
        """
        time = self._get_time_input(time)

        # Collect the dataset keys using inheritance
        def collect(species_key, time, sex, group_key):
            if snap_to_time:
                times = self.masks[species_key][sex][group_key].keys()
                times = [t for t in times if function in self.masks[species_key][sex][group_key][t].keys()]

                times = np.unique(times)
                delta = time - times
                backwards = delta >= 0
                # If no times are available, time is not updated
                if backwards.sum() > 0:
                    times = times[backwards]
                    delta = delta[backwards]
                    i = np.argmin(delta)
                    time = times[i]
            key = self.masks[species_key][sex][group_key][time][function]
            if len(key) > 0:
                return key
            else:
                return None

        if inherit:
            for _sex in [sex, None]:
                for gp in [group_key, None]:
                    key = collect(species_key, time, _sex, gp)
                    if key is not None:
                        return key
            return None
        else:
            return collect(species_key, time, sex, group_key)

    def get_carrying_capacity(self, species_key, time, sex=None, group_key=None, snap_to_time=True, inherit=False):
        """
        Collect the carrying capacity instance - key pairs

        :param str species_key: Species.name_key
        :param str sex: sex ('male', or 'female')
        :param str group_key: AgeGroup.group_key
        :param int time: Time slice
        :param snap_to_time: If the dataset queried does not exist, it can be snapped backwards in time to the nearest available dataset
        :param bool inherit: Collect data from parent species if they do not exist for the input. Used primarily for distributing by a covariate,
            and should not be used during simulations (the values may change during pre-solving inheritance)
        :return: list of instance - :class:`h5py.Dataset` keys
        """
        time = self._get_time_input(time)

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

            # If there are data in the domain, return the HDF5 dataset key, else None
            return [name_dict[key] for key in datasets]

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

        :param str species_key: Species.name_key
        :param str sex: sex ('male', or 'female')
        :param str group_key: AgeGroup.group_key
        :param int time: Time slice
        :param bool snap_to_time: If the dataset queried does not exist, it can be snapped backwards in time to the nearest available dataset
        :return list: All Carrying Capacity instances - :class:`h5py.Dataset` key pairs
        """
        time = self._get_time_input(time)

        # sex, group and age may be None, which enables return of all keys
        sexes = [sex]
        if sex is None:
            sexes += ['male', 'female']
        groups = [group_key]
        if group_key is None:
            for _sex in sexes:
                groups += [key for key in self.carrying_capacity[species_key][_sex].keys() if key is not None]

        # Duplicate group names may exist for males and females
        groups = np.unique(groups)

        cc = []
        for _sex in sexes:
            for group in groups:
                cc += self.get_carrying_capacity(species_key, time, _sex, group, snap_to_time)

        return cc

    def get_population(self, species_key, time, sex=None, group_key=None, age=None, inherit=False):
        """
        Collect the population key of a species/sex/group at a given time if it exists

        :param str species_key: Species.name_key
        :param str sex: sex ('male', or 'female')
        :param str group_key: AgeGroup.group_key
        :param int time: Time slice
        :param int age: Absolute age
        :param bool inherit: Collect data from parent species if they do not exist for the input. Used primarily for distributing by a covariate,
            and should not be used during simulations (the values may change during pre-solving inheritance)
        :return: Dataset key or None if non-existent
        """
        time = self._get_time_input(time)

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

        :param str species_key: Species.name_key
        :param str sex: sex ('male', or 'female')
        :param str group_key: AgeGroup.group_key
        :param int time: Time slice
        :param int age: Absolute age
        :return: list of :class:'h5py.Dataset` keys
        """
        time = self._get_time_input(time)

        # sex, group and age may be None, which enables return of all keys
        sexes = [sex]
        if sex is None:
            sexes += ['male', 'female']
        groups = [group_key]
        if group_key is None:
            for _sex in sexes:
                groups += [key for key in self.population[species_key][_sex].keys() if key is not None]

        # Duplicate group names may exist for males and females
        groups = np.unique(groups)

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
    """Top-level species customization"""
    restricted_names = ['global', 'total']

    def __init__(self, name, **kwargs):
        """
        :param str name: Name of the species (example "Moose"), which is limited to 25 characters. The Species name is
            transformed into a unique key that is utilized to determine inheritance between Species, Sex, and Age
            Groups in the model domain.

        :Keyword Arguments:
            **contributes_to_density** (*bool*) --
                This species contributes to density calculations (default: True)
            **density_threshold** (*float*) --
                The density at which density-dependent mortality has an effect  (default: 1.)
            **density_scale** (*float*) --
                This is the rate of density dependent mortality, applied linearly between the ``density_threshold`` and
                the maxmum density  (default: 1.)
            **minimum_viable_population** (*float*) --
                The minimum population that may exist within a statistically-derived region. All of the population
                will not be allowed to exist if below this value  (default: 0.)
            **minimum_viable_area** (*float*) --
                The minimum statistically-derived area to apply the ``minimum_viable_population`` to (default: 0.)
            **live_past_max** (*bool*) --
                This is a switch that determines whether a species may live past the oldest age group maximum age.
                (default: False)
            **global_population** (*bool*) --
                The global total population in a :class:`Domain` is calculated using all :class:`Species` with the
                `global_population` `kwarg` set to `True`. (default: True)
            **use_global_density** (*bool*) --
                If True, density-based calculations (i.e. density-dependent mortality, fecundity) will be completed
                using the global density, and not the species-level density. (default: False)
        """
        # Limit species names to 25 chars
        if len(name) > 25:
            raise PopdynError('Species names must not exceed 25 characters. '
                              'Use something simple, like "Moose".')

        if name.lower() in self.restricted_names:
            raise PopdynError('Species names may not be any of {}'.format(self.restricted_names))

        self.name = name
        self.name_key = name_key(name)

        # Does this species get included in species-wide density calculations?
        self.contributes_to_density = kwargs.get('contributes_to_density', True)
        # Point at which density-dependent mortality is effective
        self.density_threshold = np.float32(kwargs.get('density_threshold', 1.))
        # Rate of density-dependent mortality
        self.density_scale = np.float32(kwargs.get('density_scale', 1.))
        # Minimum viable population - a minimum population that may be allowed within a given area
        self.minimum_viable_population = kwargs.get('minimum_viable_population', 0.)
        # Minimum viable population area
        self.minimum_viable_area = kwargs.get('minimum_viable_area', 0.)

        # Does this species live past the maximum specified age (if it is an age group)?
        self.live_past_max = kwargs.get('live_past_max', False)

        # No dispersal by default
        self.dispersal = []

        # sex and age group are none
        self.sex = self.group_key = None

        self.global_population = kwargs.get('global_population', True)
        self.use_global_density = kwargs.get('use_global_density', False)

    def add_dispersal(self, dispersal_type, args=()):
        """
        Sequentially add disperal methods to the species

        .. attention:: Current solvers apply dispersal in the order that they are applied through these calls

        :param str dispersal_type: One of the ``dispersal.METHODS`` keywords
        :param tuple args: Arguments to accompany the dispersal method
        """
        if dispersal_type not in dispersal.METHODS.keys():
            raise PopdynError('The dispersal method {} has not been implemented'.format(dispersal_type))

        self.dispersal.append((dispersal_type, args))

    def add_disease(self, input_file, **kwargs):
        direct, env = read_cwd_input(input_file)
        if kwargs.get('direct_transmission'):
            self.direct_transmission = direct
        if kwargs.get('environmental_transmission'):
            # TODO: E data should be added in the domain, as it has a specific discretization
            self.environmental_transmission = {'C': env, 'E': kwargs.get('E_data')}

    @property
    def age_range(self):
        try:
            return range(self.min_age, self.max_age + 1)
        except AttributeError:
            return [None]

    def __repr__(self):
        return 'Species {}-{}-{}'.format(self.name_key, self.sex, self.group_key)


class Sex(Species):
    """Child of Species, containing sex-based attributes"""

    def __init__(self, name, sex, **kwargs):
        """
        :param str name: Name of the species (example "Moose"), which is limited to 25 characters. The Species name is
            transformed into a unique key that is utilized to determine inheritance between Species, Sex, and Age
            Groups in the model domain.
        :param str sex: Name of sex, must be one of 'male' or 'female'. Case is not important.

        :Keyword Arguments:
            **contributes_to_density** (*bool*) --
                This species contributes to density calculations (default: True)
            **density_threshold** (*float*) --
                The density at which density-dependent mortality has an effect  (default: 1.)
            **density_scale** (*float*) --
                This is the rate of density dependent mortality, applied linearly between the ``density_threshold`` and
                the maxmum density  (default: 1.)
            **minimum_viable_population** (*float*) --
                The minimum population that may exist within a statistically-derived region. All of the population
                will not be allowed to exist if below this value  (default: 0.)
            **minimum_viable_area** (*float*) --
                The minimum statistically-derived area to apply the ``minimum_viable_population`` to (default: 0.)
            **live_past_max** (*bool*) --
                This is a swtich that determines whether a species may live past the oldest age group maximum age.
                (default: False)
        """

        super(Sex, self).__init__(name, **kwargs)

        if sex not in ['male', 'female']:
            raise PopdynError('Sex must be one of "male" or "female"')

        self.sex = sex.lower()


class AgeGroup(Sex):
    """Child of Sex, containing age information"""

    def __init__(self, species_name, group_name, sex, min_age, max_age, **kwargs):
        """

        :param str species_name: Name of the species (example "Moose"), which is limited to 25 characters. The Species name is
            transformed into a unique key that is utilized to determine inheritance between Species, Sex, and Age
            Groups in the model domain.
        :param str group_name: Name of the group associated with the age range (example "newborn"), which is limited to
            25 characters. The group name is transformed into a unique key.
        :param str sex: Name of sex, must be one of 'male' or 'female'. Case is not important.
        :param min_age: The minimum age included in this group.
        :param max_age: The maximum age included in this group.

        :Keyword Arguments:
            **contributes_to_density** (*bool*) --
                This species contributes to density calculations (default: True)
            **density_threshold** (*float*) --
                The density at which density-dependent mortality has an effect  (default: 1.)
            **density_scale** (*float*) --
                This is the rate of density dependent mortality, applied linearly between the ``density_threshold`` and
                the maxmum density  (default: 1.)
            **minimum_viable_population** (*float*) --
                The minimum population that may exist within a statistically-derived region. All of the population
                will not be allowed to exist if below this value  (default: 0.)
            **minimum_viable_area** (*float*) --
                The minimum statistically-derived area to apply the ``minimum_viable_population`` to (default: 0.)
            **live_past_max** (*bool*) --
                This is a swtich that determines whether a species may live past the oldest age group maximum age.
                (default: False)
        """
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
        """
        :param name: Name of the parameter. This must be no more than 25 characters.
        """
        # Limit names to 25 chars
        if len(name) > 25:
            raise PopdynError('Parameter names must not exceed 25 characters. '
                              'Use something simple.')

        self.name = name
        self.name_key = name_key(name)

        self.species = self.species_table = self.population_type = None

        self.__dict__.update(kwargs)

    def add_as_species(self, species, lookup_table, population_type='density'):
        """
        Inter-species relationships are specified using this method, whereby a parameter is calculated using an
        attribute of another species. The density of the input species is used to calculate either a value, or a
        coefficient that modifies the value using a table.

        :param Species species: Species instance
        :param iterable lookup_table: A table to define the relationship between the input species population type and
            the parameter. The lookup table x-values define the population type of the input species, and the y-values
            define this parameter, in the form: ``[(x1, y1), (x2, y2)...(xn, yn)]``
        :param str population_type: Variable related to population of the input species to use to derive the parameter.
            choose from
            -``'total population'``, which is the total population of the affecting species
            -``'density'``, which is the density (n/k) of the affecting species
            -``global population`` is the global total population (n_global)
            -``'global ratio'``, which is the ratio of this species of the global total population (n/n_global)
            See the :class:`Species` attribute ``global_population``.
        """
        if all([not isinstance(species, obj) for obj in [Species, Sex, AgeGroup]]):
            raise PopdynError('Input parameter is not a species')

        self.species = species
        self.species_table = dynamic.collect_lookup(lookup_table)

        population_types = ['total population', 'density', 'global population', 'global ratio']
        if population_type.lower() not in population_types:
            raise PopdynError('Input population type "{}" not supported'.format(population_type))

        self.population_type = population_type.lower()

    def random(self, type, **kwargs):
        """
        Apply a type of random variability to this parameter. Random numbers are generated using one of the available
        distributions from the
        `numpy.random module <https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html>`_.

        :param type: Variance type. Use one of the random distribution generator methods in ``numpy.random``

        :Keyword Arguments:
            Keyword arguments should match all required arguments to the selected ``numpy.random`` method
        """
        if type not in dir(np.random):
            raise PopdynError('Unsupported random distribution generator "{}"'.format(type))

        # *
        # Note, the presence of random inputs is queried through a `hasattr` call
        # *

        self.apply_using_mean = kwargs.pop('apply_using_mean', True)

        self.random_method = type
        self.random_args = kwargs


class CarryingCapacity(Parameter):
    """
    Carrying Capacity is used to define the maximum population that may be supported for one or more Species in a Domain.

    Carrying Capacity effectively represents habitat quality, and is added to a Species or Domain alongside data. Multiple
    types of Carrying Capacity may be added to a given species, as they are added together in the domain.
    When another species is added (using the :func:`add_as_species` method) to a Carrying Capacity instance, the affecting
    species will scale the Carrying Capacity using a coefficient specified in the lookup table.

    .. Note:: multiple Carrying Capacities may be tied to an individual species, and they are stacked in the domain
    """

    def __init__(self, name, **kwargs):
        """
        :param name: Name of the Carrying Capacity (25 chars). Use something short and intuitive, such as "Marsh".
        """
        super(CarryingCapacity, self).__init__(name, **kwargs)


class Fecundity(Parameter):
    """
    Fecundity is used to define a set of reproductive attributes that may be tied to one or more Species.

    Fecundity is added to a Species and a Domain together, optionally with data. When another Species is added (using
    the :func:`add_as_species` method), the affecting species will impose a fecundity value using the included lookup
    table.
    """

    def __init__(self, name, **kwargs):
        """
        :param name: Name of the Fecundity (25 chars). Use something short and intuitive, such as "Normal Adult".
        :Keyword Arguments:
            **birth_ratio** (*bool*) --
                The birth ratio is used to allocate offspring as a ratio of males to females. The birth ratio may also
                be random (use ``'random'``), which uses a uniform random number generator to select a ratio between 0
                and 1 for each element in the domain. (Default: 0.5)
        """

        super(Fecundity, self).__init__(name, **kwargs)

        self.birth_ratio = kwargs.get('birth_ratio', 0.5)  # May be 'random' to use a random uniform query
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
    Mortality drivers are used to define rates of population decline for one or more Species.

    Mortality is added to a Species and a Domain together, optionally with data. When another Species is added (using
    the :func:`add_as_species` method), the affecting species will impose a mortality rate using the included lookup
    table. Multiple mortality instances may be added to a given Species. The mortality name cannot be any of "mortality",
    "density dependent", or "old age" because of implicit mortality types used in solvers.
    """
    forbidden_names = ['mortality', 'density dependent', 'old age']

    def __init__(self, name, **kwargs):
        """
        Mortality driver instance

        :param str name: Name of the mortality (25 chars). Use something short and intuitive, such as "Poaching".
        :param is_rate: This mortality is defined as a rate, as opposed to an absolute number (default is True)
        """
        super(Mortality, self).__init__(name, **kwargs)

        if name.strip().lower() in self.forbidden_names:
            raise PopdynError('The mortality name may not be any of: {}'.format(', '.join(self.forbidden_names)))

        self.recipient_species = None

    def add_recipient_species(self, species):
        """
        The population that succumbs to this mortality may be added to another species. This is primarily used to
        track infection and disease, whereby infected individuals are treated as a different species.

        :param Species species: The recipient Species for populations that succumb to this mortality
        """
        if all([not isinstance(species, obj) for obj in [Species, Sex, AgeGroup]]):
            raise PopdynError('Input parameter is not a species')

        self.recipient_species = species

    def add_time_based_mortality(self, rates, random_std=None):
        """
        Time-based mortality defines a set of mortality rates over a given duration

        Using the Discrete Explicit solver, at any time when a species population is greater than 0, a counter
        accumulates the number of consecutive years that the species has a non-zero population. The input mortality
        rates are used consecutively over the counted duration when time-based mortality is used.

        :param rates: An iterable of mortality rates that applies from time 0 to time len(rates)
        :param float random_std: A standard deviation to generate random variation on rates using a normal distribution
        """
        self.time_based_rates = rates
        self.time_based_std = random_std
