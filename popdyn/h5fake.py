"""
Slow and painful transition away from HDF5 and h5py using os directories and rasters

Devin Cairns
January, 2019
"""
import os
import pickle
import shutil
import numpy as np
from osgeo import gdal, gdalconst


class H5FError(Exception):
    pass


class Group(object):
    """
    Manage a directory - key structure
    """
    def __init__(self, path, **kwargs):
        if not os.path.isdir(path):
            raise KeyError('Unable to open object (component not found)')
        self.name = path
        self.path = path
        self.attrs = Attrs(path)
        self.attrs.update(kwargs)

    @staticmethod
    def decompose_key(key):
        if key[0] == '/':
            key = key[1:]
        return key.split('/')

    @staticmethod
    def clean_key(key):
        return key[1:] if key[0] == '/' else key

    @staticmethod
    def dir_members(d):
        return {os.path.join(d, f) for f in os.listdir(d) if os.path.isdir(f) or f.split('.')[-1] == 'tif'}

    def __getitem__(self, s):
        try:
            basestring
        except NameError:
            basestring = (str, bytes)
        if not isinstance(s, basestring):
            raise H5FError('Only string getters are supported')

        f = os.path.join(self.path, self.clean_key(s))
        if os.path.isfile(f + '.tif'):
            return Dataset(f)
        elif os.path.isdir(f):
            return Group(f)
        else:
            raise KeyError('Unable to find object {}'.format(s))

    def keys(self):
        return [f.replace('.tif', '') if f[-4:] == '.tif' else f
                for f in os.listdir(self.path)
                if os.path.isdir(os.path.join(self.path, f)) or f[-4:] == '.tif']

    def values(self):
        values = [os.path.join(self.path, f) for f in os.listdir(self.path) if f != 'attrs.dmp']
        for val in values:
            if val.split('.')[-1] == 'tif':
                return Dataset(val)
            else:
                return Group(val)

    def build_tree(self, key):
        if os.path.isdir(os.path.join(self.path, self.clean_key(key))):
            return
        tree = self.decompose_key(key)
        path = self.path
        for t in tree[:-1]:
            path = os.path.join(path, t)
            if not os.path.isdir(path):
                os.mkdir(path)

        os.mkdir(os.path.join(path, tree[-1]))

    def create_group(self, key, **kwargs):
        key = self.clean_key(key)
        if os.path.isdir(os.path.join(self.path, key)):
            raise ValueError('Unable to create group (name already exists)')

        self.build_tree(key)

        return Group(os.path.join(self.path, key), **kwargs)

    def create_dataset(self, key, data=None, shape=None, chunks=None, dtype=None, **kwargs):
        key = self.clean_key(key)
        if os.path.isfile(os.path.join(self.path, key + '.tif')):
            raise RuntimeError('Dataset already exists')

        tree = self.decompose_key(key)
        if len(tree) > 1:
            self.build_tree('/'.join(tree[:-1]))

        # Create the dataset
        return Dataset(os.path.join(self.path, key), data=data, shape=shape, chunks=chunks, dtype=dtype, **kwargs)

    def require_dataset(self, key, shape, dtype, chunks, **kwargs):
        if os.path.isfile(os.path.join(self.path, key) + '.tif'):
            return Dataset(os.path.join(self.path, self.clean_key(key)))
        else:
            return self.create_dataset(self.clean_key(key), shape=shape, dtype=dtype, chunks=chunks, **kwargs)

    def __delitem__(self, key):
        if len(key) > 0:
            path = os.path.join(self.path, self.clean_key(key))
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path + '.tif'):
                os.remove(path + '.tif')
            else:
                raise KeyError('{}'.format(key))


class File(Group):
    """
    The file instance is actually an os directory structure
    """
    def __init__(self, path, **kwargs):
        """
        :param path: directory with raster files
        :param tuple geotransform: GDAL Raster geotransform specification
        :param str projection: WKT projection
        :param libver: kwarg for backwards compatibility
        """
        if not os.path.isdir(path):
            os.mkdir(path)

        super(File, self).__init__(path, **kwargs)

    # Methods for backwards compatibility
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def close(self):
        """
        To avoid attibute error
        :return:
        """
        pass


class Attrs(object):
    def __init__(self, path, type='gp'):
        self.path = os.path.join(path, '{}attrs.dmp'.format(type))

        if not os.path.isfile(self.path):
            with open(self.path, 'wb') as f:
                pickle.dump({}, f)

    def __getitem__(self, key):
        with open(self.path, 'rb') as f:
            return pickle.load(f)[key]

    def __setitem__(self, key, val):
        with open(self.path, 'rb') as f:
            d = pickle.load(f)
        d[key] = val
        with open(self.path, 'wb') as f:
            pickle.dump(d, f)

    def keys(self):
        with open(self.path, 'rb') as f:
            d = pickle.load(f)
        return d.keys()

    def values(self):
        with open(self.path, 'rb') as f:
            d = pickle.load(f)
        return d.values()

    def update(self, other_dict):
        with open(self.path, 'rb') as f:
            d = pickle.load(f)
        d.update(other_dict)
        with open(self.path, 'wb') as f:
            pickle.dump(d, f)

    def get(self, key, default=None):
        with open(self.path, 'rb') as f:
            d = pickle.load(f)
            try:
                return d[key]
            except KeyError:
                return default


class Dataset(object):
    """
    Shadow of the h5py Dataset class
    """
    def __init__(self, path, data=None, shape=None, chunks=None, dtype=None, **kwargs):
        self.path = path
        self.attrs = Attrs(self.gp_path, os.path.basename(path))
        self.set_env(**kwargs)

        # Create the dataset if it does not exist
        if not os.path.isfile(self.raster_path):
            self.write_data(chunks, shape, data, dtype)
        else:
            ds = gdal.Open(self.raster_path)
            self.shape = (ds.RasterYSize, ds.RasterXSize)
            band = ds.GetRasterBand(1)
            self.chunks = band.GetBlockSize()
            self.dtype = dtype_to_np(gdal.GetDataTypeName(band.DataType))
            band = None
            ds = None

    @property
    def gp_path(self):
        return os.path.dirname(self.path)

    def set_env(self, **kwargs):
        """
        Collect spatial parameters from the local attrs, overwritten by any provided kwargs
        :return:
        """
        if os.path.isfile(self.raster_path):
            ds = gdal.Open(self.raster_path)
            self.sr = ds.GetProjectionRef()
            self.gt = ds.GetGeoTransform()
            self.nd = []
            for i in range(1, ds.RasterCount):
                band = ds.GetRasterBand(i)
                nd = band.GetNoDataValue()
                if nd is None:
                    self.nd.append(np.nan)
                else:
                    self.nd.append(nd)
            ds = None

        else:
            self.sr = kwargs.get('sr', Group(self.gp_path).attrs.get('sr'))
            if self.sr is None:
                raise H5FError('A spatial reference is required for a dataset')
            self.gt = kwargs.get('gt', Group(self.gp_path).attrs.get('gt'))
            if self.gt is None:
                raise H5FError('A geotransform is required for a dataset')
            self.nd = kwargs.get('nd', Group(self.gp_path).attrs.get('nd'))
            if self.nd is None:
                raise H5FError('A no data list is required for a dataset')


    def write_data(self, chunks=None, shape=None, data=None, dtype=None):
        if data is None and shape is None:
            raise H5FError('One of either data or shape must be specified when creating a new dataset')

        if data is not None:
            data = np.atleast_3d(data)
            dtype = dtype if dtype is not None else data.dtype.name
            data = data.astype(dtype)
            shape = data.shape
        else:
            if not hasattr(shape, '__iter__'):
                shape = (shape,)
            else:
                shape = tuple(shape)
            shape = shape + (1,) * max(0, (3 - len(shape)))
            dtype = dtype if dtype is not None else 'float64'
            data = np.empty(shape=shape, dtype=dtype)

        self.dtype = dtype

        if chunks is None:
            chunks = (256,) * (len(shape) - 1) + (int(shape[2]),)
        self.chunks = chunks

        if len(shape) > 3:
            raise H5FError('A maximum of 3 dimensions are supported')
        self.shape = shape

        driver = gdal.GetDriverByName('GTiff')
        options = ['TILED=YES', 'COPY_SRC_OVERVIEWS=YES', 'COMPRESS=LZW',
                   'BLOCKXSIZE={}'.format(self.chunks[1]),
                   'BLOCKYSIZE={}'.format(self.chunks[0])]

        ds = driver.Create(
            self.raster_path,
            int(shape[1]),
            int(shape[0]),
            int(shape[2]),
            gdal.GetDataTypeByName(dtype_to_gdal(dtype)),
            options
        )
        if ds is None:
            raise TypeError('GDAL error while creating new raster')

        ds.SetGeoTransform(self.gt)
        ds.SetProjection(self.sr)

        for i in range(1, shape[2] + 1):
            band = ds.GetRasterBand(i)
            band.WriteArray(data[:, :, i - 1])
            band.SetNoDataValue(self.nd[i - 1])
            band.FlushCache()
            band = None
        ds = None

    def flush(self):
        """
        Push data to the network
        :return:
        """
        pass

    def __getitem__(self, s):
        ds = gdal.Open(self.raster_path)
        offset = gdal_args_from_slice(s, self.shape)
        return ds.ReadAsArray(*offset)

    def __setitem__(self, s, a):
        xoff, yoff, xsize, ysize = gdal_args_from_slice(s, self.shape)
        a = np.atleast_3d(np.broadcast_to(a, shape=(ysize, xsize)))

        if a.shape[0] != ysize:
            raise IndexError('Array dimension 0 of size {} not equal to slice ({})'.format(a.shape[0], ysize))
        if a.shape[1] != xsize:
            raise IndexError('Array dimension 1 of size {} not equal to slice ({})'.format(a.shape[1], xsize))

        ds = gdal.Open(self.raster_path, gdalconst.GA_Update)
        if ds is None:
            raise H5FError('Unable to open the raster file {}'.format(self.raster_path))
        for i in range(1, a.shape[2] + 1):
            band = ds.GetRasterBand(i)
            band.WriteArray(a[:, :, i - 1], xoff, yoff)
            band.FlushCache()
            band = None
        ds = None

    @property
    def raster_path(self):
        return self.path + '.tif'


def dtype_to_gdal(dtype):
    dtypecorr = {
        'uint8': 'Byte',
        'int8': 'Int16',
        'uint16': 'UInt16',
        'int16': 'Int16',
        'uint32': 'UInt32',
        'int32': 'Int32',
        'uint64': 'Float64',
        'int64': 'Float64',
        'float32': 'Float32',
        'float64': 'Float64'
    }
    try:
        return dtypecorr[dtype]
    except KeyError:
        return dtypecorr[dtype.__name__]


def dtype_to_np(dtype):
    dtypecorr = {
        'Byte': 'uint8',
        'UInt16': 'uint16',
        'Int16': 'int16',
        'UInt32': 'uint32',
        'Int32': 'int32',
        'Float32': 'float32',
        'Float64': 'float64'
    }
    return dtypecorr[dtype]


def gdal_args_from_slice(s, shape):
    """Factory for __getitem__ and __setitem__"""
    if type(s) == int:
        xoff = 0
        yoff = s
        win_xsize = shape[1]
        win_ysize = 1
    elif type(s) == tuple:
        # Convert numpy objects to integers
        s = [int(o) if 'numpy' in str(type(o)) else o for o in s]
        if type(s[0]) == int:
            yoff = s[0]
            win_ysize = 1
        elif s[0] is None:
            yoff = 0
            win_ysize = shape[0]
        else:
            yoff = s[0].start
            start = yoff
            if start is None:
                start = 0
                yoff = 0
            stop = s[0].stop
            if stop is None:
                stop = shape[0]
            win_ysize = stop - start
        if type(s[1]) == int:
            xoff = s[1]
            win_xsize = 1
        elif s[1] is None:
            xoff = 0
            win_xsize = shape[1]
        else:
            xoff = s[1].start
            start = xoff
            if start is None:
                start = 0
                xoff = 0
            stop = s[1].stop
            if stop is None:
                stop = shape[1]
            win_xsize = stop - start
    elif type(s) == slice:
        xoff = 0
        win_xsize = shape[1]
        if s.start is None:
            yoff = 0
        else:
            yoff = s.start
        if s.stop is None:
            stop = shape[0]
        else:
            stop = s.stop
        win_ysize = stop - yoff
    return xoff, yoff, win_xsize, win_ysize
