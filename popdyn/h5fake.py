"""
Slow and painful transition away from HDF5 and h5py using os directories and rasters

Devin Cairns
January, 2019
"""
import os
import shutil
import pickle
import tempfile
from requests import head, models
from osgeo import gdal


class File(object):
    """
    The file instance is actually an os directory structure
    """
    def __init__(self, path, libver='latest'):
        """
        :param path: directory with raster files
        :param libver: kwarg for backwards compatibility
        """
        # Check whether the path is local or http
        try:
            head(path)
            self.network = True
            # TODO: Implement cloud methods

        except models.MissingSchema:
            self.network = False
            if not os.path.isdir(path):
                os.mkdir(path)

        self.path = path
        self.attrs = Attrs(self.path, self.network)

    def keys(self):
        return Group(self.path, self.network).keys()

    def values(self):
        return Group(self.path, self.network).values()

    def __getitem__(self, key):
        tree = self.decompose_key(key)
        if self.network:
            # TODO: Implement cloud methods
            raise NotImplementedError('Need to set up this method for network data sources')
        else:
            path = self.path
            for t in tree[:-1]:
                path = os.path.join(path, t)
                if not os.path.isdir(t):
                    raise KeyError('No object at key "{}"'.format(key))
            last_member = tree[-1]
            if last_member.split('.')[-1] == 'tif':
                last_member = last_member[:-4]
            final_d = os.path.join(path, last_member)
            if os.path.isfile(final_d + '.tif'):
                return Dataset(final_d, self.network)
            else:
                return Group(final_d, self.network)

    def create_group(self, key):
        Group(self.path, self.network).create_group(key)

    def create_dataset(self, key, data=None, shape=None, chunks=None, compression=None):
        if self.network:
            # TODO: Implement cloud methods
            raise NotImplementedError('Need to set up this method for network data sources')
        else:
            tree = self.decompose_key(key)

            # Ensure the tree exists
            path = self.path
            for t in tree[:-1]:
                path = os.path.join(path, t)
                if not os.path.isdir(path):
                    os.mkdir(path)

            # Create the dataset
            path = os.path.join(path, tree[-1])
            Dataset(path, self.network, data=data, shape=shape, chunks=chunks,
                    overwrite=True, defn=self.attrs)

    @staticmethod
    def dir_members(d):
        return {os.path.join(d, f) for f in os.listdir(d) if os.path.isdir(f) or f.split('.')[-1] == 'tif'}

    @staticmethod
    def decompose_key(key):
        if key[0] == '/':
            key = key[1:]
        return key.split('/')

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
    def __init__(self, path, network=False):
        if network:
            # TODO: Implment network attrs
            raise NotImplementedError('Need to set up this method for network data sources')
        else:
            if not os.path.isfile(path):
                with open(os.path.join(path, 'attrs.dmp'), 'wb') as f:
                    pickle.dump({}, f)
        self.path = path
        self.network = network

    def __getitem__(self, key):
        if self.network:
            # TODO: Implment network attrs
            raise NotImplementedError('Need to set up this method for network data sources')
        else:
            with open(os.path.join(self.path, 'attrs.dmp'), 'rb') as f:
                return pickle.load(f)[key]

    def __setitem__(self, key, val):
        if self.network:
            # TODO: Implment network attrs
            raise NotImplementedError('Need to set up this method for network data sources')
        else:
            with open(os.path.join(self.path, 'attrs.dmp'), 'rb') as f:
                d = pickle.load(f)
            d[key] = val
            with open(os.path.join(self.path, 'attrs.dmp'), 'wb') as f:
                pickle.dump(d, f)

    def keys(self):
        with open(os.path.join(self.path, 'attrs.dmp'), 'rb') as f:
            d = pickle.load(f)
        return d.keys()

    def values(self):
        with open(os.path.join(self.path, 'attrs.dmp'), 'rb') as f:
            d = pickle.load(f)
        return d.values()

    def update(self, other_dict):
        if self.network:
            # TODO: Implment network attrs
            raise NotImplementedError('Need to set up this method for network data sources')
        else:
            with open(os.path.join(self.path, 'attrs.dmp'), 'rb') as f:
                d = pickle.load(f)
            d.update(other_dict)
            with open(os.path.join(self.path, 'attrs.dmp'), 'wb') as f:
                pickle.dump(d, f)


class Group(object):
    def __init__(self, path, network):
        self.path = path
        self.attrs = Attrs(path)
        self.network = network

    def keys(self):
        if self.network:
            # TODO: Implement cloud methods
            raise NotImplementedError('Need to set up this method for network data sources')
        else:
            return [f.replace('.tif', '') for f in os.listdir(self.path) if f != 'attrs.dmp']

    def values(self):
        if self.network:
            # TODO: Implement cloud methods
            raise NotImplementedError('Need to set up this method for network data sources')
        else:
            values = [os.path.join(self.path, f) for f in os.listdir(self.path) if f != 'attrs.dmp']
            for val in values:
                if val.split('.')[-1] == 'tif':
                    return Dataset(val, self.network)
                else:
                    return Group(val, self.network)

    def create_group(self, key):
        if self.network:
            # TODO: Implement cloud methods
            raise NotImplementedError('Need to set up this method for network data sources')
        else:
            tree = File.decompose_key(key)
            path = self.path
            for t in tree:
                path = os.path.join(path, t)
                if os.path.isfile(path):
                    raise ValueError('Unable to use key "{}" due to existing file "{}"'.format(key, path))
                if not os.path.isdir(path):
                    os.mkdir(path)

    def __del__(self):
        if self.network:
            # TODO: Implement cloud methods
            raise NotImplementedError('Need to set up this method for network data sources')
        else:
            if os.path.isdir(self.path):
                shutil.rmtree(self.path)


class Dataset(object):
    """
    Shadow of the h5py Dataset class
    """
    def __init__(self, path, network, **kwargs):
        self.network = network
        self.path_key = path
        self.to_flush = False

        if self.network:
            # TODO: Implement cloud methods
            raise NotImplementedError('Need to set up this method for network data sources')
        else:
            if os.path.isfile(self.path) and kwargs.get('overwrite', False):
                os.remove(self.path)

            # Create the dataset if it does not exist
            if not os.path.isfile(self.path):
                self.write_data(kwargs.get('defn'), kwargs.get('chunks'),
                                kwargs.get('shape', None), kwargs.get('data', None))
            else:
                ds = gdal.Open(self.path)
                self.shape = (ds.RasterYSize, ds.RasterXSize)

    def write_data(self, defn, chunks, shape=None, data=None):
        if shape is None and data is None:
            raise ValueError('Either one of "shape" or "data" must be specified when creating a dataset')

        # Create a local file regardless
        if self.network:
            staging_file = os.path.join(tempfile.gettempdir(), os.path.basename(self.path))
            self.to_flush = staging_file
        else:
            staging_file = self.path

        if shape is None:
            shape = data.shape
            dtype = data.dtype.name
        else:
            dtype = 'float64'

        driver = gdal.GetDriverByName('GTiff')
        options = ['TILED=YES', 'COPY_SRC_OVERVIEWS=YES', 'COMPRESS=LZW',
                   'BLOCKXSIZE={}'.format(chunks[1]),
                   'BLOCKYSIZE={}'.format(chunks[0])]

        ds = driver.Create(
            staging_file,
            int(shape[1]),
            int(shape[0]),
            1,
            gdal.GetDataTypeByName(dtype_to_gdal(dtype)),
            options
        )
        if ds is None:
            raise TypeError('GDAL error while creating new raster')

        ds.SetGeoTransform((defn['left'], float(defn['csx']), 0,
                            defn['top'], 0, defn['csy'] * -1.))
        ds.SetProjection(defn['projection'])

        band = ds.GetRasterBand(1)
        if data is None:
            band.WriteArray(data)
        band.SetNoDataValue(defn['nodata'])
        band.FlushCache()
        band = None
        ds = None

        self.shape = shape

    def flush(self):
        """
        Push data to the network
        :return:
        """
        if self.to_flush:
            # TODO: Implement cloud methods
            raise NotImplementedError('Need to set up this method for network data sources')

    def __getitem__(self, s):
        if self.network:
            # TODO: Implement cloud methods
            raise NotImplementedError('Need to set up this method for network data sources')
        else:
            ds = gdal.Open(self.path)
            offset = gdal_args_from_slice(s, self.shape)
            return ds.ReadAsArray(*offset)

    def __setitem__(self, s, a):
        offset = gdal_args_from_slice(s, self.shape)
        if self.to_flush:
            ds = gdal.Open(self.to_flush)
        else:
            ds = gdal.Open(self.path)
        band = ds.GetRasterBand(1)
        band.WriteArray(a, offset[0], offset[1])
        band.FlushCache()

    @property
    def path(self):
        return self.path_key + '.tif'

    def __del__(self):
        if self.network:
            # TODO: Implement cloud methods
            raise NotImplementedError('Need to set up this method for network data sources')
        else:
            if os.path.isfile(self.path):
                os.remove(self.path)


def dtype_to_gdal(dtype):
    dtypecorr = {
        'Byte': 'uint8',
        'Int16': 'int16',
        'Int32': 'int32',
        'UInt16': 'uint16',
        'UInt32': 'uint32',
        'Int64': 'int64',
        'UInt64': 'uint64',
        'Float32': 'float32',
        'Float64': 'float64',
        'Long': 'int64',
    }
    outdtype = {v: k for k, v in dtypecorr.items()}
    outdtype['bool'] = 'Byte'
    try:
        return dtypecorr[dtype]
    except KeyError:
        return outdtype[dtype]


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
