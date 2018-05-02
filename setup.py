#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import pkg_resources


version = open('popdyn/VERSION', 'r').read().strip()


def is_installed(name):
    try:
        pkg_resources.get_distribution(name)
        return True
    except:
        return False


requires = ['h5py', 'dask', 'numba', 'xlsxwriter', 'python-dateutil']

setup(name='popdyn',
      version=version,
      description='popdyn is a population dynamics simulator that fits nicely in the python ecosystem',
      url='https://bitbucket.org/alceslanduse/popdyn',
      author='Devin Cairns',
      install_requires=requires,
      author_email='dcairns@alces.ca',
      license='Copyright, ALCES',
      packages=['popdyn'],
      zip_safe=False)
