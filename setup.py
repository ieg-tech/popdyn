#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

version = open('popdyn/VERSION', 'r').read().strip()


setup(name='popdyn',
      version=version,
      description='popdyn is a population dynamics simulator that fits nicely in the python ecosystem',
      url='https://bitbucket.org/alceslanduse/popdyn',
      author='Devin Cairns',
      install_requires=[],
      author_email='dcairns@alces.ca',
      license='Copyright, ALCES',
      packages=['popdyn'],
      zip_safe=False)
