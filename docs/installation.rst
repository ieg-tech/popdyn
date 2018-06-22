.. _install:

=================
Installation
=================

.. contents:: Table of contents
    :depth: 1
    :local:

Module Installation
-------------------

It is expected that ``popdyn`` will be available in the Python Package Index soon, which will require only:

``pip install popdyn``

However, ``popdyn`` must currently be requested from the developers, and is on a private repository. Once you have
obtained the ``popdyn`` package, it can be installed from the root ``popdyn`` package directory using ``pip``:

``pip install .``

GDAL must be installed separately. See :ref:`gdal` to install the GIS requirements of the domain if using rasters.

Requirements Only
-----------------

#. Ensure your working directory is set to the root of the ``popdyn`` package
#. Install using the python package manager: ``pip install -r requirements.txt``

GDAL must be installed separately. See :ref:`gdal` to install the GIS requirements of the domain if using rasters.

.. _gdal:

GDAL
^^^^

The recommended way to build the GDAL dependency is to use the
`Anaconda Python Distribution <https://www.anaconda.com/download>`_, which is easily installed using their graphical
installer.

Once installed, GDAL can be built and installed using the following command:

``conda install gdal``

If not using Anaconda, review the `GDAL documentation <http://www.gdal.org/>`_ to learn about installation.