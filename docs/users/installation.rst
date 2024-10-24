Installation
============

We provide AutoCnet as a binary package via conda and for
installation via the standard setup.py script.

Via Conda
---------

1. Download and install miniconda using the `Python 3.x Miniconda installer`_.  Respond ``Yes`` when prompted to add conda to your BASH profile.
2. (Optional) We like to sequester applications in their own environments to avoid any dependency conflicts.  To do this::
   - ``conda create -n <your_environment_name> && source activate <your_environment_name>``
3. Bring up a command line and add three channels to your conda config (``~/condarc``)::
   - ``conda config --add channels usgs-astrogeology``

   - ``conda config --add channels conda-forge``

   - ``conda config --add channels menpo``
4. Finally, install autocnet:: 
   conda install -c usgs-astrogeology autocnet

Via setup.py
------------
This method assumes that you have the necessary dependencies already
installed. The installation of dependencies can be non-trivial because of GDAL.
We supply an ``environment.yml`` file that works with Anaconda Python's ``conda
env`` environment management tool.

Manual Development Environment
------------------------------
To manually install AutoCnet (for example in a development environment) we must install the necessary dependencies.

1. Create a virtual environment:  ``conda create -n autocnet && source activate autocnet``
2. As above, add conda-forge to the channel list.
3. To install the planetary I/O module and the OpenCV computer vision module: ``conda install -c usgs-astrogeology plio opencv3``
4. To install the optional VLFeature module (for SIFT): ``conda install -c conda-forge vlfeat``
5. To install the Cython wrapper to vlfeat: ``conda install -c menpo cyvlfeat``
6. To install PIL and PySAL: ``pip install pillow pysal``
7. To install additional conda packages: ``conda install scipy networkx numexpr dill cython pyyaml matplotlib``

This ensures that all dependencies needed to run AutoCnet are availble.  We also have development dependencies to
support automated testing, documentation builds, etc.

1. Install Nose and Sphinx: ``conda install nose sphinx``
2. Install coveralls: ``pip install coveralls``
3. Install the nbsphinx plugin: ``pip install nbshpinx``
4. Install Jupyter for notebook support: ``conda install jupyter``

.. _Python 3.x Miniconda installer: https://www.continuum.io/downloads
