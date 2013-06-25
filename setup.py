'''
Created on Nov 18, 2012

@author: andre
'''

import sys, os
from specmorph.version import __version__, __short_version__
import numpy

from distutils.core import setup
from Cython.Build import cythonize

setup(name='SpecMorph',
      version=__version__,
      description='Spectral morphology decomposition of galaxies',
      author='Andre Luiz de Amorim',
      author_email='streetomon@gmail.com',
      license='MIT',
      url='https://github.com/streeto/SpecMorph/',
      download_url='https://github.com/streeto/SpecMorph/archive/master.zip',
      packages=['specmorph'],
      ext_modules=cythonize(['specmorph/*.pyx']),
      include_dirs=[numpy.get_include()],
      provides=['specmorph'],
      requires=['numpy', 'scipy', 'pystarlight', 'pycasso', 'astropy'],
      keywords=['Scientific/Engineering'],
      classifiers=[
                   'Development Status :: 4 - Beta',
                   'Programming Language :: Python',
                   'License :: OSI Approved :: MIT License',
                  ],
     )