#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

__minimum_numpy_version__ = '1.9.0'
__minimum_tensorflow_version__ = '1.4.0'

setup_requires = ['numpy>=' + __minimum_numpy_version__, 
'tensorflow>='+__minimum_tensorflow_version__]

setup(name='heterogp',
      version='0.0.2',
      description='heteroscedastic error SVGP',
      author=['Josh Albert'],
      author_email=['albert@strw.leidenuniv.nl'],
    setup_requires=setup_requires,  
    tests_require=[
        'pytest>=2.8',
    ],
    package_dir = {'':'src'},
    packages=find_packages('src')
     )

