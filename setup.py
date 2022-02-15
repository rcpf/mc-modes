#!/usr/bin/env python
# coding=utf-8

import codecs
import os
from distutils.core import setup

from setuptools import find_packages

setup_path = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(setup_path, 'README.rst'), encoding='utf-8') as f:
    README = f.read()

setup(name='oc-modes',
      version='0.1',
      url='https://github.com/rcpf/modes',
      maintainer='Rogério C. P. Fragoso',
      maintainer_email='rcpf@cin.ufpe.br',
      author='Rogério C. P. Fragoso',
      author_email='rcpf@cin.ufpe.br',
      description='One-class Classifier Dynamic Ensemble Selection for Multi-class problems',
      long_description=README,
      license='MIT',

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      install_requires=[
            'numpy~=1.19.5',
            'pandas~=1.1.5',
            'scikit-learn~=0.24.2',
            'rpy2==3.4.5',
      ],
      python_requires='>=3.6',

      packages=find_packages())
