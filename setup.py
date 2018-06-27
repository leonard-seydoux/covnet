#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setup script for covnet."""

import setuptools

from numpy.distutils.core import setup

# convert markdown README.md to restructured text .rst for pypi
# pandoc can be installed with
# conda install -c conda-forge pandoc pypandoc
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')

except(IOError, ImportError):
    print('no pandoc installed. Careful, pypi description will not be '
          'formatted correctly.')
    long_description = open('README.md').read()

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: GPLv3.0',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics'
]

KEYWORDS = ['wavefield covariance', 'array processing']

INSTALL_REQUIRES = [
    'numpy (>=1.0.0)']

metadata = dict(
    name='covnet',
    version='0.0.1',
    description='Array processing tools',
    long_description=long_description,
    url='http://bitbuc.ipgp.fr',
    download_url='https://bitbucket.com/xy/zipball/master',
    author='L. Seydoux',
    author_email="",
    license='GPLv3.0',
    keywords=KEYWORDS,
    requires=INSTALL_REQUIRES,
    platforms='OS Independent',
    packages=['covnet'],
    classifiers=CLASSIFIERS
)


setup(**metadata)
