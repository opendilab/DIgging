from __future__ import division
from __future__ import print_function

import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
meta = {}
with open(os.path.join(here, 'digging', '__init__.py'), 'r') as f:
    exec(f.read(), meta)

description = """DIgging: """

setup(
    name=meta['__TITLE__'],
    version=meta['__VERSION__'],
    description=meta['__DESCRIPTION__'],
    long_description=description,
    author=meta['__AUTHOR__'],
    license='Apache License, Version 2.0',
    keywords='searching',
    packages=[
        *find_packages(include=('digging', 'digging.*')),
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch<=1.11',
        'di-engine>=0.3',
        'scikit-learn>=0.18.0',
        'protobuf<=3.20.1',  # for di-engine 0.3.1 bug
    ],
    extras_require={
        'doc': [
            'sphinx>=2.2.1',
            'sphinx_rtd_theme~=0.4.3',
            'enum_tools',
            'sphinx-toolbox',
            'recommonmark',
            'sphinx-multiversion~=0.2.4',
        ],
        'test': [
            'pytest~=6.2.5',
            'pytest-cov~=3.0.0',
            'pytest-mock~=3.6.1',
            'pytest-xdist>=1.34.0',
            'pytest-rerunfailures~=10.2',
            'pytest-timeout~=2.0.2',
            'pytest-benchmark~=3.4.0'
        ],
        'style': [
            'yapf==0.29.0',
            'flake8',
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

)