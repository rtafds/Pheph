#!/usr/bin/env python

from __future__ import absolute_import, print_function, division
import os
import codecs
from fnmatch import fnmatchcase
from distutils.util import convert_path
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Education
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: Python
Topic :: Software Development :: Code Generators
Topic :: Software Development :: Compilers
Topic :: Scientific/Engineering :: Mathematics
Operating System :: Unix
Operating System :: MacOS
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
"""
NAME                = 'Pheph'
MAINTAINER          = "rtafds"
MAINTAINER_EMAIL    = "n.rtafds@gmail.com"
DESCRIPTION         = ('Find the optimal explanatory variables ' +
                       'by using ML adn GA')
LONG_DESCRIPTION    = codecs.open("Readme.rst", encoding='utf-8').read()
URL                 = "https://github.com/rtafds/Pheph"
DOWNLOAD_URL        = ""
LICENSE             = 'BSD'
CLASSIFIERS         = [_f for _f in CLASSIFIERS.split('\n') if _f]
AUTHOR              = "rtafds"
AUTHOR_EMAIL        = "n.rtafds@gmail.com"
PLATFORMS           = ["Linux","Mac OS-X"]


def find_packages(where='.', exclude=()):
    out = []
    stack = [(convert_path(where), '')]
    while stack:
        where, prefix = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where, name)
            if ('.' not in name and os.path.isdir(fn) and
                    os.path.isfile(os.path.join(fn, '__init__.py'))):
                out.append(prefix + name)
                stack.append((fn, prefix + name + '.'))
    for pat in list(exclude) + ['ez_setup', 'distribute_setup']:
        out = [item for item in out if not fnmatchcase(item, pat)]
    return out



def do_setup():
    setup(name=NAME,
          #description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          classifiers=CLASSIFIERS,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          url=URL,
          license=LICENSE,
          platforms=PLATFORMS,
          packages=find_packages(),
          install_requires=['numpy','pandas','sklearn','deap'],
          # pygments is a dependency for Sphinx code highlight
          extras_require={
              'test': ['nose>=1.3.0', 'parameterized', 'flake8'],
              'doc': ['Sphinx>=0.5.1', 'pygments']
          },
          package_data={
              '': ['*.txt', '*.rst', '*.cu', '*.cuh', '*.c', '*.sh', '*.pkl',
                   '*.h', '*.cpp', 'ChangeLog', 'c_code/*'],
              'theano.misc': ['*.sh'],
              'theano.d3viz': ['html/*', 'css/*', 'js/*']
          },
          entry_points={
              'console_scripts': ['theano-cache = bin.theano_cache:main',
                                  'theano-nose = bin.theano_nose:main']
          },
          keywords=' '.join([
              'theano', 'math', 'numerical', 'symbolic', 'blas',
              'numpy', 'gpu', 'autodiff', 'differentiation'
          ]),
    )


if __name__ == "__main__":
    do_setup()