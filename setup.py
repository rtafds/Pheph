from setuptools import setup, find_packages
from os import path
# using non-linear correlation, install minepy or PyHSICLasso
# using visualization, install matplotlib or plotly
# using deep-learning, install pytorch

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst')) as f:
    long_description = f.read()

with open(path.join(here,'LICENSE')) as f:
    license = f.read()
    
def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name='pheph',
    version='0.0.1',
    url='https://github.com/rtafds/',
    author='rtafds',
    author_email='n.rtafds@gmail.com',
    maintainer='rtafs',
    maintainer_email='n.rtafds@gmail.com',
    description='Optimization of experimental conditions',
    #long_description=long_description,
    packages=['pheph'],
    #install_requires=_requires_from_file('requirements.txt'),
    install_requires=['numpy','pnadas','sklearn','deap','minepy','PyHSICLasso'],
    license=license,
    test_suite = 'tests'
)

