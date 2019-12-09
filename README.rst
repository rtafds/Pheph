.. -*- mode: rst -*-

|PythonVersion|_

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/scikit-learn.svg
.. _PythonVersion: https://img.shields.io/pypi/pyversions/scikit-learn.svg


Pheph
============
Post hoc ergo propter hoc.
~~~~~~~~~~~~
For experiment design.
Pheph is a Python module for Approximate explanatory variable → objective variable with ML,DL or others and then,
the optimum objective variable is found by GA using the evaluation function made from the objective variable.

It is currently maintained by a rtafds.


Installation
------------

Dependencies
~~~~~~~~~~~~

Phepe requires:

- Python (>= 3.6)
- NumPy (>= 1.12.0)
- Pandas (>= 0.21.0)
- scikit-learn (>= 0.21)
- deap (>=1.3)
- Pytorch

**Pheph 0.00 and later require Python 3.6 or newer.

Pheph plotting capabilities (i.e., functions start with "plot_"
and classes end with "Display") require Matplotlib (>= 1.5.1) or Plotly.
A few examples require pandas >= 0.18.0.

Pheph use non-linear correlation require minepy or PyHSICLasso.

User installation
~~~~~~~~~~~~~~~~~

Clone and current directory Pheph, run the following script in the terminal::

    python setup.py install

or ::

    pip install -e .


Changelog
---------


Development
-----------

I welcome new contributors of all experience levels. 
And I have a hard time collecting data for transfer learning, so if datasetsIf you look for or find data, it will be helpful.
Any field is acceptable, but I want the experimental data on chemistry and materials.

Important links
~~~~~~~~~~~~~~~

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/rtafds/Pheph.git

Contributing
~~~~~~~~~~~~

Testing
~~~~~~~

Simultaneously with installation, you can launch the test suite in the solution directory (you will need to have ``pytest`` >= 3.3.0 installed)::

    python setup.py test


Submitting a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~

Pull Request welcome.


Project History
---------------

The project was started in 2019.

Help and Support
----------------

Documentation
~~~~~~~~~~~~~

Communication
~~~~~~~~~~~~~

- Mailing address: n.rtafds@gmail.com

Citation
~~~~~~~~

If you use Pheph in a scientific publication, we would appreciate citations: Is not yet.
