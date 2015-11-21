# Descent

[![Build Status](https://travis-ci.org/nirum/descent.svg?branch=master)](https://travis-ci.org/nirum/descent)
[![Documentation Status](https://readthedocs.org/projects/descent/badge/?version=latest)](http://descent.readthedocs.org/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/nirum/descent/badge.svg?branch=master&service=github)](https://coveralls.io/github/nirum/descent?branch=master)
[![PyPi version](https://img.shields.io/pypi/v/descent.svg)](https://pypi.python.org/pypi/descent)

*Descent is under active development and subject to change*

Descent is a package for performing constrained and unconstrained first-order optimization.

It contains routines for running a number of different optimization algorithms, given a function that computes the gradient of the objective you wish to optimize.

It also contains a bunch of useful helper files for converting parameter structures (lists or dictionaries) into arrays and back.

Full documentation is available at [descent.readthedocs.org](http://descent.readthedocs.org/en/latest/).

## Contact
For bugs, comments, concerns: use the Github issue tracker.

Author: [Niru Maheswaranathan](http://niru.org/), nirum [a] stanford.edu

## License
MIT. See `LICENSE.md`

## Requirements

- Python 3.3-3.5 or Python 2.7
- [numpy](http://www.numpy.org)
- [toolz](https://github.com/pytoolz/toolz)
- [multipledispatch](https://github.com/mrocklin/multipledispatch)
