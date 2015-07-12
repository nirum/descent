# Descent

**Descent is just getting started and is under active development. Please check back soon for more updates!**

[![Build Status](https://travis-ci.org/nirum/descent.svg?branch=master)](https://travis-ci.org/nirum/descent)
[![Coverage Status](https://coveralls.io/repos/nirum/descent/badge.svg?branch=master&service=github)](https://coveralls.io/github/nirum/descent?branch=master)

Descent is a package for performing first-order optimization. It is just getting of the ground, check back soon for more updates.

It contains routines for running a number of different optimization algorithms, given a function that computes the gradient of the objective you wish to optimize.

It also contains a bunch of useful helper files for converting parameter structures (lists or dictionaries) into arrays and back.

# Requirements

- Python 2.7 or higher
- [numpy](http://www.numpy.org)
- [toolz](https://github.com/pytoolz/toolz)
- [multipledispatch](https://github.com/mrocklin/multipledispatch)

# Algorithms

## Implemented
- Gradient descent (with momentum) `gdm`

## Todo
- Stochastic gradient descent `sgd`
- Stochastic average gradient `sag`
- RMSprop `rmsprop`

# Features (coming soon)

- Logging and display updates each iteration
- Visualization (post-optimization)
