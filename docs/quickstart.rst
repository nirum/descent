==========
Quickstart
==========

Overview
--------
This document is a work in progress, but for now, check out example code:

Gradient-based algorithms
-------------------------
Each of the gradient based algorithms has the following interface. Given a function ``f_df``
that computes the objective and gradient of the function you want to minimize:

.. code:: python

    >>> opt = descent.GradientDescent(theta_init, f_df, 'sgd', {'lr': learning_rate})
    >>> opt.run(maxiter=1000)
    >>> plt.plot(opt.theta)

Proximal algorithms
-------------------
Example code for ADMM, for solving a linear system with a sparsity penalty:

.. code:: python

    >>> opt = descent.Consensus(theta_init)
    >>> opt.add('linsys', A, b)
    >>> opt.add('sparse', 0.1)
    >>> opt.run()
    >>> plt.plot(opt.theta)


Storage
-------
After calling the ``run`` command, the history of objective values is stored on the optimizer object: 

.. code:: python
    
    >>> opt.run(maxiter=1000)
    >>> plt.plot(opt.storage['objective'])


Utilities
---------
Some other features that might be of interest:

- memoization (see: ``descent.utils.wrap``)
- function wrapping (see: ``descent.utils.destruct`` and ``descent.utils.restruct``)
- gradient checking (see: ``descent.check_grad``)


Tutorial
--------
There is a tutorial consisting of jupyter notebooks demoing the features of descent at: `github.com/nirum/descent-tutorial <https://github.com/nirum/descent-tutorial/>`_.
