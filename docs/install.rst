============
Installation
============

Basic
-----

The fastest way to install is by grabbing the code from Github:

.. code:: bash

    $ git clone https://github.com/baccuslab/pyret.git
    $ cd pyret
    $ pip install -r requirements.txt
    $ python setup.py install

Dependencies
------------

Pyret requires the following dependencies:

- ``numpy`` 
- ``scipy``
- ``matplotlib``
- ``toolz``
- ``multipledispatch``

Development
-----------

To contribute to ``descent``, you'll need to also install ``sphinx`` and ``numpydoc`` for documentation and
``nose`` for testing. We adhere to the `NumPy/SciPy documentation standards <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard>`_.
