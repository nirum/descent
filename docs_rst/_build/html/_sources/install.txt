============
Installation
============

Basic
-----

The easiest way to install is with ``pip``:

.. code:: bash

    $ pip install descent


You can also install from source by grabbing the code from GitHub:

.. code:: bash

    $ git clone https://github.com/nirum/descent.git
    $ cd descent
    $ pip install -r requirements.txt
    $ python setup.py install

Dependencies
------------

Descent works on Python 3.3-3.5. It has only been tested on CPython (not tested on PyPy yet!).

In addition, descent requires the following packages:

- ``numpy`` 
- ``toolz``
- ``multipledispatch``
- ``tableprint``

And the following are optional (necessary for some features):

- ``scipy``

Development
-----------

Please submit any and all issues to the GitHub issue tracker.

To contribute to descent, you'll need to also install ``sphinx`` and ``numpydoc`` for documentation and
``nose`` for testing. We adhere to the `NumPy/SciPy documentation standards <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard>`_.
