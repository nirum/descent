import re
import os
from setuptools import setup, find_packages

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
with open(os.path.join(__location__, 'descent/metadata.py'), 'r') as f:
    metadata = dict(re.findall(r"__([a-z_]+)__\s*=\s*'([^']+)'", f.read()))

setup(
    name='descent',
    version=metadata['version'],
    description=metadata['description'],
    author=metadata['author'],
    author_email=metadata['author_email'],
    url=metadata['url'],
    install_requires=[
        'numpy',
        'toolz',
        'scipy',
        'multipledispatch',
        'custom_inherit',
        'tableprint',
    ],
    extras_require={
        'dev': [],
        'test': ['flake8', 'pytest', 'coverage'],
    },
    long_description='''
        The descent package contains tools for performing first order
        optimization of functions. That is, given the gradient of an
        objective you wish to minimize, descent provides algorithms for
        finding local minima of that function.
        ''',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering'],
    packages=find_packages(),
    license=metadata['license']
)
