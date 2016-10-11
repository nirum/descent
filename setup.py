from setuptools import setup, find_packages

setup(
    name='descent',
    version='0.2.2',
    description='First order optimization tools',
    author='Niru Maheshwaranathan',
    author_email='nirum@stanford.edu',
    url='https://github.com/nirum/descent',
    install_requires=[
        'six',
        'numpy',
        'toolz',
        'scipy',
        'multipledispatch',
        'custom_inherit',
        'tableprint',
    ],
    extras_require={
        'dev': [],
        'test': ['coveralls', 'pytest', 'coverage'],
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
    license='MIT'
)
