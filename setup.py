from setuptools import setup, find_packages
import descent

setup(name='descent',
      version=descent.__version__,
      description='First order optimization tools',
      author='Niru Maheshwaranathan',
      author_email='nirum@stanford.edu',
      url='https://github.com/nirum/descent.git',
      requires=[i.strip() for i in open("requirements.txt").readlines()],
      long_description='''
          The descent package contains tools for performing first order
          optimization of functions. That is, given the gradient of an
          objective you wish to minimize, descent provides algorithms for
          finding local minima of that function.
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      packages=find_packages(),
      license='LICENSE.md'
      )
