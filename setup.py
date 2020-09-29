from setuptools import setup, find_packages

setup(name="fma",
      packages=find_packages(),
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy'
        ]
      )