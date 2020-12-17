from setuptools import setup, find_packages

setup(name="fma",
      version='0.0.1',
      description='Implementation of Benoit Mandelbrot\'s Brownian Motion in Multifractal Time and Edgar E. Peter\'s Fractal Market Analysis.',
      author='malhar',
      author_email='jrscott.w@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy'
        ]
      )