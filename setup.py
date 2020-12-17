from setuptools import setup, find_packages

setup(name="fractalmarkets",
      version='1.0.1-beta',
      description='Implementation of Benoit Mandelbrot\'s Brownian Motion in Multifractal Time and Edgar E. Peter\'s Fractal Market Analysis.',
      author='Jona Scott & Daniel Luftspring',
      author_email='jrscott.w@gmail.com',
      url='https://github.com/hyperstripe50/fractal-market-analysis',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy'
        ]
      )