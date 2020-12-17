# fractalmarkets
## Motivation
We are hard pressed to find a concrete implementation of both Beniot Mandelbrot's "A Multifractal Model of Asset Returns" and Edgar E. Peters' "Fractal Market Analysis" so, with apologies to the Authors, we attempt to fill this vacancy.

## Installation
```
pip install fractalmarkets
```

## Features
This package offers time series simulation processes as well as rescaled range analysis.

The time series simulations are implemented as per Beniot Mandelbrot's description in "A Multifractal Model of Asset Returns", and further elaborated on within "Scaling in financial prices: III. Cartoon Brownian motions in multifractal time". 

The rescaled range analysis is implemented as per Edgar E. Peters "Fractal Market Analysis".

## Simulation Processes
* fractalmarkets
  * mmar
    * BrownianMotion
    * BrownianMotionMultifractalTime
  * rs
    * RS
    
## Usage
To simulate timeseries with ```fma```, instantiate the simulation process that you want with the required parameters and run ```simulate```.

### Brownian Motion in Multifractal Time
```python
from fractalmarkets.mmar.brownian_motion_multifractal_time import BrownianMotionMultifractalTime
import matplotlib.pyplot as plt; plt.style.use('ggplot')
from scipy import interpolate
import numpy as np

bmmt = BrownianMotionMultifractalTime(9, x=0.457, y=0.603, randomize_segments=True, randomize_time=True, M=[0.6, 0.4])
data = bmmt.simulate() # [ [x, y], ..., [x_n, y_n]]

f = interpolate.interp1d(data[:,0], data[:,1])

y = f(np.arange(0, 1, .001))
x = np.linspace(0, 1, len(y), endpoint=True)

y_diff = [b - a for a, b in zip(y[:-1], y[1:])]

fig, axs = plt.subplots(2)
fig.suptitle('MMAR')

axs[0].plot(x, y, 'b-')
axs[1].plot(x[:-1], y_diff)

z1 = np.array(y)
z2 = np.array([0] * len(y))

axs[0].fill_between(x, y, 0,
                where=(z1 >= z2),
                alpha=0.30, color='green', interpolate=True)

axs[0].fill_between(x, y, 0,
                where=(z1 < z2),
                alpha=0.30, color='red', interpolate=True)

plt.show()
```

### Brownian Motion
```python
from fractalmarkets.mmar.brownian_motion import BrownianMotion
import matplotlib.pyplot as plt; plt.style.use('ggplot')
from scipy import interpolate
import numpy as np

bm =  BrownianMotion(9, .457, .603, randomize_segments=True)
data = bm.simulate() # [ [x, y], ..., [x_n, y_n]]

f = interpolate.interp1d(data[:,0], data[:,1])

y = f(np.arange(0, 1, .001))
x = np.linspace(0, 1, len(y), endpoint=True)

y_diff = [b - a for a, b in zip(y[:-1], y[1:])]

<<<<<<< HEAD
fig, axs = plt.subplots(2)
fig.suptitle('MMAR')

axs[0].plot(x, y, 'b-')
axs[1].plot(x[:-1], y_diff)

z1 = np.array(y)
z2 = np.array([0] * len(y))

axs[0].fill_between(x, y, 0,
                where=(z1 >= z2),
                alpha=0.30, color='green', interpolate=True)

axs[0].fill_between(x, y, 0,
                where=(z1 < z2),
                alpha=0.30, color='red', interpolate=True)

plt.show()
```

### RS Analysis
```python
from fractalmarkets.rs.rs import RS
from fractalmarkets.mmar.brownian_motion_multifractal_time import BrownianMotionMultifractalTime

bmmt = BrownianMotionMultifractalTime(9, x=4/9, y=0.603, randomize_segments=False, randomize_time=False, M=[0.6, 0.4])
data = bmmt.simulate()

rs = RS(data[1:,1]) # timeseries starts at zero which must be omitted to avoid division error
(H, c) = rs.get_Hc()
print("Estimated H from RS Analysis: {}".format(H))

rs.plot_vstat() # plot vstat and RS
```
```
>> Estimated H from RS Analysis: 0.6237751068336207
```
![R/S Annalysis](https://github.com/hyperstripe50/fractal-market-analysis/blob/master/examples/RSA.png)

## Developer Guide
### Install virtualenv
```javascript
pip install virtualenv
```

### Create a virtual environment
```javascript
// from fractal_market_analysis directory
python -m virtualenv venv
// sometimes that does not work so try below
// virtualenv venv
```

### Start the virtual environment
```javascript
// from fractal_market_analysis directory
source venv/Scripts/activate
// it is possible that your venv has a bin directory rather than a Scripts directory. If so run the following
// source venv/bin/activate
```

### Install proper packages
```javascript
// from fractal_market_analysis directory
pip install -r requirements.txt
```

### Install our application and watch for edits
```javascript
// from fractal_market_analysis directory. This will allow us to use fma imports in our modules.
pip install -e .
```

### Bonus for the anaconda users...
```javascript
// from fractal_market_analysis directory
conda env create -f environment.yml
conda activate fractalmarkets
```

### Run an example
```javascript
// from fractal_market_analysis directory
python examples/rs_analysis.py // or any other file
```

### Run the Tests
```javascript
// from fractal_market_analysis directory
pytest
```
