# fma
## Motivation
We are hard pressed to find a concrete implementation of both Beniot Mandelbrot's "A Multifractal Model of Asset Returns" and Edgar E. Peters' "Fractal Market Analysis" so, with apologies to the Authors, we attempt to fill this vacancy.

## Installation
TODO: make the ```fma``` package available on pypi so that it can be installed by 
```
pip install fma
```

## Features
This package offers time series simulation processes as well as rescaled range analysis.

The time series simulations are implemented as per Beniot Mandelbrot's description in "A Multifractal Model of Asset Returns", and further elaborated on within "Scaling in financial prices: III. Cartoon Brownian motions in multifractal time". 

The rescaled range analysis is implemented as per Edgar E. Peters "Fractal Market Analysis".

## Simulation Processes
* fma
  * mmar
    * BrownianMotion
    * BrownianMotionMultifractalTime
  * rs
    * RS
    
## Usage
To simulate timeseries with ```fma```, instantiate the simulation process that you want with the required parameters and run ```simulate```.

### Brownian Motion in Multifractal Time
```python
from fma.mmar.brownian_motion_multifractal_time import BrownianMotionMultifractalTime

bmmt = BrownianMotionMultifractalTime(9, x=0.457, y=0.603, randomize_segments=True, randomize_time=True, M=[0.6, 0.4])
data = bmmt.simulate() # [ [x, y], ..., [x_n, y_n]]
```

### Brownian Motion
```python
from fma.mmar.brownian_motion import BrownianMotion

bm =  BrownianMotion(12, .457, .603, randomize_segments=True)
data = bm.simulate() # [ [x, y], ..., [x_n, y_n]]
```

### RS Analysis
```python
from fma.rs.rs import RS

bmmt = BrownianMotionMultifractalTime(9, x=4/9, y=0.603, randomize_segments=False, randomize_time=False, M=[0.6, 0.4])
data = bmmt.simulate()
print("Expected H {}".format(bmmt.get_H()))

rs = RS(data[1:,1]) # timeseries starts at zero which must be omitted to avoid division error
(H, c, data) = rs.get_H()

rs.plot_vstat() # plot vstat and RS
```
```
>> Expected H 0.6237751068336207
>> H=0.6465, c=-0.0362
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
// install individual reqs that do not install well on arm devices
pip install matplotlib==3.3.1
pip install scipy==1.5.2
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
conda activate fma
```

### Run an example
```javascript
// from fractal_market_analysis directory
python examples/rs_log_log.py // or any other file
```

### Run the Tests
```javascript
// from fractal_market_analysis directory
pytest
```
