# Fractal Market Analysis Code

## QuickStart
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

### Bonus for the anaconda users...
```javascript
// from fractal_market_analysis directory
conda env create -f environment.yml
conda activate fma
```

## Run the program
```javascript
// from fractal_market_analysis directory
python rescaled-range-analysis/__init__.py
```

## Run the Tests
```javascript
// from fractal_market_analysis directory
pytest
```