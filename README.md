RCA++
===
Resolved Component Analysis

v1.0

## Description
RCA++ is a PSF modelling python package. It is an extension to the RCA method which the documentation can be found [here](https://morganschmitz.github.io/rca/). It enforces constraints related to positivity, sparsity and spatial structure to build a spatially-varying PSF model from observed, noisy and possibly undersampled stars and galaxies. Some modicum of documentation can be found [here](https://morganschmitz.github.io/rca/) - see also [quick start](#quick-start) below.

## Requirements
The following python packages are required:

  - numpy
  - scipy
  - [ModOpt](https://github.com/CEA-COSMIC/ModOpt)
  
You will also need a compiled version of the sparse2d module of [ISAP](http://www.cosmostat.org/software/isap); alternatively, you should be able to install [PySAP](https://github.com/CEA-COSMIC/pysap) and let it handle the compilation and installation of sparse2d.

## Installation
After installing all dependencies, RCA++ just needs to be cloned and python-installed:

```
git clone https://github.com/charlesglucas/rca.git
cd rca
python setup.py install
```

## References
  - [Ngolè et al., 2016](https://arxiv.org/abs/1608.08104) - Inverse Problems, 32(12)
  - [Schmitz et al., 2019](https://arxiv.org/abs/1906.07676)
  
## Quick start
The basic syntax to run RCA++ is as follows:

```
from rca import RCA

# initialize RCA instance:
rca_fitter = RCA(4)

# fit it to data
S, A = rca_runner.fit(stars, galaxies, stars_posistions, galaxies_positions)

# return PSF model at positions of interest
psfs = rca_fitter.estimate_psf(galaxy_positions)
```
A complete list of the parameters for `RCA` and its `fit` and `estimate_psf` methods can be found in [the documentation](https://morganschmitz.github.io/rca/rca.html#module-rca). The parameters for `RCA++` are the ones for `RCA` and the boolean parameter `method`. The main parameters to take into account are:

  - RCA++ initialization:
    - `n_comp`, the number of eigenPSFs to learn ("r" in the papers)
    - `upfact`, the upsampling factor if superresolution is required ("m_d" or "D" in the papers)
  - `fit`:
    - `obs_stars` should contain your observed stars (see note below for formatting conventions)
    - `obs_gal` should contain your observed galaxies (see note below for formatting conventions)
    - `stars_pos` and `gal_pos`, their respective positions
    - either `shifts` (with their respective centroid shifts wrt. a common arbitrary grid) or, if they are to be estimated from the data, a rough estimation of the `psf_size` (for the window function - can be given in FWHM, R^2 or Gaussian sigma)
  - `estimate_psf`:
    - `test_pos`, the positions at which the PSF should be estimated
    -  `method` is a boolean to select either the RCA method (`method=1`) and the RCA++ method (`method=2`). By default, `method=2`.

The rest can largely be left to default values for basic usage.

Note `RCA.fit` expects the data to be stored in a `(p, p, n_data)` array, that is, with the indexing (over objects) in the _last_ axis. You can use `rca.utils.rca_format` to convert to the proper format from the more conventional `(n_stars, p, p)`.

An example with simulated images can also be found in the `example` folder.

## Changelog
RCA++ is an extension of RCA which can be accessed [here](https://github.com/CosmoStat/rca). It adds information from galaxies using the [sf_deconvolve](https://github.com/CosmoStat/sf_deconvolve) algorithm.
