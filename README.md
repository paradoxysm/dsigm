# DESIGM Clustering Algorithm
**Density-sensitive Evolution-based Self-stabilization of Independent Gaussian Mixtures (DESIGM) Clustering**

## Overview

The DESIGM Clustering Algorithm is a novel algorithm that seeks to identify ideal clusters in data that allows for predictive classifications. It is based on generative Gaussian Mixture Models (GMMs) along with OPTICS to produce an algorithm that fits a variable number of Gaussian components to the given data and then groups these components into clusters in a density sensitive manner. This allows the DESIGM algorithm to overcome the limitations of GMMs and OPTICS. Namely, DESIGM, using an evolutionary approach, self-stabilizes the number of Gaussian components. By the use of Gaussian components, DESIGM provides capacity for predictive classifications in a sensible manner as opposed to OPTICS.

More details regarding DESIGM can be found in the documentation [here](https://github.com/paradoxysm/desigm/tree/master/doc).

## Installation

### Dependencies

desigm requires:
- numpy
- scipy
- sklearn

desigm is tested and supported on Python 3.4+ up to Python 3.7. Usage on other versions of Python is not guaranteed to work as intended.

### User Installation

desigm can be easily installed using ```pip```

```
pip install desigm
```

## Changelog

See the [changelog](https://github.com/paradoxysm/desigm/blob/master/CHANGES.md) for a history of notable changes to desigm.

## Development

desigm is still under development.

There are three main branches for development and release. `master` is the current development build; `staging` is the staging branch for releases; `release` is the current public release build.

## Help and Support

### Documentation

Documentation for kdtrees can be found [here](https://github.com/paradoxysm/desigm/tree/master/doc).

### Issues and Questions

Issues and Questions should be posed to the issue tracker [here](https://github.com/paradoxysm/desigm/issues).
