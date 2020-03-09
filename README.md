## DSIGM Clustering Algorithm

[![Build Status](https://travis-ci.com/paradoxysm/dsigm.svg?branch=master)](https://travis-ci.com/paradoxysm/dsigm)
[![codecov](https://codecov.io/gh/paradoxysm/dsigm/branch/master/graph/badge.svg?token=cpaPkUHKah)](https://codecov.io/gh/paradoxysm/dsigm)
[![GitHub](https://img.shields.io/github/license/paradoxysm/dsigm?color=blue)](https://github.com/paradoxysm/dsigm/blob/master/LICENSE)

## Overview

The Density-sensitive Self-stabilization of Independent Gaussian Mixtures (DSIGM) Clustering Algorithm is a novel algorithm that seeks to identify ideal clusters in data that allows for predictive classifications. It is based on generative Gaussian Mixture Models (GMMs) along with OPTICS to produce an algorithm that fits a variable number of Gaussian components to the given data and then groups these components into clusters in a density sensitive manner. This allows the DSIGM algorithm to overcome the limitations of GMMs and OPTICS. Namely, DSIGM, optimizing on the Bayesian Information Criterion, self-stabilizes the number of Gaussian components. By the use of Gaussian components, DSIGM provides capacity for predictive classifications in a sensible manner as opposed to OPTICS.

More details regarding DSIGM can be found in the documentation [here](https://github.com/paradoxysm/dsigm/tree/master/doc).

## Installation

### Dependencies

dsigm requires:
- numpy
- scipy
- sklearn

dsigm is tested and supported on Python 3.4+ up to Python 3.7. Usage on other versions of Python is not guaranteed to work as intended.

### User Installation

dsigm can be easily installed using ```pip```

```
pip install dsigm
```

## Changelog

See the [changelog](https://github.com/paradoxysm/dsigm/blob/master/CHANGES.md) for a history of notable changes to dsigm.

## Development

[![Maintainability](https://api.codeclimate.com/v1/badges/db50b93805392126d265/maintainability)](https://codeclimate.com/github/paradoxysm/dsigm/maintainability)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fparadoxysm%2Fdsigm.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2Fparadoxysm%2Fdsigm?ref=badge_shield)

dsigm is still under development.

There are three main branches for development and release. `master` is the current development build; `staging` is the staging branch for releases; `release` is the current public release build.

## Help and Support

### Documentation

Documentation for dsigm can be found [here](https://github.com/paradoxysm/dsigm/tree/master/doc).

### Issues and Questions

Issues and Questions should be posed to the issue tracker [here](https://github.com/paradoxysm/dsigm/issues).
