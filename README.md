## DSIGM Clustering Algorithm

[![Travis (.com)](https://img.shields.io/travis/com/paradoxysm/dsigm?style=flat-square)](https://travis-ci.com/paradoxysm/dsigm)
[![Codecov](https://img.shields.io/codecov/c/gh/paradoxysm/dsigm?style=flat-square&token=5e48e76aa703404f901dea510983281a)](https://codecov.io/gh/paradoxysm/dsigm)
[![GitHub](https://img.shields.io/github/license/paradoxysm/dsigm?color=blue&style=flat-square)](https://github.com/paradoxysm/dsigm/blob/master/LICENSE)

## Overview

The Density-sensitive Self-stabilization of Independent Gaussian Mixtures (DSIGM) Clustering Algorithm is a novel algorithm that seeks to identify ideal clusters in data that allows for predictive classifications. DSIGM can be conceptualized as a two layer clustering algorithm. The base layer is a Self-stabilizing Gaussian Mixture Model (SGMM) that identifies the mixture components of the underlying distribution of data. This is followed by a top layer clustering algorithm that seeks to group these components into clusters in a density sensitive manner. The result is a clustering that allows for variable and irregularly shaped clusters that can sensibly categorize new data assumed to be part of the same distribution.

More details regarding DSIGM can be found in the documentation [here](https://github.com/paradoxysm/dsigm/tree/0.3.1/doc).

## Installation

### Dependencies

`dsigm` requires:
```
numpy
scipy
sklearn
```
`dsigm` is tested and supported on Python 3.4+ up to Python 3.7. Usage on other versions of Python is not guaranteed to work as intended.

### User Installation

`dsigm` can be easily installed using ```pip```

```
pip install dsigm
```

For more details on usage, see the documentation [here](https://github.com/paradoxysm/dsigm/tree/0.3.1/doc).

## Changelog

See the [changelog](https://github.com/paradoxysm/dsigm/blob/0.3.1/CHANGES.md) for a history of notable changes to dsigm.

## Development

[![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability-percentage/paradoxysm/dsigm?style=flat-square)](https://codeclimate.com/github/paradoxysm/dsigm/maintainability)

`dsigm` is still under development. As of `0.3.1`, only the Self-stabilizing Gaussian Mixture Model (SGMM) has been implemented.

There are three main branches for development and release. [`master`](https://github.com/paradoxysm/dsigm) is the current development build; [`staging`](https://github.com/paradoxysm/dsigm/tree/staging) is the staging branch for releases; [`release`](https://github.com/paradoxysm/dsigm/tree/release) is the current public release build.

## Help and Support

### Documentation

Documentation for `dsigm` can be found [here](https://github.com/paradoxysm/dsigm/tree/0.3.1/doc).

### Issues and Questions

Issues and Questions should be posed to the issue tracker [here](https://github.com/paradoxysm/dsigm/issues).
