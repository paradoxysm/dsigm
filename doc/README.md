# Documentation

All documentation for dsigm is located here!

[**pydoc**](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc) : Documentation regarding python classes and functions.

[**guides**](https://github.com/paradoxysm/dsigm/tree/master/doc/guides) : Guides on using DSIGM for cluster analysis and classification.

# DSIGM Overview

The DSIGM Clustering Algorithm is a novel algorithm that seeks to identify ideal clusters in data that allows for predictive classifications. It is based on generative Gaussian Mixture Models (GMMs) along with OPTICS to produce an algorithm that fits a variable number of Gaussian components to the given data and then groups these components into clusters in a density sensitive manner. This allows the DSIGM algorithm to overcome the limitations of GMMs and OPTICS. Namely, DSIGM, optimizing on the Bayesian Information Criterion, self-stabilizes the number of Gaussian components. By the use of Gaussian components, DSIGM provides capacity for predictive classifications in a sensible manner as opposed to OPTICS.
