# Documentation

All documentation for dsigm is located here!

[**pydoc**](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc) : Documentation regarding python classes and functions.

[**guides**](https://github.com/paradoxysm/dsigm/tree/master/doc/guides) : Guides on using DSIGM for cluster analysis and classification.

# DSIGM Overview

The Density-sensitive Self-stabilization of Independent Gaussian Mixtures (DSIGM) Clustering Algorithm is a novel algorithm that seeks to identify ideal clusters in data that allows for predictive classifications. DSIGM can be conceptualized as a two layer clustering algorithm. The base layer is a Self-stabilizing Gaussian Mixture Model (SGMM) that identifies the mixture components of the underlying distribution of data. This is followed by a top layer clustering algorithm that seeks to group these components into clusters in a density sensitive manner. The result is a clustering that allows for variable and irregularly shaped clusters that can sensibly categorize new data assumed to be part of the same distribution.
