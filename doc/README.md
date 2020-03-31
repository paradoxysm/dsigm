# Documentation

All documentation for dsigm is located here!

[**pydoc**](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc) : Documentation regarding python classes and functions.

[**guides**](https://github.com/paradoxysm/dsigm/tree/master/doc/guides) : Guides on using DSIGM for cluster analysis and classification.

# DSIGM Overview

The Density-sensitive Self-stabilization of Independent Gaussian Mixtures (DSIGM) Clustering Algorithm is a novel algorithm that seeks to identify ideal clusters in data that allows for predictive classifications. DSIGM can be conceptualized as a two layer clustering algorithm. The base layer is a Self-stabilizing Gaussian Mixture Model (SGMM) that identifies the mixture components of the underlying distribution of data. This is followed by a top layer clustering algorithm that seeks to group these components into clusters in a density sensitive manner. The result is a clustering that allows for variable and irregularly shaped clusters that can sensibly categorize new data assumed to be part of the same distribution.

# SGMM
**A Self-stabilizing Gaussian Mixture Model**

The SGMM acts as the base clustering layer for the DSIGM algorithm. The SGMM is a modified Gaussian Mixture Model (GMM) capable of automatically determining an optimal number of components.

When self-stabilization is disabled, the SGMM becomes a normal GMM, modelled after sklearn's GaussianMixture. Implementation will not be covered in depth here and can be found in sklearn's documentation [here](https://scikit-learn.org/stable/modules/mixture.html#gmm). In brief, the SGMM initializes *k* components using the k-means algorithm, and then using the Expectation-Maximization algorithm, iteratively updates the model parameters to maximize the log likelihood until convergence.

Self-stabilization is based on an exploratory approach attempting to minimize a composite (ABIC) of the Akaike Information Criterion (AIC) ad the Bayesiang Information Criteria (BIC). Both AIC and BIC are popular criteria for model selection, with BIC using a stronger penalty for excessive components to prevent overfitting. The BIC can be plotted over different number of components, *k*, to generate a curve where the minimum points to the ideal *k*, *k'*. Past usages of BIC for automatically finding *k'* typically initialize a model with one component and iteratively increment the number of components until the BIC has been minimized. While sufficient when *k'* is small, should *k'* be large, these methods are inefficient. ABIC is used to balance between the tendencies for BIC to prefer smaller models and AIC to prioritize fitting, albeit at the cost of larger models.

The SGMM approaches this optimization as iteratively narrowing an interval that contains *k'* until it converges on the ideal number of components. Beginning with the user-defined *k*, SGMM orients itself to produce this interval based on ABIC. It then repeatedly narrows the interval by splitting the interval at the midpoint and determining in which half-interval, *k'* is contained. This strategy can significantly cut the number of iterations, particularly for large *k'*.

For specific implementation details on how SGMM self-stabilizes, see the guide [here](https://github.com/paradoxysm/dsigm/blob/master/doc/guides/SGMM_stabilization.ipynb).

# DSG
**Density Sensitive Grouping**

The DSG acts as the top clustering layer for the DSIGM algorithm. Through agglomerative hierarchical clustering, DSG groups the components produced by the SGMM into clusters. Similarity between components is measured as a variance-weighted Euclidean distance. In this regard, the distance between the means of any two components is modified by the variances of the two components in each other's direction. DSG provides a dendrogram hierarchy. Clustering is done with Ward's method, using a criterion that composes inertia (variance) from a between-cluster inertia and a within-cluster inertia. Partitioning of the dendrogram is done automatically by a criterion proposed by Husson and Josse [here](http://factominer.free.fr/more/HCPC_husson_josse.pdf). Namely, the ratio of the change of between-cluster inertia from *Q* - 1 clusters to *Q* clusters to the change of between-cluster inertia from *Q* clusters to *Q* + 1 clusters is minimized.
