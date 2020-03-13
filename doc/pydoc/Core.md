# dsigm.Core

A Core is `dsigm`'s mixture component, defining a Gaussian distribution with a mean, covariance, and weight.

**Attributes**
```
mu : array-like, shape (n_features,), default=[0]
	The mean vector of the Gaussian Distribution.

sigma : array-like, shape (n_features, n_features), default=[1]
	The variance or covariance vector of the Gaussian Distribution.
	Core uses fully independent vectors.

delta : array-like, shape (1,), default=[1]
	The weight of the Gaussian Distribution as a component in a larger
	Gaussian Mixture

cluster : CoreCluster, default=None
	The parent CoreCluster this Core is associated with.
```

**Methods**
[`__init__`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/Core.md#__init__) : Instantiates a Core.
[`_validate_init`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/Core.md#_validate_init) : Ensures all attributes have valid values.
[`pdf`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/Core.md#pdf) : Calculate the probability densities for some given data.
[`logpdf`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/Core.md#logpdf) : Calculate the log probability densities for some given data.

## __init__
```python
Core(self, mu=[0], sigma=[1], delta=[1], cluster=None)
```

A Core Point that defines a Gaussian Distribution.

**Parameters**
```
mu : array-like, shape (n_features,), default=[0]
	The mean vector of the Gaussian Distribution.

sigma : array-like, shape (n_features, n_features), default=[1]
	The variance or covariance vector of the Gaussian Distribution.
	Core uses fully independent vectors.

delta : array-like, shape (1,), default=[1]
	The weight of the Gaussian Distribution as a component in a larger
	Gaussian Mixture

cluster : CoreCluster, default=None
	The parent CoreCluster this Core is associated with.
```

## _validate_init
```python
Core._validate_init()
```
Validate the attributes of the Core.
If `dim` has not yet been set, set it now.

## pdf
```python
Core.pdf(data)
```

Multivariate normal probability density function.

**Parameters**
```
data : array-like
	Quantiles, with the last axis of `data` denoting the features.
```

**Returns**
```
pdf : ndarray or scalar
	Probability density function evaluated at `data`.
```

## logpdf
```python
Core.logpdf(data)
```

Log of multivariate normal probability density function.

**Parameters**
```
data : array-like
	Quantiles, with the last axis of `data` denoting the features.
```

**Returns**
```
pdf : ndarray or scalar
	Log probability density function evaluated at `data`.
```
