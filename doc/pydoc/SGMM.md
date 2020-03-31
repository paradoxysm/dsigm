# dsigm.SGMM

The Self-stabilizing Gaussian Mixture Model (SGMM) is a modified Gaussian Mixture Model that is capable of automatically converging to the optimal number of components during fitting. The SGMM refines the number of components by narrowing a search interval through a composite Akaike-Bayesian Information Criterion until converged.

**Attributes**
```
dim : int
	The dimensionality of the model; the number of features the model
	expects.

init_cores : int, default=5
	The initial number of Cores (Gaussian components) to fit the data.

init : {'random', 'kmeans'}, default='kmeans'
	The method used to initialize the weights, the means and the
	precisions.
	Must be one of::
		'kmeans' : responsibilities are initialized using kmeans.
		'random' : responsibilities are initialized randomly.

stabilize : float or None, default=0.5
	A float within [0., 1.] that determines the weighting of
	BIC scores to AIC scores in calculating the ABIC composite.
	Also acts to enable stabilization. If None, stabilization
	is disabled.

n_stabilize : int, default=5
	Number of times the SGMM will run individual fittings during
	the stabilization process.

n_init : int, default=10
	Number of times the SGMM  will be run with different
	Core seeds. The final results will be the best output of
	n_init consecutive runs in terms of inertia.

max_iter : int, default=100
	Maximum number of iterations of the SGMM for a
	single run.

tol : float, default=1e-3
	Relative tolerance with regards to the difference in inertia
	of two consecutive iterations to declare convergence.

reg_covar : float, default=1e-6
	Non-negative regularization added to the diagonal of covariance.
	Allows to assure that the covariance matrices are all positive.

random_state : None or int or RandomState, default=None
	Determines random number generation for Core initialization. Use
	an int to make the randomness deterministic.

inertia : float
	Average of maximal probabilities of each sample to each Core.

converged : bool
	True when convergence was reached in fit(), False otherwise.

cores : array-like, shape (n_cores,)
	A list of Cores.

_data_range : array-like, shape (2, n_features)
	The range that encompasses the data in each axis.
```

**Methods**
- [`__init__`](https://github.com/paradoxysm/dsigm/blob/0.3.1/doc/pydoc/SGMM.md#__init__) : Instantiates an SGMM.
- [`fit`](https://github.com/paradoxysm/dsigm/blob/0.3.1/doc/pydoc/SGMM.md#fit) : Fit the model to some given data.

**Private Methods**
- [`_fit_stabilize`](https://github.com/paradoxysm/dsigm/blob/0.3.1/doc/pydoc/SGMM.md#_fit_stabilize) : A single attempt at fitting with stabilization.
- [`_truncate_interval`](https://github.com/paradoxysm/dsigm/blob/0.3.1/doc/pydoc/SGMM.md#_truncate_interval) : Truncates the search interval for stabilization.
- [`_halve_interval`](https://github.com/paradoxysm/dsigm/blob/0.3.1/doc/pydoc/SGMM.md#_halve_interval) : Halves the search interval for stabilization.
- [`_orient_stabilizer`](https://github.com/paradoxysm/dsigm/blob/0.3.1/doc/pydoc/SGMM.md#_orient_stabilizer) : Initializes the search interval for stabilization.

## __init__
```python
SGMM(self, init_cores=5, init='kmeans', stabilize=0.5, n_stabilize=5, n_init=10, max_iter=100, tol=1e-3, reg_covar=1e-6, random_state=None)
```

A Self-stabilizing Gaussian Mixture Model.

**Parameters**
```
init_cores : int, default=5
	The initial number of Cores (Gaussian components) to fit the data.

init : {'random', 'kmeans'}, default='kmeans'
	The method used to initialize the weights, the means and the
	precisions.
	Must be one of::
		'kmeans' : responsibilities are initialized using kmeans.
		'random' : responsibilities are initialized randomly.

stabilize : float or None, default=0.5
	A float within [0., 1.] that determines the weighting of
	BIC scores to AIC scores in calculating the ABIC composite.
	Also acts to enable stabilization. If None, stabilization
	is disabled.

n_stabilize : int, default=5
	Number of times the SGMM will run individual fittings during
	the stabilization process.

n_init : int, default=10
	Number of times the SGMM  will be run with different
	Core seeds. The final results will be the best output of
	n_init consecutive runs in terms of inertia.

max_iter : int, default=100
	Maximum number of iterations of the SGMM for a
	single run.

tol : float, default=1e-3
	Relative tolerance with regards to the difference in inertia
	of two consecutive iterations to declare convergence.

reg_covar : float, default=1e-6
	Non-negative regularization added to the diagonal of covariance.
	Allows to assure that the covariance matrices are all positive.

random_state : None or int or RandomState, default=None
	Determines random number generation for Core initialization. Use
	an int to make the randomness deterministic.
```

## fit
```python
SGMM.fit(data)
```
Estimate model parameters with the EM algorithm.

The method fits the model `n_init` times and sets
the parameters with which the model has the
largest likelihood or lower bound. Within each trial,
the method iterates between E-step and M-step for
`max_iter` times until the change of likelihood or lower bound
is less than `tol`, otherwise, a ConvergenceWarning is raised.

**Parameters**
```
data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.
```

**Returns**
```
self : SGMM
	Itself, now updated with fitted parameters.
```

## _fit_stabilize
```python
SGMM._fit_stabilize(data)
```
A single attempt to estimate model parameters
with the EM algorithm.

The method repeatedly converges for various n_cores
to pinpoint optimal n_cores. It does so by determining
a search interval that contains the optimal n_cores and
repeatedly narrows the interval until the optimal n_cores
is determined.

**Parameters**
```
data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.
```

**Returns**
```
inertia : float
	Log likelihood of the model.

cores : array-like, shape (n_cores,)
	A list of Cores for this fit trial.
```

## _truncate_interval
```python
SGMM._truncate_interval(data, interval, abic)
```
Truncate the interval by reducing the upper or lower limit
based on which has the higher BIC/AIC (ABIC).

**Parameters**
```
data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.

interval : tuple, shape (2,)
	The interval which contains the optimal number of Cores.
	Interpreted as [min, max].

abic : tuple, shape (2,)
	The abic scores corresponding to the interval.
```

**Returns**
```
interval : tuple, shape (2,)
	The interval which contains the optimal number of Cores.
	Interpreted as [min, max).

abic : tuple, shape (2,)
	The abic scores corresponding to the interval.
```

## _halve_interval
```python
SGMM._halve_interval(data, interval, abic, midpoint, abic_m)
```
Halve the interval based on the BIC/AIC (ABIC) of the midpoint.

**Parameters**
```
data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.

interval : tuple, shape (2,)
	The interval which contains the optimal number of Cores.
	Interpreted as [min, max].

abic : tuple, shape (2,)
	The abic scores corresponding to the interval.

midpoint : int
	The midpoint of the interval.

abic_m : float
	The abic score corresponding to the midpoint.
```

**Returns**
```
interval : tuple, shape (2,)
	The interval which contains the optimal number of Cores.
	Interpreted as [min, max].

abic : tuple, shape (2,)
	The abic scores corresponding to the interval.
```

## _orient_stabilizer
```python
SGMM._orient_stabilizer(data)
```
Orient the initial interval for the stabilizer.

**Parameters**
```
data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.
```

**Returns**
```
interval : tuple, shape (2,)
	The interval which contains the optimal number of Cores.
	Interpreted as [min, max).

bic : tuple, shape (2,)
	The bic scores corresponding to the interval.
```
