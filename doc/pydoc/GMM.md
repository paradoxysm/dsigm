# dsigm.GMM

The Gaussian Mixture Model (GMM) is based on `sklearn.mixture.GaussianMixture` and is used to model data distributions as a mixture of *k* gaussian components.

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

n_init : int, default=10
	Number of times the GMM  will be run with different
	Core seeds. The final results will be the best output of
	n_init consecutive runs in terms of inertia.

max_iter : int, default=100
	Maximum number of iterations of the GMM for a
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
[`__init__`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#__init__) : Instantiates an GMM.
[`get_params`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#get_params) : Get the model parameters of the GMM.
[`fit`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#fit) : Fit the model to some given data.
[`predict`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#predict) : Label some given data to the best fitting component.
[`score`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#score) : Calculate the per-sample log-likelihood of the model.
[`bic`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#bic) : Calculate Bayesian Information Criterion.
[`aic`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#aic) : Calculate Akaike Information Criterion.
[`abic`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#abic) : Calculate a composite of the Akaike and Bayesian Information Criteria.

**Private Methods**
[`_validate_data`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#_validate_data) : Validate the given data to the correct format.
[`_fit_single`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#_fit_single) : A single attempt at fitting with no stabilization.
[`_estimate_parameters`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#_estimate_parameters) : Estimate new model parameters given posterior probabilities.
[`_initialize`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#_initialize) : Initialize a set of components for the model.
[`_initialize_core`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#_initialize_core) : Initialize a single component for the model.
[`_expectation`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#_expectation) : Conduct the expectation step of the EM Algorithm.
[`_maximization`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#_maximization) : Conduct the maximization step of the EM Algorithm.
[`_n_parameters`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/GMM.md#_n_parameters) : Calculate the number of free parameters of the model.

## __init__
```python
GMM(self, init_cores=5, init='kmeans', n_init=10, max_iter=100, tol=1e-3, reg_covar=1e-6, random_state=None)
```

A Gaussian Mixture Model.

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

n_init : int, default=10
	Number of times the GMM  will be run with different
	Core seeds. The final results will be the best output of
	n_init consecutive runs in terms of inertia.

max_iter : int, default=100
	Maximum number of iterations of the GMM for a
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

## get_params
```python
GMM.get_params()
```
Return the parameters of the model.

**Returns**
```
mu : array-like, shape (n_cores, n_features)
	List of the means for all Cores in the model.

sigma : array-like, shape (n_cores, n_features, n_features)
	List of the covariances for all Cores in the model.

delta : array-like, shape (n_cores, 1)
	List of the weights for all Cores in the model.
```

## fit
```python
GMM.fit(data)
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
self : GMM
	Itself, now updated with fitted parameters.
```

## predict
```python
GMM.predict(data)
```
Predict the labels for `data` using the model.

**Parameters**
```
data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.
```

**Returns**
```
labels : array, shape (n_samples,)
	Component labels.
```

## score
```python
GMM.score(data)
```
Compute the per-sample average log-likelihood.

**Parameters**
```
data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.
```

**Returns**
```
log_likelihood : float
	Log likelihood of the model.
```

## bic
```python
GMM.bic(data)
```
Bayesian Information Criterion for the current model
on the input `data`.

**Parameters**
```
data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.
```

**Returns**
```
bic : float
	Bayesian Information Criterion. The lower the better.
```

## aic
```python
GMM.aic(data)
```
Akaike Information Criterion for the current model
on the input `data`.

**Parameters**
```
data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.
```

**Returns**
```
aic : float
	Akaike Information Criterion. The lower the better.
```

## abic
```python
GMM.abic(data, bic_weight=0.5)
```
Weighted composite of the AIC and BIC for
the current model on the input `data`.

**Parameters**
```
data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.

bic_weight : float, default=0.5
	A float within [0., 1.] that determines the
	weighting of BIC scores to AIC scores in
	calculating the ABIC composite.
```

**Returns**
```
abic : float
	Weighted average of Akaike Information Criterion
	and Bayesian Information Criterion.
	The lower the better.
```

## _validate_data
```python
GMM._validate_data(data)
```
Validate and format the given `data`.
Ensure the data matches the model's dimensionality.
If the model has yet to set a dimensionality, set it
to match the dimensionality of `data`.

**Parameters**
```
data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.
```

**Returns**
```
formatted_data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.
```

## _fit_single
```python
GMM._fit_single(data)
```
A single attempt to estimate model parameters
with the EM algorithm.

The method iterates between E-step and M-step for
`max_iter` times until the change of likelihood or lower bound
is less than `tol`.

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

## _estimate_parameters
```python
GMM._estimate_parameters(data, resp)
```
Estimate model parameters.

**Parameters**
```
data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.

resp : array-like, shape (n_samples, n_cores)
	The normalized probabilities for each data sample in `data`.
```

**Returns**
```
params : dict
	Estimated parameters for the model.
```

## _initialize
```python
GMM._initialize(data)
```
Initialize a set of Cores within the data space.

**Parameters**
```
data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.
```

**Returns**
```
cores : array, shape (n_cores,)
	List of Cores within the data space given by `data`.
```

## _initialize_core
```python
GMM._initialize_core(mu=None, sigma=None, delta=None)
```
Initialize a Core within the data space.

**Parameters**
```
mu : array-like, shape (n_features,), default=None
	Mean of the Core.

sigma : array-like, shape (n_features, n_features), default=None
	Covariance of the Core.

delta : array-like, shape (n_features,), default=None
	Weight of the Core.
```

**Returns**
```
core : Core
	A Core within the data space given by `data`.
```

## _expectation
```python
GMM._expectation(data)
```
Conduct the expectation step.

**Parameters**
```
data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.
```

**Returns**
```
p : array, shape (n_cores, n_samples)
	Probabilities of samples under each Core.

p_norm : array, shape (n_samples,)
	Total probabilities of each sample.

resp : array-like, shape (n_cores, n_samples)
	The normalized probabilities for each data sample in `data`.
```

## _maximization
```python
GMM._maximization(data, resp)
```
Conduct the maximization step.

**Parameters**
```
data : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.

resp : array-like, shape (n_cores, n_samples)
	The normalized probabilities for each data sample in `data`.
```

## _n_parameters
```python
GMM._n_parameters()
```
Return the number of free parameters in the model.

**Returns**
```
n_parameters : int
	The number of free parameters in the model.
```
