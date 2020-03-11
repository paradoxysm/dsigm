"""Self-stabilizing Gaussian Mixture Model"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
import warnings
from sklearn.datasets import make_spd_matrix
from sklearn.cluster import KMeans

from ._utils import format_array, create_random_state
from ._exceptions import ConvergenceWarning, InitializationWarning
from . import Core

class SGMM:
	"""
	A modified Gaussian Mixture Model that stabilizes
	the optimal number of components during fitting.

	SGMM refines the number of components during each
	iteration of the EM algorithm with a modified
	Bayesian Information Criterion.

	Parameters
	----------
	init_cores : int, default=5
		The initial number of Cores (Gaussian components) to fit the data.

	init : {'random', 'kmeans'}, default='kmeans'
		The method used to initialize the weights, the means and the
        precisions.
        Must be one of::
            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

	stabilize : array-like or None, shape (2,2),
					default=((2.5, -10), (2.5, 10))
		Rates and thresholds for stabilization of the number of Cores.
		If None, stabilization is disabled.

	n_init : int, default=10
		Number of times the SGMM  will be run with different
        Core seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

	max_iter : int, default=200
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

	Attributes
	----------
	dim : int
		The dimensionality of the model; the number of features the model
		expects.

	inertia : float
		Average of maximal probabilities of each sample to each Core.

	converged : bool
		True when convergence was reached in fit(), False otherwise.

	cores : array-like, shape (n_cores,)
		A list of Cores.

	_data_range : array-like, shape (2, n_features)
		The range that encompasses the data in each axis.
	"""
	def __init__(self, init_cores=5, init='kmeans',
					stabilize=((1.6, -10), (2.5, 10)), n_init=10, max_iter=200,
					tol=1e-3, reg_covar=1e-6, random_state=None):
		self.dim = -1
		self.init_cores = init_cores
		self.init = init
		self.stabilize = True if stabilize is not None else None
		self.growth_rate = stabilize[0][0] if stabilize is not None else None
		self.growth_threshold = stabilize[0][1]  if stabilize is not None else None
		self.decay_rate = stabilize[1][0] if stabilize is not None else None
		self.decay_threshold = stabilize[1][1] if stabilize is not None else None
		self.n_init = n_init
		self.max_iter = max_iter
		self.tol = tol
		self.reg_covar = reg_covar
		self.random_state = create_random_state(seed=random_state)
		self.inertia = -np.inf
		self.cores = []
		self._data_range = None
		self.converged = False

	def get_params(self):
		"""
		Return the parameters of the model.

		Returns
		-------
		mu : array-like, shape (n_cores, n_features)
			List of the means for all Cores in the model.

		sigma : array-like, shape (n_cores, n_features, n_features)
			List of the covariances for all Cores in the model.

		delta : array-like, shape (n_cores, 1)
			List of the weights for all Cores in the model.
		"""
		mu, sigma, delta = [], [], []
		for c in self.cores:
			mu.append(c.mu)
			sigma.append(c.sigma)
			delta.append(c.delta)
		return np.asarray(mu), np.asarray(sigma), np.asarray(delta)

	def fit(self, data):
		"""
		Estimate model parameters with the EM algorithm.

		The method fits the model `n_init` times and sets
		the parameters with which the model has the
		largest likelihood or lower bound. Within each trial,
		the method iterates between E-step and M-step for
		`max_iter` times until the change of likelihood or lower bound
		is less than `tol`, otherwise, a ConvergenceWarning is raised.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.
		"""
		data = self._validate_data(data)
		best_inertia, best_cores = self.inertia, self.cores
		for init in range(1, self.n_init + 1):
			inertia, cores = self._fit_single(data)
			if inertia > best_inertia:
				best_inertia, best_cores = inertia, cores
		if not self.converged:
			warnings.warn('Initialization %d did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.'
                          % (init), ConvergenceWarning)
		self.cores, self.inertia = best_cores, best_inertia
		return self

	def predict(self, data):
		"""
		Predict the labels for `data` using the model.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		Returns
		-------
		labels : array, shape (n_samples,)
			Component labels.
		"""
		if len(self.cores) == 0:
			warnings.warn('Model not initialized so prediction ignored.', InitializationWarning)
		else:
			data = self._validate_data(data)
			estimates = np.asarray(self._expectation(data)).T
			return estimates.argmax(axis=-1)

	def _validate_data(self, data):
		"""
		Validate and format the given `data`.
		Ensure the data matches the model's dimensionality.
		If the model has yet to set a dimensionality, set it
		to match the dimensionality of `data`.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		Returns
		-------
		formatted_data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.
		"""
		data = format_array(data)
		if self.dim == -1:
			self.dim = data.shape[-1]
		elif self.dim != data.shape[-1]:
			raise ValueError("Mismatch in dimensions between model and input data.")
		return data

	def _fit_single(self, data):
		"""
		A single attempt to estimate model parameters
		with the EM algorithm.

		The method iterates between E-step and M-step for
		`max_iter` times until the change of likelihood or lower bound
		is less than `tol`.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		Returns
		-------
		inertia : float
			Log likelihood of the model.

		cores : array-like, shape (n_cores,)
			A list of Cores for this fit trial.
		"""
		self._initialize(data)
		self.inertia = -np.inf
		for iter in range(1, self.max_iter + 1):
			p, p_norm, resp = self._expectation(data)
			self._maximization(data, resp)
			prev_inertia, self.inertia = self.inertia, self.score(p_norm)
			if np.abs(self.inertia - prev_inertia) < self.tol:
				change = False
				if self.stabilize:
					change = self._stabilize(data)
				if change:
					continue
				else:
					self.converged = True
					break
		return self.inertia, self.cores

	def _initialize(self, data):
		"""
		Initialize a set of Cores within the data space.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		Returns
		-------
		cores : array, shape (n_cores,)
			List of Cores within the data space given by `data`.
		"""
		data = self._validate_data(data)
		self._data_range = np.asarray([np.min(data, axis=0), np.max(data, axis=0)])
		cores = []
		params = None
		if self.init == 'kmeans':
			resp = np.zeros((len(data), self.init_cores))
			label = KMeans(n_clusters=self.init_cores, n_init=1,
							random_state=self.random_state).fit(data).labels_
			resp[np.arange(len(data)), label] = 1
			params = self._estimate_parameters(data, resp)
		elif self.init == 'random':
			none = np.full((self.init_cores,), None)
			params = {'mu':none, 'sigma':none, 'delta':none}
		else:
			raise ValueError("Unimplemented initialization method '%s'"
                             % self.init)
		for n in range(self.init_cores):
			mu, sigma, delta = params['mu'][n], params['sigma'][n], params['delta'][n]
			core = self._initialize_core(mu=mu, sigma=sigma, delta=delta)
			cores.append(core)
		self.cores = np.asarray(cores)
		return self.cores

	def _estimate_parameters(self, data, resp):
		"""
		Initialize model parameters.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		resp : array-like, shape (n_samples, n_cores)
			The unweighted probabilities for each data sample in `data`.

		Returns
		-------
		params : dict
			Estimated parameters for the model.
		"""
		delta = np.sum(resp, axis=0) + 10 * np.finfo(resp.dtype).eps
		mu = np.dot(resp.T, data) / delta[:,np.newaxis]
		sigma = np.empty((self.init_cores, self.dim, self.dim))
		for k in range(self.init_cores):
			diff = data - mu[k]
			sigma[k] = np.dot(resp[:, k] * diff.T, diff) / delta[k]
			sigma[k].flat[::self.dim + 1] += self.reg_covar
		delta = (delta / len(data))[:,np.newaxis]
		return {'mu':mu, 'sigma':sigma, 'delta':delta}

	def _initialize_core(self, mu=None, sigma=None, delta=None):
		"""
		Initialize a Core within the data space.

		Parameters
		----------
		mu : array-like, shape (n_features,), default=None
			Mean of the Core.

		sigma : array-like, shape (n_features, n_features), default=None
			Covariance of the Core.

		delta : array-like, shape (n_features,), default=None
			Weight of the Core.

		Returns
		-------
		core : Core
			A Core within the data space given by `data`.
		"""
		if mu is not None and sigma is not None and delta is not None:
			return Core(mu=mu, sigma=sigma, delta=delta)
		elif self._data_range is not None:
			mu = self.random_state.rand(self.dim) * \
					(self._data_range[1] - self._data_range[0]) + \
					self._data_range[0]
			sigma = make_spd_matrix(self.dim)
			if len(self.cores):
				delta = np.ones((1)) / len(self.cores)
			else:
				delta = np.ones((1)) / self.init_cores
			return Core(mu=mu, sigma=sigma, delta=delta)
		else:
			raise RuntimeError("Data Range hasn't been set, likely because SGMM hasn't been initialized yet")

	def _expectation(self, data):
		"""
		Conduct the expectation step.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		Returns
		-------
		b : array, shape (n_cores, n_samples)
			Probabilities of samples under each Core.

		p_norm : array, shape (n_samples,)
			Probabilities of each sample.

		resp : array-like, shape (n_cores, n_samples)
			The unweighted probabilities for each data sample in `data`.
		"""
		data = self._validate_data(data)
		p_unweighted = []
		for core in self.cores:
			p_unweighted.append(core.pdf(data))
		p_unweighted = np.asarray(p_unweighted)
		if p_unweighted.shape != (len(self.cores), len(data)):
			raise RuntimeError("Expectation Step found erroneous shape")
		delta_cores = [self.cores[i].delta for i in range(len(self.cores))]
		p = p_unweighted * delta_cores
		p_norm = np.sum(p, axis=0)
		resp = p / (p_norm + 1e-8)
		return p, p_norm, resp

	def _maximization(self, data, resp):
		"""
		Conduct the maximization step.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		resp : array-like, shape (n_cores, n_samples)
			The unweighted probabilities for each data sample in `data`.
		"""
		data = self._validate_data(data)
		params = self._estimate_parameters(data, resp.T)
		for i in range(len(self.cores)):
			mu = params['mu'][i]
			sigma = params['sigma'][i]
			delta = params['delta'][i]
			self.cores[i] = Core(mu=mu, sigma=sigma, delta=delta)

	def _stabilize(self, data):
		"""
		Estimate the ideal number of Cores at the current step.
		Change the cores in the model to fit this estimate.

		New cores are seeded randomly within the data space.
		Cores are removed based on their probability and overlap with
		other Cores.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		p : array, shape (n_cores, n_samples)
			Probabilities of samples under each Core.
		"""
		p = self._expectation(data)
		coverage = []
		for i in range(len(p)):
			c = np.sum((p[i] - np.sum(np.delete(p, i, axis=0), axis=0)).clip(min=0))
			coverage.append(c)
		coverage = np.asarray(coverage)
		cumulate = np.sum(p, axis=1)
		growth = np.log(cumulate)*self.growth_rate - \
					np.log(len(data) / len(self.cores))
		fitness = self.decay_rate / (coverage + 1e-8) - growth
		print(np.median(fitness))
		change = False
		return change

	def score(self, p_norm):
		"""
		Compute the per-sample average log-likelihood.

		Parameters
		----------
		p_norm : array-like, shape (n_samples,)
			Probabilities of samples.

		Returns
		-------
		log_likelihood : float
			Log likelihood of the model.
		"""
		return np.mean(np.log(p_norm+1e-8))

	def bic(self, data):
		"""
		Bayesian Information Criterion for the current model
		on the input `data`.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		Returns
		-------
		bic : float
			Bayesian Information Criterion. The lower the better.
		"""
		p, p_norm, resp = self._expectation(data)
		fit = -2 * self.score(p_norm) * len(data)
		penalty = self._n_parameters() * np.log(len(data))
		return fit + penalty

	def aic(self, data):
		"""
		Akaike Information Criterion for the current model
		on the input `data`.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
				List of `n_features`-dimensional data points.
				Each row corresponds to a single data point.

		-------
		aic : float
				Akaike Information Criterion. The lower the better.
		"""
		p, p_norm, resp = self._expectation(data)
		fit = -2 * self.score(p_norm) * len(data)
		penalty = 2 * self._n_parameters()
		return fit + penalty

	def _n_parameters(self):
		"""
		Return the number of free parameters in the model.

		Returns
		-------
		n_parameters : int
			The number of free parameters in the model.
		"""
		sigma_params = len(self.cores) * self.dim * (self.dim + 1) / 2
		mu_params = self.dim * len(self.cores)
		delta_params = len(self.cores)
		return int(sigma_params + mu_params + delta_params - 1)
