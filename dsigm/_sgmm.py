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

	stabilize : int or None, default=0.05
		The adaption rate for stabilization of the number of Cores.
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
	def __init__(self, init_cores=5, init='kmeans', stabilize=0.05, n_init=10, max_iter=200,
					tol=1e-3, reg_covar=1e-6, random_state=None):
		self.dim = -1
		self.init_cores = init_cores
		self.init = init
		self.stabilize = stabilize
		self.n_init = n_init
		self.max_iter = max_iter
		self.tol = tol
		self.reg_covar = reg_covar
		self.random_state = create_random_state(seed=random_state)
		self.inertia = -np.inf
		self.cores = []
		self._data_range = None

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
			p = self._expectation(data)
			self._maximization(data, p)
			prev_inertia, self.inertia = self.inertia, self.score(p)
			if np.abs(self.inertia - prev_inertia) < self.tol:
				self.converged = True
				break
			if self.stabilize is not None:
				self._stabilize(data)
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
		params = _estimate_parameters(data)
		for n in range(self.init_cores):
			mu, sigma, delta = params['mu'][n], params['sigma'][n], params['delta'][n]
			core = self._initialize_core(mu=mu, sigma=sigma, delta=delta)
			cores.append(core)
		self.cores = np.asarray(cores)
		return self.cores

	def _estimate_parameters(self, data):
		"""
		Initialize model parameters.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		Returns
		-------
		params : dict
			Initial parameters for the model.
		"""
		if self.init == 'kmeans':
			resp = np.zeros((len(data), self.n_cores))
			label = KMeans(n_clusters=self.init_cores, n_init=1,
							random_state=self.random_state).fit(X).labels_
			resp[np.arange(len(data)), label] = 1
			delta = np.sum(resp, axis=0) + 10 * np.finfo(resp.dtype).eps
			mu = np.dot(resp.T, data) / delta[:,np.newaxis]
			sigma = np.empty((n_components, n_features, n_features))
			for k in range(self.init_cores):
				diff = data - mu[k]
				sigma[k] = np.dot(resp[:, k] * diff.T, diff) / delta[k]
				sigma[k].flat[::self.dim + 1] += self.reg_covar
			delta /= len(data)
			return {'mu'=mu, 'sigma'=sigma, 'delta'=delta}
		elif self.init == 'random':
			return {'mu'=None, 'sigma'=None, 'delta'=delta}
		else:
			raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

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
		if mu and sigma and delta:
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
		p : array, shape (n_cores, n_samples)
			Probabilities of samples under each Core.
		"""
		data = self._validate_data(data)
		p = []
		for core in self.cores:
			p.append(core.pdf(data))
		p = np.asarray(p)
		if p.shape != (len(self.cores), len(data)):
			raise RuntimeError("Expectation Step found erroneous shape")
		return p

	def _maximization(self, data, prob):
		"""
		Conduct the maximization step.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		prob : array, shape (n_cores, n_samples)
			Probabilities of samples under each Core.
		"""
		data = self._validate_data(data)
		delta_cores = [self.cores[i].delta for i in range(len(self.cores))]
		b_vector = prob * delta_cores
		b = b_vector / (np.sum(b_vector, axis=0) + 1e-8)
		for i in range(len(self.cores)):
			mu = np.sum(b[i].reshape(len(data), 1) * data, axis=0) / np.sum(b[i] + 1e-8)
			sigma = np.dot((b[i].reshape(len(data), 1) * (data - mu)).T,
										(data - mu)) / np.sum(b[i] + 1e-8)
			np.fill_diagonal(sigma, sigma.diagonal() + self.reg_covar)
			delta = [np.mean(b[i])]
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
		gradient = self.bic_gradient(data, p)
		# DOESN'T CONVERGE WELL
		'''
			Using pre maximization p and post maximization p
			???
		'''
		step = round(gradient * self.stabilize)
		if step > 0:
			fitness = []
			for i in range(len(p)):
				f = np.sum((p[i] - np.sum(np.delete(p, i, axis=0), axis=0)).clip(min=0))
				fitness.append((f, i))
			fitness = np.asarray(sorted(fitness, key=lambda x: x[1]))
			step = step if step < len(self.cores) else 0
			mask = fitness[:int(step), 1].astype(int)
			if len(mask) > 0:
				self.cores = np.delete(self.cores, mask, axis=0)
		elif step < 0:
			for i in range(np.abs(int(step))):
				self.cores = np.concatenate((self.cores, [self._initialize_core()]))

	def score(self, p):
		"""
		Compute the per-sample average log-likelihood.

		Parameters
		----------
		p : array-like, shape (n_cores, n_samples)
			Probabilities of samples under each Core.

		Returns
		-------
		log_likelihood : float
			Log likelihood of the model.
		"""
		return np.mean(np.sum(np.log(p+1e-8), axis=0))

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
		fit = -2 * self.score(self._expectation(data)) * len(data)
		penalty = self._n_parameters() * np.log(len(data))
		return fit + penalty

	def bic_gradient(self, data, p):
		"""
		Return an approximate gradient of the Bayesian Information
		Criterion for the current model on the input `data`.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		p : array-like, shape (n_cores, n_samples)
			Probabilities of samples under each Core.

		Returns
		-------
		gradient : float
			Estimated gradient of the curve. Positive gradients
			imply excess Cores, while negative gradients imply
			lack of Cores.
		"""
		score = np.exp(self.score(p))
		fit_coeff = (len(self.cores) * (1 - score)) / (score + 1e-8)
		fit_gradient = -2 * (90 / (len(self.cores) * (len(self.cores) + fit_coeff))) * len(data)
		penalty_gradient = (np.square(self.dim) + 1.5 * self.dim + 1) * np.log(len(data))
		gradient = fit_gradient + penalty_gradient
		return gradient

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
