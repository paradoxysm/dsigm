"""Self-stabilizing Gaussian Mixture Model"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
import warnings

from ._utils import format_array, create_random_state
from ._exceptions import ConvergenceWarning

class SGMM:
	"""
	A modified Gaussian Mixture Model that stabilizes
	the optimal number of components during fitting.

	SGMM refines the number of components during each
	iteration of the EM algorithm with a modified
	Bayesian Information Criterion.

	Parameters
	----------
	init_cores : int, default=10
		The initial number of Cores (Gaussian components) to fit the data.

	n_init : int, default=10
		Number of times the SGMM  will be run with different
        Core seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

	max_iter : int, default=200
		Maximum number of iterations of the SGMM for a
        single run.

	tol : float, default=1e-4
		Relative tolerance with regards to the difference in inertia
		of two consecutive iterations to declare convergence.

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
	"""
	def __init__(self, init_cores=10, stabilize=True, n_init=10, max_iter=200,
					tol=1e-4, random_state=None):
		self.dim = -1
		self.init_cores = n_cores
		self.stabilize = stabilize
		self.n_init = n_init
		self.max_iter = max_iter
		self.tol = tol
		self.random_state = create_random_state(seed=random_state)
		self.inertia = -np.inf
		self.cores = []

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
			cores = self._initialize(data)
			inertia, bic = -np.inf, np.inf
			for iter in range(1, self.max_iter + 1):
				prev_inertia = inertia
				p = self._expectation(data)
				self._maximization(data, p)
				inertia = self.score(p)
				if np.abs(inertia - prev_inertia) < self.tol:
					self.converged = True
					break
				prev_bic, bic = bic, self.bic(data)
				self._stabilize(bic, prev_bic)
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
		data_range = np.asarray([np.min(data, axis=0), np.max(data, axis=0)])
		cores = []
		for n in range(self.init_cores):
			mu = self.random_state.rand(self.dim) * (data_range[1] - data_range[0]) + data_range[0]
			sigma = make_spd_matrix(self.dim)
			delta = np.ones((1)) / self.init_cores
			cores_fast.append(Core(mu=mu, sigma=sigma, delta=delta))
		cores = np.asarray(cores)
		return cores

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
		p = []
		for core in self.cores:
			m, s = core.mu, core.sigma
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
		for p, c in zip(range(len(prob)), range(len(self.cores))):
			b_num = prob[p] * self.cores[c].delta
			b_denom = np.sum([prob[i] * self.cores[i].delta for i in range(len(self.cores))], axis=0) + 1e-8
			b = b_num / b_denom
			self.cores[c].mu = np.sum(b.reshape(len(data), 1) * data, axis=0) / np.sum(b + 1e-8)
			self.cores[c].sigma = np.dot((b.reshape(len(data), 1) * (data - self.cores[c].mu)).T,
										(data - self.cores[c].mu)) / np.sum(b + 1e-8)
			self.cores[c].delta = np.mean(b)

	def _stabilize(self, bic, prev_bic):
		"""
		Estimate the ideal number of Cores at the current step.

		Parameters
		----------
		bic : float
			Bayesian Information Criterion of the current step.

		prev_bic : float
			Bayesian Information Criterion of the previous step.
		"""
		# Account for prev_bic = np.inf
		# Amplify the strength so that it gets closer to minimum
		# Decrease learning rate as it approaches minimum
		pass

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
		return np.mean(np.sum(np.log(p), axis=0))

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

	def _n_parameters(self):
		"""
		Return the number of free parameters in the model.

		Returns
		-------
		n_parameters : int
			The number of free parameters in the model.
		"""
		sigma_params = self.n_cores * self.dim * (self.dim + 1) / 2
		mu_params = self.dim * self.n_cores
		delta_params = self.n_cores
		return int(sigma_params + mu_params + delta_params - 1)
