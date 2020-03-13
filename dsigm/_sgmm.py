"""Self-stabilizing Gaussian Mixture Model"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
import warnings
from sklearn.datasets import make_spd_matrix
from sklearn.cluster import KMeans

from ._utils import format_array, create_random_state
from ._exceptions import ConvergenceWarning, InitializationWarning, \
							StabilizationWarning
from . import Core

class SGMM:
	"""
	A modified Gaussian Mixture Model that can stabilize
	the optimal number of components during fitting.

	SGMM refines the number of components during each
	iteration of the EM algorithm using a narrowing
	interval based on the Bayesian Information Criterion.

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

	stabilize : True or False, default=True
		Enable stabilization of the number of Cores.

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
					stabilize=True, n_init=10, max_iter=100,
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

	def fit(self, data, stabilize=None, init_cores=None):
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

		stabilize : True, False, or None, default=None
			Enable stabilization of the number of Cores.
			If None, determine based on model.

		init_cores : int, default=None
			The initial number of Cores (Gaussian components)
			to fit the data.

		Returns
		-------
		self : SGMM
			Itself, now updated with fitted parameters.
		"""
		data = self._validate_data(data)
		best_inertia, best_cores = self.inertia, self.cores
		stabilize = self.stabilize if stabilize is None else stabilize
		init_cores = self.init_cores if init_cores is None else init_cores
		for init in range(1, self.n_init + 1):
			if stabilize:
				inertia, cores = self._fit_stabilize(data, init_cores)
			else:
				inertia, cores = self._fit_single(data, init_cores)
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
			warnings.warn('Model not initialized so prediction ignored.',
							InitializationWarning)
		else:
			data = self._validate_data(data)
			p, p_norm, resp = self._expectation(data)
			estimates = np.asarray(p).T
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

	def _fit_single(self, data, init_cores):
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

		init_cores : int, default=5
			The initial number of Cores (Gaussian components)
			to fit the data.

		Returns
		-------
		inertia : float
			Log likelihood of the model.

		cores : array-like, shape (n_cores,)
			A list of Cores for this fit trial.
		"""
		self._initialize(data, init_cores)
		self.inertia = -np.inf
		for iter in range(1, self.max_iter + 1):
			p, p_norm, resp = self._expectation(data)
			self._maximization(data, resp)
			prev_inertia, self.inertia = self.inertia, self.score(p_norm)
			if np.abs(self.inertia - prev_inertia) < self.tol:
				self.converged = True
				break
		return self.inertia, self.cores

	def _fit_stabilize(self, data, init_cores):
		"""
		A single attempt to estimate model parameters
		with the EM algorithm.

		The method repeatedly converges for various n_cores
		to pinpoint optimal n_cores. It does so by determining
		a search interval that contains the optimal n_cores and
		repeatedly narrows the interval until the optimal n_cores
		is determined.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		init_cores : int, default=5
			The initial number of Cores (Gaussian components)
			to fit the data.

		Returns
		-------
		inertia : float
			Log likelihood of the model.

		cores : array-like, shape (n_cores,)
			A list of Cores for this fit trial.
		"""
		interval, bic = self._orient_stabilizer(data, init_cores)
		while interval[1] - interval[0] > 1:
			midpoint = (interval[0] + interval[1]) // 2
			bic_m = SGMM(stabilize=False, init_cores=midpoint).fit(data).bic(data)
			if bic[0] > bic_m and bic_m > bic[1]:
				interval, bic = (midpoint, interval[1]), (bic_m, bic[1])
			elif bic[1] > bic_m and bic_m > bic[0]:
				interval, bic = (interval[0], midpoint), (bic[0], bic_m)
			elif bic_m <= bic[0] and bic_m <= bic[1]:
				m0 = (interval[0] + midpoint) // 2
				if m0 == interval[0]:
					interval = (m0, interval[1])
					break
				else:
					bic_m0 = SGMM(stabilize=False, init_cores=m0).fit(data).bic(data)
					if bic_m0 < bic_m:
						interval, bic = (interval[0], midpoint), (bic[0], bic_m)
					else:
						interval, bic = (midpoint, interval[1]), (bic_m, bic[1])
			else:
				min = 0 if bic[0] <= bic[1] else 1
				warnings.warn("Stabilization encountered local maxima",
								StabilizationWarning)
				self.fit(data, stabilize=False, init_cores=interval[min])
				return self.inertia, self.cores
		best = 0 if bic[0] <= bic[1] else 1
		self.fit(data, stabilize=False, init_cores=interval[best])
		return self.inertia, self.cores

	def _orient_stabilizer(self, data, init_cores):
		"""
		Orient the initial interval for the stabilizer.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		init_cores : int, default=5
			The initial number of Cores (Gaussian components)
			to fit the data.

		Returns
		-------
		interval : tuple, shape (2,)
			The interval which contains the optimal number of Cores.
			Interpreted as [min, max).

		bic : tuple, shape (2,)
			The bic scores corresponding to the interval.
		"""
		interval = (1, np.inf)
		bic = (np.inf, np.inf)
		ceiling = len(np.unique(data, axis=0))
		i, j = init_cores, init_cores + 1
		if j > ceiling:
			i, j = ceiling - 1, ceiling
		bic_i = SGMM(stabilize=False, init_cores=i).fit(data).bic(data)
		bic_j = SGMM(stabilize=False, init_cores=j).fit(data).bic(data)
		if bic_j - bic_i >= 0:
			bic_1 = SGMM(stabilize=False, init_cores=1).fit(data).bic(data)
			interval, bic = (1, j), (bic_1, bic_j)
		else:
			min, bic_min = j, bic_j
			while bic_j - bic_i < 0:
				inc = int(np.abs(bic_j - bic_i) / (10 * np.log(len(data)))) + 1
				if j + inc > ceiling:
					j = ceiling
					break
				else:
					j += inc
					bic_j = SGMM(stabilize=False, init_cores=j).fit(data).bic(data)
			interval, bic = (min, j), (bic_min, bic_j)
		return interval, bic

	def _estimate_parameters(self, data, resp, init_cores):
		"""
		Initialize model parameters.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		resp : array-like, shape (n_samples, n_cores)
			The normalized probabilities for each data sample in `data`.

		init_cores : int, default=5
			The initial number of Cores (Gaussian components)
			to fit the data.

		Returns
		-------
		params : dict
			Estimated parameters for the model.
		"""
		delta = np.sum(resp, axis=0) + 10 * np.finfo(resp.dtype).eps
		mu = np.dot(resp.T, data) / delta[:,np.newaxis]
		sigma = np.empty((init_cores, self.dim, self.dim))
		for k in range(init_cores):
			diff = data - mu[k]
			sigma[k] = np.dot(resp[:, k] * diff.T, diff) / delta[k]
			sigma[k].flat[::self.dim + 1] += self.reg_covar
		delta = (delta / len(data))[:,np.newaxis]
		return {'mu':mu, 'sigma':sigma, 'delta':delta}

	def _initialize(self, data, init_cores):
		"""
		Initialize a set of Cores within the data space.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		init_cores : int, default=5
			The initial number of Cores (Gaussian components)
			to fit the data.

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
			resp = np.zeros((len(data), init_cores))
			label = KMeans(n_clusters=init_cores, n_init=1,
							random_state=self.random_state).fit(data).labels_
			resp[np.arange(len(data)), label] = 1
			params = self._estimate_parameters(data, resp, init_cores)
		elif self.init == 'random':
			none = np.full((init_cores,), None)
			params = {'mu':none, 'sigma':none, 'delta':none}
		else:
			raise ValueError("Unimplemented initialization method '%s'"
                             % self.init)
		for n in range(init_cores):
			mu, sigma, delta = params['mu'][n], params['sigma'][n], params['delta'][n]
			core = self._initialize_core(mu=mu, sigma=sigma, delta=delta)
			cores.append(core)
		self.cores = np.asarray(cores)
		return self.cores

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

		p_norm : array, shape (n_samples,)
			Total probabilities of each sample.

		resp : array-like, shape (n_cores, n_samples)
			The normalized probabilities for each data sample in `data`.
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
			The normalized probabilities for each data sample in `data`.
		"""
		data = self._validate_data(data)
		params = self._estimate_parameters(data, resp.T, len(resp))
		for i in range(len(self.cores)):
			mu = params['mu'][i]
			sigma = params['sigma'][i]
			delta = params['delta'][i]
			self.cores[i] = Core(mu=mu, sigma=sigma, delta=delta)

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
