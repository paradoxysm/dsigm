"""Self-stabilizing Gaussian Mixture Model"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
import warnings

from .._exceptions import ConvergenceWarning, StabilizationWarning
from . import GMM

class SGMM(GMM):
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
		super().__init__(init_cores=5, init='kmeans',n_init=10, max_iter=100,
						tol=1e-3, reg_covar=1e-6, random_state=None)
		self.stabilize = stabilize

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
		n_init = self.n_init // 2
		interval, abic = self._orient_stabilizer(data, init_cores)
		while interval[1] - interval[0] > 1:
			midpoint = (interval[0] + interval[1]) // 2
			abic_m = GMM(n_init=n_init, init_cores=midpoint).fit(data)._abic(data)
			if (abic[0] > abic_m and abic_m >= abic[1]) or \
					(abic[0] >= abic_m and abic_m > abic[1]):
				interval, abic = (midpoint, interval[1]), (abic_m, abic[1])
			elif (abic[1] > abic_m and abic_m > abic[0]) or \
					(abic[1] >= abic_m and abic_m > abic[1]):
				interval, abic = (interval[0], midpoint), (abic[0], abic_m)
			elif abic_m <= abic[0] and abic_m <= abic[1]:
				interval, abic = self._halve_interval(data, interval, abic,
									midpoint, abic_m, n_init)
			else:
				interval, abic = self._truncate_interval(data, interval, abic, n_init)
		best = 0 if abic[0] < abic[1] else 1
		self.fit(data, stabilize=False, init_cores=interval[best])
		return self.inertia, self.cores

	def _truncate_interval(self, data, interval, abic, n_init):
		"""
		Truncate the interval by reducing the upper or lower limit
		based on which has the higher BIC/AIC (ABIC).

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		interval : tuple, shape (2,)
			The interval which contains the optimal number of Cores.
			Interpreted as [min, max).

		abic : tuple, shape (2,)
			The abic scores corresponding to the interval.

		n_init : int, default=10
			Number of times the SGMM  will be run with different
	        Core seeds. The final results will be the best output of
	        n_init consecutive runs in terms of inertia.

		Returns
		-------
		interval : tuple, shape (2,)
			The interval which contains the optimal number of Cores.
			Interpreted as [min, max).

		abic : tuple, shape (2,)
			The abic scores corresponding to the interval.
		"""
		if abic[1] >= abic[0]:
			interval_l = interval[1] - 1
			abic_l = GMM(n_init=n_init, init_cores=interval_l).fit(data)._abic(data)
			interval, abic = (interval[0], interval_l), (abic[0], abic_l)
		else:
			interval_l = interval[1] + 1
			abic_l = GMM(n_init=n_init, init_cores=interval_l).fit(data)._abic(data)
			interval, abic = (interval_l, interval[1]), (abic_l, abic[1])
		return interval, abic

	def _halve_interval(self, data, interval, abic, midpoint, abic_m, n_init):
		"""
		Halve the interval based on the BIC/AIC (ABIC) of the midpoint.

		Parameters
		----------
		data : array-like, shape (n_samples, n_features)
			List of `n_features`-dimensional data points.
			Each row corresponds to a single data point.

		interval : tuple, shape (2,)
			The interval which contains the optimal number of Cores.
			Interpreted as [min, max).

		abic : tuple, shape (2,)
			The abic scores corresponding to the interval.

		midpoint : int
			The midpoint of the interval.

		abic_m : float
			The abic score corresponding to the midpoint.

		n_init : int, default=10
			Number of times the SGMM  will be run with different
	        Core seeds. The final results will be the best output of
	        n_init consecutive runs in terms of inertia.

		Returns
		-------
		interval : tuple, shape (2,)
			The interval which contains the optimal number of Cores.
			Interpreted as [min, max).

		abic : tuple, shape (2,)
			The abic scores corresponding to the interval.
		"""
		m0 = (interval[0] + midpoint) // 2
		if m0 == interval[0]:
			return (midpoint, interval[1]), (abic_m, abic[1])
		else:
			abic_m0 = GMM(n_init=n_init, init_cores=m0).fit(data)._abic(data)
			if abic_m0 < abic_m:
				return (interval[0], midpoint), (abic[0], abic_m)
			m1 = (interval[1] + midpoint) // 2
			abic_m1 = GMM(n_init=n_init, init_cores=m1).fit(data)._abic(data)
			if abic_m1 < abic_m:
				return (midpoint, interval[1]), (abic_m, abic[1])
			else:
				return (m0, m1), (abic_m0, abic_m1)

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

		abic : tuple, shape (2,)
			The abic scores corresponding to the interval.
		"""
		interval = (1, np.inf)
		abic = (np.inf, np.inf)
		ceiling = len(np.unique(data, axis=0))
		n_init = self.n_init // 2
		i, j = init_cores, init_cores + 1
		if j > ceiling:
			i, j = ceiling - 1, ceiling
		abic_i = GMM(n_init=n_init, init_cores=i).fit(data)._abic(data)
		abic_j = GMM(n_init=n_init, init_cores=j).fit(data)._abic(data)
		if abic_j - abic_i >= 0:
			abic_1 = GMM(n_init=n_init, init_cores=1).fit(data)._abic(data)
			interval, abic = (1, j), (abic_1, abic_j)
		else:
			min, abic_min = j, abic_j
			abic_threshold = [abic_i]
			while abic_j - np.mean(abic_threshold) < 0:
				abic_threshold.append(abic_j)
				inc = int(np.abs(abic_j - abic_i) / (10 * np.log(len(data)))) + 1
				if j + inc > ceiling:
					j = ceiling
					break
				else:
					j += inc
					abic_j = GMM(n_init=n_init, init_cores=j).fit(data)._abic(data)
			interval, abic = (min, j), (abic_min, abic_j)
		return interval, abic
