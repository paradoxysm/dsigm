"""DESIGM Clustering Algorithm"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

import ._utils as utils

class DESIGM:
	"""
	A Clustering Model using the DESIGM Clustering Algorithm.

	DESIGM - Density-sensitive Evolution-based
	Self-stabilization of Independent Gaussian Mixtures.
	Fits a self-stabilized number of Gaussian components
	and groups them into clusters in a density sensitive manner.

	Parameters
	----------
	n_clusters : int or None, default=None
		Number of CoreClusters to be fitted to the data.
		When `n_clusters` is None, determine best fit of CoreClusters.

	n_cores : int, default=10
		Number of Cores (Gaussian components) to fit the data.
		The initial number is the number of Cores at initialization.
		Subsequently, it tracks the actual number of Cores.

	n_init : int, default=10
		Number of time the DESIGM algorithm will be run with different
        Core seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

	max_iter : int, default=200
		Maximum number of iterations of the DESIGM algorithm for a
        single run.

	ds : bool, default=True
		DESIGM algorithm groups Cores in a density sensitive manner,
		akin to the OPTICS algorithm.

	tol : float, default=1e-4
		Relative tolerance with regards to the difference in inertia
		of two consecutive iterations to declare convergence.

	random_state : None or int or RandomState, default=None
		Determines random number generation for Core initialization. Use
        an int to make the randomness deterministic.

	Attributes
	----------
	inertia : float
		Average of maximal probabilities of each sample to each Core.

	cores : array-like, shape (n_cores,)
		A list of Cores.

	clusters : CoreCluster or None
		A graph of CoreClusters.
	"""
	def __init__(self, n_clusters=None, n_cores=10, n_init=10, max_iter=200, ds=True,
					tol=1e-4, random_state=None):
		self.sgmm = SGMM(n_cores=n_cores, n_init=n_init, max_iter=max_iter,
							tol=tol, random_state=random_state)

	def fit(data, weights=None):
		"""
		Fit the model to `data`.
		"""
		pass

	def predict(data, weights=None):
		"""
		Predict the clusters `data` belongs to.
		"""
		pass
