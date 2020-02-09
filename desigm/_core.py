"""Core and Cluster"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

from scipy.stats import multivariate_normal as mvn

class Core:
	"""
	A Core Point that defines a Gaussian Distribution.

	Parameters
	----------
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
	"""
	def __init__(self, mu=[0], sigma=[1], delta=[1], cluster=None):
		self.mu = mu
		self.sigma = sigma
		self.delta = delta
		self.cluster = cluster

	def pdf(self, data):
		"""
		Multivariate normal probability density function.

		Parameters
        ----------
        data : array_like
            Quantiles, with the last axis of `data` denoting the features.

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `datas`
		"""
		return mvn.pdf(x=data, mean=self.mu, cov=self.sigma)


class CoreCluster:
	"""
	A CoreCluster defines a collection of Cores that belong to a cluster.

	Parameters
	----------
	cores : array-like, shape (some_cores,), default=[]
		A list of Cores associated with this CoreCluster.

	parent : CoreCluster, default=None
		The parent CoreCluster in a hierarchical manner.

	children : array-like, shape (some_cores,), default=[]
		A list of CoreCluster children in a hierarchical manner.
	"""
	def __init__(self, cores=[], parent=None, children=[]):
		self.cores = []
		self.parent = parent
		self.children = []
