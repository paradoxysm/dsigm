"""Core and Cluster"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

from scipy.stats import multivariate_normal as mvn
import numpy as np

from ._utils import format_array

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
		self.dim = -1
		self.mu = np.asarray(mu)
		self.sigma = np.asarray(sigma)
		self.delta = np.asarray(delta)
		self.cluster = cluster
		self._validate_init()

	def __eq__(self, other):
		if isinstance(other, Core):
			mu = np.all(self.mu == other.mu)
			sigma = np.all(self.sigma == other.sigma)
			delta = np.all(self.delta == other.delta)
			cluster = np.all(self.cluster == self.cluster)
			return mu and sigma and delta and cluster
		return False

	def _validate_init(self):
		"""
		Validate the attributes of the Core.
		If `dim` has not yet been set, set it now.
		"""
		if self.mu.ndim != 1:
			raise ValueError("Invalid argument provided for mu. Must be a vector")
		if not (self.sigma.ndim == 1 and len(self.sigma) == 1 or \
				self.sigma.ndim == 2 and len(self.sigma) == self.sigma.shape[-1]):
			raise ValueError("Invalid argument provided for sigma. Must be a vector")
		if self.delta.ndim != 1 or self.delta.size != 1:
			raise ValueError("Invalid argument provided for delta. Must be a vector")
		if not isinstance(self.cluster, (type(None), CoreCluster)):
			raise ValueError("Invalid argument provided for cluster. Must be None or CoreCluster. Found " + type(self.cluster).__name__)
		if self.mu.shape[-1] == self.sigma.shape[-1]:
			self.dim = self.mu.shape[-1]
		else:
			raise ValueError("Mismatch in dimensions between mu and sigma")

	def pdf(self, data, weight=False):
		"""
		Multivariate normal probability density function.

		Parameters
		----------
		data : array-like
			Quantiles, with the last axis of `data` denoting the features.

		weight :  bool, default=False
			Calculate the pdf with the delta of the Core.

		Returns
		-------
		pdf : ndarray or scalar
			Probability density function evaluated at `data`.
		"""
		data = format_array(data)
		self._validate_init()
		p = mvn.pdf(x=data, mean=self.mu, cov=self.sigma)
		if weight:
			return p * self.delta
		else:
			return p

	def logpdf(self, data, weight=False):
		"""
		Log of multivariate normal probability density function.

		Parameters
		----------
		data : array-like
			Quantiles, with the last axis of `data` denoting the features.

		weight :  bool, default=False
			Calculate the logpdf with the delta of the Core.

		Returns
		-------
		pdf : ndarray or scalar
			Log probability density function evaluated at `data`.
		"""
		data = format_array(data)
		self._validate_init()
		p = mvn.logpdf(x=data, mean=self.mu, cov=self.sigma)
		if weight:
			return p + np.log(self.delta)
		else:
			return p

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
	def __init__(self, cores=[], parents=[], children=[]):
		self.cores = np.asarray(cores)
		self.parents = np.asarray(parents)
		self.children = np.asarray(children)
		self._validate_init()

	def __eq__(self, other):
		if isinstance(other, CoreCluster):
			cores = np.all(self.cores == other.cores)
			parents = np.all(self.parents == other.parents)
			children = np.all(self.children == other.children)
			return cores and parents and children
		return False

	def _validate_init(self):
		"""
		Validate the attributes of the CoreCluster.
		"""
		if self.cores.ndim != 1 or (self.cores.ndim == 0 and self.cores.size > 0):
			raise ValueError("Invalid argument provided for cores. Must be a list of Cores")
		core_types = set([type(item) for item in self.cores])
		if len(core_types) > 1 or (len(core_types) == 1 and Core not in core_types):
			raise ValueError("Invalid argument provided for cores. Must be a list of Cores")

		if self.parents.ndim != 1 or (self.parents.ndim == 0 and self.parents.size > 0):
			raise ValueError("Invalid argument provided for parents. Must be a list of CoreClusters")
		parent_types = set([type(item) for item in self.parents])
		if len(parent_types) > 1 or (len(parent_types) == 1 and CoreCluster not in parent_types):
			raise ValueError("Invalid argument provided for parents. Must be a list of CoreClusters")

		if self.children.ndim != 1 or (self.children.ndim == 0 and self.children.size > 0):
			raise ValueError("Invalid argument provided for children. Must be a list of CoreClusters")
		children_types = set([type(item) for item in self.children])
		if len(children_types) > 1 or (len(children_types) == 1 and CoreCluster not in children_types):
			raise ValueError("Invalid argument provided for children. Must be a list of CoreClusters")
