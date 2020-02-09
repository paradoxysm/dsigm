"""DESIGM Clustering Algorithm"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

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

	vir : float, default=5e-3
		Virulence factor for constructing new Cores. Points with probability
		relative to a Core less than `vir` will construct a Core.

	maxcons : int, default=1
		Maximum number of Cores that can be constructing in a single
		iteration of the DESIGM algorithm.

	tolmu : float, default=1e-3
		Relative tolerance with regards to the distance of two Cores
		to determine if a Core should be deconstructed.

	tolsig : float, default=5
		Relative tolerance with regards to the angles between the
		covariances of two Cores to determine if a Core should be deconstructed.

	maxdec : int, default=3
		Maximum number of Cores that can be deconstructed in a single
		iteration of the DESIGM algorithm.

	random_state : None or int or RandomState, default=None
		Determines random number generation for Core initialization. Use
        an int to make the randomness deterministic.

	Attributes
	----------
	inertia : float
		Average of maximal probabilities of each sample to each Core.

	cores_full : KDTree
		A KDTree representation of Cores.

	cores : array-like, shape (n_cores,)
		A list of Cores.

	clusters : CoreCluster or None
		A graph of CoreClusters.
	"""
	def __init__(self, n_clusters=None, n_cores=10, n_init=10, max_iter=200, ds=True,
					tol=1e-4, vir=5e-3, maxcons=1, tolmu=1e-3, tolsig=5, maxdec=3,
					random_state=None):
		self.dim = None
		self.n_clusters = 0 if n_clusters is None else n_clusters
		self.n_cores = n_cores
		self.n_init = n_init
		self.max_iter = max_iter
		self.ds = ds
		self.tol = tol
		self.vir = vir
		self.maxcons = maxcons
		self.tolmu = tolmu
		self.tolsig = tolsig
		self.maxdec = maxdec
		self.random_state = createRandomState(seed=random_state)
		self.inertia = -np.inf
		self.cores_full = KDTree()
		self.cores = []
		self.clusters = None
