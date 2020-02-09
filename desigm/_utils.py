"""Utilities for Various Tasks"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

def format_arrays(*args):
	"""
	Format all arrays given into numpy ndarrays
	such that the last axis denotes the features.

	Parameters
	----------
	*args : tuple, default=()
		Tuple of array-like objects to convert into ndarrays.

	Returns
	-------
	arrays : list
		List of formatted ndarrays where the last axis denotes the features.
	"""
	pass

def check_dimensionality(*args):
	"""
	Check that all arguments have the same dimensionality.
	Return that dimensionality.

	Parameters
	----------
	*args : tuple, default=()
		Tuple of array-like objects or scalars to evaluate.

	Returns
	-------
	dim : int
		The dimensionality of all given arguments.
	"""
	pass

def create_random_state(seed=None):
	"""
	Create a RandomState.

	Parameters
	----------
	seed : None or int or RandomState
		Initial seed for the RandomState. If seed is None,
		return the RandomState singleton. If seed is an int,
		return a RandomState with the seed set to the int.
		If seed is a RandomState, return that RandomState.

	Returns
	-------
	random_state : RandomState
		A RandomState object.
	"""
	if seed is None:
		return np.random.mtrand._rand
	elif isinstance(seed, (int, np.integer)):
		return np.random.RandomState(seed=seed)
	elif isinstance(seed, np.random.RandomState):
		return seed
	else:
		raise ValueError("Seed must be None, an int, or a Random State")
