"""Utilities for Various Tasks"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

def format_array(arr):
	"""
	Format `arr` into an ndarray where each row
	corresponds to a single data point with a
	1D list of features.
	All data is formatted as a float. If `arr` cannot
	be formatted in this manner, raise a ValueError.

	Parameters
	----------
	arr : array-like, shape (n_samples, n_features)
		List of data points.

	Returns
	-------
	arr : array-like, shape (n_samples, n_features)
		List of `n_features`-dimensional data points.
		Each row corresponds to a single data point.
	"""
	arr = np.asarray(arr)
	try:
		arr = arr.astype(float)
	except Exception as e:
		raise
	if arr.ndim == 1:
		arr = arr[:,np.newaxis]
	elif arr.ndim > 2:
		arr = np.squeeze(arr)
		if arr.size == 1:
			arr = np.asarray([arr])
		arr = format_array(arr)
	return arr

def create_random_state(seed=None):
	"""
	Create a RandomState.

	Parameters
	----------
	seed : None or int or RandomState, default=None
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
