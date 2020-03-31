# dsigm._utils

A collection of utility functions.

**functions**
- [`format_array`](https://github.com/paradoxysm/dsigm/blob/master/doc/pydoc/utils.md#format_array) : Format an array into an acceptable format.
- [`create_random_state`](https://github.com/paradoxysm/dsigm/blob/master/doc/pydoc/utils.md#create_random_state) : Create a RandomState.

## format_array
```python
format_array(arr)
```

Format `arr` into an ndarray where each row
corresponds to a single data point with a
1D list of features.
All data is formatted as a float. If `arr` cannot
be formatted in this manner, raise a ValueError.

**Parameters**
```
arr : array-like, shape (n_samples, n_features)
	List of data points.
```

**Returns**
```
arr : array-like, shape (n_samples, n_features)
	List of `n_features`-dimensional data points.
	Each row corresponds to a single data point.
```

## create_random_state
```python
create_random_state(seed=None)
```

Create a RandomState.

**Parameters**
```
seed : None or int or RandomState, default=None
	Initial seed for the RandomState. If seed is None,
	return the RandomState singleton. If seed is an int,
	return a RandomState with the seed set to the int.
	If seed is a RandomState, return that RandomState.
```

**Returns**
```
random_state : RandomState
	A RandomState object.
```
