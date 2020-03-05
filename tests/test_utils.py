import pytest
import numpy as np

from dsigm._utils import format_array, create_random_state

"""
Test
----
format_array
"""

@pytest.mark.parametrize("arr, exp", [
	([0]),
	([[0]]),
	([[3,4,5]]),
	([[3,4],[5,6]]),
	([1,2,34]),
	([[[1,2],[3,4]]]),
])

def test_format_array(arr, exp):
	arr = format_array(arr)
	assert np.array_equal(arr, exp)

@pytest.mark.parametrize("arr", [
	(0),
	([[3], [3,4]]),
	([3, 'd']),
	([[[2,3],[4,5]],[[5,6],[7,8]]]),
])

def test_format_array_except(arr):
	with pytest.raises(ValueError):
		arr = format_array(arr)

"""
Test
----
create_random_state
"""

@pytest.mark.parametrize("seed, exp", [
	(None, np.random.mtrand._rand),
	(1, np.random.RandomState(seed=1)),
	(np.random.RandomState(seed=2), np.random.RandomState(seed=2)),
])

def test_(seed, exp):
	rs = create_random_state(seed)
	assert np.all(rs[1] == exp[1])

@pytest.mark.parametrize("seed", [
	("b"),
	(1.3),
])

def test_format_array_except(seed):
	with pytest.raises(ValueError):
		rs = create_random_state(seed)
