import pytest
import numpy as np

from dsigm import SGMM

"""
Test
----
SGMM
"""

@pytest.mark.parametrize("data, exp, dim_exp", [
	([0], np.asarray([[0.]]), 1),
	([[0]], np.asarray([[0.]]), 1),
	([[[0]]], np.asarray([[0.]]), 1),
	([[3,4,5]], np.asarray([[3.,4.,5.]]), 3),
	([[[3],[4],[5]]], np.asarray([[3.],[4.],[5.]]), 1),
	([[3,4],[5,6]], np.asarray([[3.,4.],[5.,6.]]), 2),
	([1,2,34], np.asarray([[1.],[2.],[34.]]), 1),
	([[[1,2],[3,4]]], np.asarray([[1.,2.],[3.,4.]]), 2),
])

def test_validate_data(data, exp, dim_exp):
	sgmm = SGMM()
	assert np.all(sgmm._validate_data(data) == exp)
	assert sgmm.dim == dim_exp

def test_initialize_core_error():
	sgmm = SGMM()
	with pytest.raises(RuntimeError):
		sgmm._initialize_core()

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,21,3],[2,4,3],[34,3,2]]),
])

def test_initialize(data):
	sgmm = SGMM()
	sgmm._initialize(data)
