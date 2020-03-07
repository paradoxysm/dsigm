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

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,21,3],[2,4,3],[34,3,2]]),
])

def test_expectation(data):
	sgmm = SGMM()
	sgmm._initialize(data)
	p = sgmm._expectation(data)
	assert len(p) == len(sgmm.cores) and p.shape[-1] == len(data)

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,21,3],[2,4,3],[34,3,2]]),
])

def test_maximization(data):
	sgmm = SGMM()
	sgmm._initialize(data)
	p = sgmm._expectation(data)
	sgmm._maximization(data, p)

@pytest.mark.parametrize("data, n_parameters", [
	([0,1,3,4,1,2], 29),
	([[0,21,3],[2,4,3],[34,3,2]], 99),
])

def test_score_bic_parameters(data, n_parameters):
	sgmm = SGMM()
	sgmm._initialize(data)
	p = sgmm._expectation(data)
	sgmm.score(p)
	assert sgmm._n_parameters() == n_parameters
	sgmm.bic(data)

@pytest.mark.parametrize("data, bic, prev_bic, n_cores", [
	([0,1,3,4,1,2], 0.5, 5, 8),
	([[0,21,3],[2,4,3],[34,3,2]], 5, 0.5, 12),
])

def test_stabilize(data, bic, prev_bic, n_cores):
	sgmm = SGMM()
	sgmm._initialize(data)
	p = sgmm._expectation(data)
	sgmm._stabilize(bic, prev_bic, p)
	#assert len(sgmm.cores) == n_cores
