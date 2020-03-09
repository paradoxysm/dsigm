import pytest
import numpy as np

from dsigm import SGMM
from dsigm._exceptions import InitializationWarning

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

def test_validate_error():
	sgmm = SGMM()
	sgmm.dim = 0
	with pytest.raises(ValueError):
		sgmm._validate_data([0])

def test_initialize_core_error():
	sgmm = SGMM()
	with pytest.raises(RuntimeError):
		sgmm._initialize_core()

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]]),
])

def test_initialize(data):
	sgmm = SGMM()
	sgmm._initialize(data)
	sgmm = SGMM(init='random')
	sgmm._initialize(data)

def test_initialize_error():
	with pytest.raises(ValueError):
		sgmm = SGMM(init='bad')._initialize([0])

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]]),
])

def test_expectation(data):
	sgmm = SGMM()
	sgmm._initialize(data)
	p = sgmm._expectation(data)
	assert len(p) == len(sgmm.cores) and p.shape[-1] == len(data)

def test_expectation_error():
	sgmm = SGMM()
	with pytest.raises(RuntimeError):
		sgmm._expectation([0])

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]]),
])

def test_maximization(data):
	sgmm = SGMM()
	sgmm._initialize(data)
	p = sgmm._expectation(data)
	sgmm._maximization(data, p)

@pytest.mark.parametrize("data, n_parameters", [
	([0,1,3,4,1,2], 14),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]], 49),
])

def test_score_bic_parameters(data, n_parameters):
	sgmm = SGMM()
	sgmm._initialize(data)
	p = sgmm._expectation(data)
	sgmm.score(p)
	assert sgmm._n_parameters() == n_parameters
	sgmm.bic(data)

@pytest.mark.parametrize("data, n_cores", [
	([0,1,3,4,1,2], 5),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]], 4),
])

def test_stabilize(data, n_cores):
	sgmm = SGMM()
	sgmm._initialize(data)
	sgmm._stabilize(data)
	assert len(sgmm.cores) == n_cores

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,0,0],[1,1,1],[0,0,0],[-1,1,1],[0,0,1]]),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]]),
])

def test_fit_single(data):
	sgmm = SGMM(init_cores=2)
	sgmm._initialize(data)
	sgmm._fit_single(data)

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,0,0],[1,1,1],[0,0,0],[-1,1,1],[0,0,1]]),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]]),
])

def test_fit(data):
	sgmm = SGMM(init_cores=2)
	sgmm.fit(data)

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,0,0],[1,1,1],[0,0,0],[-1,1,1],[0,0,1]]),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]]),
])

def test_predict_warns(data):
	sgmm = SGMM(init_cores=2)
	with pytest.warns(InitializationWarning):
		sgmm.predict(data)

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,0,0],[1,1,1],[0,0,0],[-1,1,1],[0,0,1]]),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]]),
])

def test_predict(data):
	sgmm = SGMM(init_cores=2)
	sgmm._initialize(data)
	sgmm.predict(data)
