import pytest
import numpy as np

from dsigm.mixture import GMM
from dsigm import Core
from dsigm._exceptions import InitializationWarning

"""
Test
----
GMM
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
	gmm = GMM()
	assert np.all(gmm._validate_data(data) == exp)
	assert gmm.dim == dim_exp

def test_validate_error():
	gmm = GMM()
	gmm.dim = 0
	with pytest.raises(ValueError):
		gmm._validate_data([0])

def test_initialize_core_error():
	gmm = GMM()
	with pytest.raises(RuntimeError):
		gmm._initialize_core()

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]]),
])

def test_initialize(data):
	gmm = GMM()
	gmm._initialize(data)
	gmm = GMM(init='random')
	gmm._initialize(data)

def test_initialize_error():
	with pytest.raises(ValueError):
		gmm = GMM(init='bad')
		gmm._initialize([0])

def test_get_params():
	gmm = GMM()
	gmm.cores = [Core(mu=[0], sigma=[1], delta=[1])]
	assert gmm.get_params() == (np.asarray([[0]]), np.asarray([[1]]), np.asarray([[1]]))

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]]),
])

def test_expectation(data):
	gmm = GMM()
	gmm._initialize(data)
	p, p_norm, resp = gmm._expectation(data)
	assert len(p) == len(gmm.cores) and p.shape[-1] == len(data)

def test_expectation_error():
	gmm = GMM()
	with pytest.raises(RuntimeError):
		gmm._expectation([0])

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]]),
])

def test_maximization(data):
	gmm = GMM()
	gmm._initialize(data)
	p, p_norm, resp = gmm._expectation(data)
	gmm._maximization(data, resp)

@pytest.mark.parametrize("data, n_parameters", [
	([0,1,3,4,1,2], 14),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]], 49),
])

def test_score_bic_parameters(data, n_parameters):
	gmm = GMM()
	gmm._initialize(data)
	p, p_norm, resp = gmm._expectation(data)
	gmm.score(p_norm)
	assert gmm._n_parameters() == n_parameters
	gmm.bic(data)

@pytest.mark.parametrize("data, n_parameters", [
	([0,1,3,4,1,2], 14),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]], 49),
])

def test_score_aic_parameters(data, n_parameters):
	gmm = GMM()
	gmm._initialize(data)
	p, p_norm, resp = gmm._expectation(data)
	gmm.score(p_norm)
	assert gmm._n_parameters() == n_parameters
	gmm.aic(data)

@pytest.mark.parametrize("data, n_parameters", [
	([0,1,3,4,1,2], 14),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]], 49),
])

def test_score_abic_parameters(data, n_parameters):
	gmm = GMM()
	gmm._initialize(data)
	p, p_norm, resp = gmm._expectation(data)
	gmm.score(p_norm)
	assert gmm._n_parameters() == n_parameters
	gmm.abic(data)

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,0,0],[1,1,1],[0,0,0],[-1,1,1],[0,0,1],[2,0,-1]]),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]]),
])

def test_fit_single(data):
	gmm = GMM(init_cores=2)
	gmm._initialize(data)
	gmm._fit_single(data)

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2,2,-1,2.5]),
	([[1,0,0],[1,1,1],[0,1,0],[-1,1,1],[0,0,1],[2,0,-1],[1.5,0,0.5]]),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]]),
])

def test_fit(data):
	gmm = GMM(init_cores=2)
	gmm.fit(data)
	gmm = GMM(init_cores=1)
	gmm.fit(data)

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,0,0],[1,1,1],[0,0,0],[-1,1,1],[0,0,1]]),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]]),
])

def test_predict_warns(data):
	gmm = GMM(init_cores=2)
	with pytest.warns(InitializationWarning):
		gmm.predict(data)

@pytest.mark.parametrize("data", [
	([0,1,3,4,1,2]),
	([[0,0,0],[1,1,1],[0,0,0],[-1,1,1],[0,0,1],[2,0,-1]]),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]]),
])

def test_predict(data):
	gmm = GMM(init_cores=2)
	gmm._initialize(data)
	gmm.predict(data)
