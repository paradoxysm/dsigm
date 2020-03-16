import pytest
import numpy as np
from scipy.stats import norm
from sklearn.datasets import make_spd_matrix

from dsigm.mixture import SGMM
from dsigm import Core
from dsigm._exceptions import InitializationWarning

"""
Test
----
SGMM
"""

@pytest.mark.parametrize("data, init_cores", [
	([0,1,3,4,1,2,2,-1,2.5], 2),
	([[1,0,0],[1,1,1],[0,1,0],[-1,1,1],[0,0,1],[2,0,-1],[1.5,0,0.5]], 2),
	([[1,0,0],[1,1,1],[0,1,0],[-1,1,1],[0,0,1],[2,0,-1],[1.5,0,0.5]], 10),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]], 4),
])

def test_orient_stabilizer(data, init_cores):
	sgmm = SGMM(init_cores=init_cores)
	sgmm._orient_stabilizer(data)

@pytest.mark.parametrize("data, interval, abic, midpoint, abic_m", [
	([0,1,3,4,1,2,2,-1,2.5], (1,3), (250, 600), 2, 300),
	([[1,0,0],[1,1,1],[0,1,0],[-1,1,1],[0,0,1],[2,0,-1],[1.5,0,0.5]], (1,3), (250, 600), 2, 300),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]], (1,10), (250, 600), 5, 300),
])

def test_halve_interval(data, interval, abic, midpoint, abic_m):
	sgmm = SGMM()
	sgmm._halve_interval(data, interval, abic, midpoint, abic_m)

@pytest.mark.parametrize("data, interval, abic", [
	([0,1,3,4,1,2,2,-1,2.5], (1,3), (250, 600)),
	([[1,0,0],[1,1,1],[0,1,0],[-1,1,1],[0,0,1],[2,0,-1],[1.5,0,0.5]], (1,3), (250, 600)),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]], (1,5), (250, 600)),
])

def test_truncate_interval(data, interval, abic):
	sgmm = SGMM()
	sgmm._truncate_interval(data, interval, abic)

@pytest.mark.parametrize("data, init_cores", [
	([0,1,3,4,1,2,2,-1,2.5], 2),
	([[1,0,0],[1,1,1],[0,1,0],[-1,1,1],[0,0,1],[2,0,-1],[1.5,0,0.5]], 2),
	([[0,21,3],[2,4,3],[34,3,2],[2,5,1],[1,6,3],[23,12,5],[2,6,9]], 10),
])

def test_fit_stabilize(data, init_cores):
	sgmm = SGMM(init_cores=init_cores)
	sgmm._fit_stabilize(data)

def test_fit_stabilize_1D():
	n = 1000
	mu = [-6, 5, 13] #+ [-20, 40, 80]
	sigma = [2, 3, 2.5] #+ [1, 1.8, 5]
	X = []
	for i in range(n):
	    Z = np.random.choice(np.arange(len(mu))) # select the synthetic component
	    X.append(np.random.normal(mu[Z], sigma[Z], 1))
	X = np.array(X)
	sgmm = SGMM()
	sgmm.fit(X)
	assert len(sgmm.cores) == 3

def test_fit_stabilize_2D():
	n = 200
	mu = [[10.4, 10.2], [-1.4, 1.6], [2.4, 5.4], [6.4, 2.4]]
	sigma = []
	for s in range(len(mu)):
	  sigma.append(make_spd_matrix(2))
	X = []
	for m, s in zip(mu,sigma):
	  x = np.random.multivariate_normal(m, s, n)
	  X += list(x)
	X = np.array(X)
	np.random.shuffle(X)
	sgmm = SGMM()
	sgmm.fit(X)
	assert len(sgmm.cores) == 4

def test_fit_no_stabilize():
	sgmm = SGMM(stabilize=None)
	sgmm.fit([0,1,5,2,35,4,5,7,5,3,5,3,2])
	assert len(sgmm.cores) == 5
