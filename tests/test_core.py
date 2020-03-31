import pytest
import numpy as np

from dsigm import Core, CoreCluster

"""
Test
----
Core
"""

@pytest.mark.parametrize("mu, sigma, delta, cluster, exp", [
	([0],[1],[1],None, Core()),
	([1],[1],[1],None, Core(mu=[1])),
	([1,3],[[2,4],[0,3]],[1],None, Core(mu=[1,3],sigma=[[2,4],[0,3]])),
	([0],[1],[1],CoreCluster(), Core(cluster=CoreCluster())),
])

def test_core(mu, sigma, delta, cluster, exp):
	core = Core(mu=mu, sigma=sigma, delta=delta, cluster=cluster)
	assert core == exp

def test_core_unequal():
	assert Core() != 0

@pytest.mark.parametrize("mu, sigma, delta, cluster", [
	(1,[1],[1],None),
	([1],1,[1],None),
	([0],[1],1,0),
	([1,3],[3],[1],None),
	(None,[1],[1],None),
	(object(),[1],[1],None),
	([0],[1],3,None),
	([0],[1],[1],object())
])

def test_core_error(mu, sigma, delta, cluster):
	with pytest.raises(ValueError):
		Core(mu=mu, sigma=sigma, delta=delta, cluster=cluster)

def test_core_pdf():
	core = Core()
	data = [-4,-2,0,1,5,2,2]
	p = np.around(core.pdf(data), decimals=4)
	assert np.all(p == [0.0001, 0.0540, 0.3989, 0.2420, 0.0000, 0.0540, 0.0540])
	p = np.around(core.pdf(data, weight=True), decimals=4)
	assert np.all(p == [0.0001, 0.0540, 0.3989, 0.2420, 0.0000, 0.0540, 0.0540])

def test_core_logpdf():
	core = Core()
	data = [-4,-2,0,1,5,2,2]
	p = np.around(core.logpdf(data), decimals=4)
	assert np.all(p == [-8.9189, -2.9189, -0.9189, -1.4189, -13.4189, -2.9189, -2.9189])
	p = np.around(core.logpdf(data, weight=True), decimals=4)
	assert np.all(p == [-8.9189, -2.9189, -0.9189, -1.4189, -13.4189, -2.9189, -2.9189])

"""
Test
----
CoreCluster
"""

@pytest.mark.parametrize("cores, parents, children, exp", [
	([],[],[], CoreCluster()),
	([Core(),Core()],[CoreCluster(),CoreCluster()],[CoreCluster()],
		CoreCluster(cores=[Core(),Core()], parents=[CoreCluster(),CoreCluster()],
						children=[CoreCluster()])),
])

def test_cluster(cores, parents, children, exp):
	cluster = CoreCluster(cores=cores, parents=parents, children=children)
	assert cluster == exp

def test_cluster_unequal():
	assert CoreCluster() != 0

@pytest.mark.parametrize("cores, parents, children", [
	(1,[],[]),
	([1, Core()],[],[]),
	([],1,1),
	([],[],1),
	([CoreCluster(), Core()],[],[]),
	([],[],[Core()]),
	([],[Core()],[]),
	([],[],[CoreCluster(),1]),
])

def test_cluster_error(cores, parents, children):
	with pytest.raises(ValueError):
		CoreCluster(cores=cores, parents=parents, children=children)
