import pytest
import numpy as np

from dsigm import Core, CoreCluster

"""
Test
----
Core
"""

def test_core_init():
	Core()
	Core(mu=[1])
	Core(mu=[1,3], sigma=[2,4], delta=1)
	Core(cluster=CoreCluster())

def test_core_init_error():
	with pytest.raises(ValueError):
		Core(mu=1)
	with pytest.raises(ValueError):
		Core(mu=[1,3], sigma=[3])
	with pytest.raises(ValueError):
		Core(mu=None)
	with pytest.raises(ValueError):
		Core(mu=object())

def test_core_pdf():
	core = Core()
	data = [-4,-2,0,1,5,2,2]
	p = np.around(core.pdf(data), decimals=4)
	assert np.all(p == [0.0001, 0.0540, 0.3989, 0.2420, 0.0000, 0.0540, 0.0540])

"""
Test
----
CoreCluster
"""
