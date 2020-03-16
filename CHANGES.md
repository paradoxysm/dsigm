# Changelog

### Legend

- ![Feature](https://img.shields.io/badge/-Feature-blueviolet) : Something that you couldn’t do before.
- ![Enhancement](https://img.shields.io/badge/-Enhancement-purple) : A miscellaneous minor improvement.
- ![Efficiency](https://img.shields.io/badge/-Efficiency-indigo) : An existing feature now may not require as much computation or memory.
- ![Fix](https://img.shields.io/badge/-Fix-red) : Something that previously didn’t work as documentated or as expected should now work.
- ![Documentation](https://img.shields.io/badge/-Documentation-blue) : An update to the documentation.
- ![Other](https://img.shields.io/badge/-Other-lightgrey) : Miscellaneous updates such as package structure or GitHub quality of life updates.

### Version 0.3.1
This is a pre-release record of changes that will be implemented in `dsigm 0.3.1`.

- ![Feature](https://img.shields.io/badge/-Feature-blueviolet) : [`SGMM`](https://github.com/paradoxysm/dsigm/blob/master/dsigm/mixture/_sgmm.py) refactored to a parent [`GMM`](https://github.com/paradoxysm/dsigm/blob/master/dsigm/mixture/_gmm.py) class and a subclass for [`SGMM`](https://github.com/paradoxysm/dsigm/blob/master/dsigm/mixture/_sgmm.py).
- ![Enhancement](https://img.shields.io/badge/-Enhancement-purple) : [`SGMM`](https://github.com/paradoxysm/dsigm/blob/master/dsigm/mixture/_sgmm.py) now operates with log probabilities.
- ![Fix](https://img.shields.io/badge/-Fix-red) : [`SGMM._fit_stabilize`](https://github.com/paradoxysm/dsigm/blob/master/dsigm/mixture/_sgmm.py) implements a new algorithm that converges properly as per ISS#2.
- ![Fix](https://img.shields.io/badge/-Fix-red) : [`SGMM.fit`](https://github.com/paradoxysm/dsigm/blob/master/dsigm/mixture/_sgmm.py) now fits the same way as `sklearn.GaussianMixture` as per [ISS #3](https://github.com/paradoxysm/dsigm/issues/6).
- ![Fix](https://img.shields.io/badge/-Fix-red) : [`SGMM._expectation`](https://github.com/paradoxysm/dsigm/blob/master/dsigm/mixture/_sgmm.py) now weights the probabilities so that all referring functions get the proper result as per [ISS #4](https://github.com/paradoxysm/dsigm/issues/4).
- ![Documentation](https://img.shields.io/badge/-Documentation-blue) : Documentation for [`GMM`](https://github.com/paradoxysm/dsigm/blob/master/doc/pydoc/GMM.md), [`SGMM`](https://github.com/paradoxysm/dsigm/blob/master/doc/pydoc/SGMM.md), [`utils`](https://github.com/paradoxysm/dsigm/blob/master/doc/pydoc/utils.md), [`Core`](https://github.com/paradoxysm/dsigm/blob/master/doc/pydoc/Core.md), and [`CoreCluster`](https://github.com/paradoxysm/dsigm/blob/master/doc/pydoc/CoreCluster.md).
- ![Documentation](https://img.shields.io/badge/-Documentation-blue) : Stabilization Guide for [`SGMM`](https://github.com/paradoxysm/dsigm/blob/master/doc/guides/stabilization_analysis.ipynb).
- ![Documentation](https://img.shields.io/badge/-Documentation-blue) : Updates to the [1D](https://github.com/paradoxysm/dsigm/blob/master/doc/guides/SGMM_1D.ipynb) and [2D](https://github.com/paradoxysm/dsigm/blob/master/doc/guides/SGMM_2D.ipynb) Guides for `SGMM`.

### Version 0.3.0
This is a pre-release record of changes that will be implemented in `dsigm 0.3.0`.

- ![Feature](https://img.shields.io/badge/-Feature-blueviolet) : [`SGMM`](https://github.com/paradoxysm/dsigm/blob/master/dsigm/mixture/_sgmm.py) implemented with fit and predict capacity.
- ![Feature](https://img.shields.io/badge/-Feature-blueviolet) : [`Core`](https://github.com/paradoxysm/dsigm/blob/master/dsigm/_core.py) and [`CoreCluster`](https://github.com/paradoxysm/dsigm/blob/master/dsigm/_core.py) implemented.
- ![Feature](https://img.shields.io/badge/-Feature-blueviolet) : `format_array` implemented in [`_utils`](https://github.com/paradoxysm/dsigm/blob/master/dsigm/_utils.py).
- ![Feature](https://img.shields.io/badge/-Feature-blueviolet) : `create_random_state` implemented in [`_utils`](https://github.com/paradoxysm/dsigm/blob/master/dsigm/_utils.py).
- ![Enhancement](https://img.shields.io/badge/-Enhancement-purple) : [`SGMM`](https://github.com/paradoxysm/dsigm/blob/master/dsigm/mixture/_sgmm.py) initializes through `sklearn.cluster.KMeans`.
- ![Documentation](https://img.shields.io/badge/-Documentation-blue) : [1D](https://github.com/paradoxysm/dsigm/blob/master/doc/guides/SGMM_1D.ipynb) and [2D](https://github.com/paradoxysm/dsigm/blob/master/doc/guides/SGMM_2D.ipynb) Guides for `SGMM`.
- ![Other](https://img.shields.io/badge/-Other-lightgrey) : Package structure and repository established.
