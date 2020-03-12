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

- ![Enhancement](https://img.shields.io/badge/-Enhancement-purple) :`SGMM` now operates with log pdf as opposed to pdf.
- ![Fix](https://img.shields.io/badge/-Fix-red) : `SGMM._stabilize` implements a new algorithm that converges properly as per [ISS #2](https://github.com/paradoxysm/dsigm/issues/2).
- ![Fix](https://img.shields.io/badge/-Fix-red) : `SGMM.fit` now fits the same way as `sklearn.GaussianMixture` as per [ISS #3](https://github.com/paradoxysm/dsigm/issues/3).
- ![Fix](https://img.shields.io/badge/-Fix-red) : `SGMM._expectation` now weights the probabilities so that all referring functions get the proper result as per [ISS #4](https://github.com/paradoxysm/dsigm/issues/4).
- ![Documentation](https://img.shields.io/badge/-Documentation-blue) : Documentation `SGMM`, `_utils`, `Core`, and `CoreCluster`.
- ![Documentation](https://img.shields.io/badge/-Documentation-blue) : Stabilization Guides for `SGMM`.
- ![Documentation](https://img.shields.io/badge/-Documentation-blue) : Updates to the 1D and 2D Guides for `SGMM`.

### Version 0.3.0
This is a pre-release record of changes that will be implemented in `dsigm 0.3.0`.

- ![Feature](https://img.shields.io/badge/-Feature-blueviolet) : `SGMM` implemented with fit and predict capacity.
- ![Feature](https://img.shields.io/badge/-Feature-blueviolet) : `Core` and `CoreCluster` implemented.
- ![Feature](https://img.shields.io/badge/-Feature-blueviolet) : `format_array` implemented in `_utils`.
- ![Feature](https://img.shields.io/badge/-Feature-blueviolet) : `create_random_state` implemented in `_utils`.
- ![Enhancement](https://img.shields.io/badge/-Enhancement-purple) : `SGMM` initializes through `sklearn.cluster.KMeans`.
- ![Documentation](https://img.shields.io/badge/-Documentation-blue) : 1D and 2D Guides for `SGMM`.
- ![Other](https://img.shields.io/badge/-Other-lightgrey) : Package structure and repository established.
