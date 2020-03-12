"""Warnings"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

class ConvergenceWarning(UserWarning):
	"""
	Custom warning to capture convergence issues.
	"""

class InitializationWarning(UserWarning):
	"""
	Custom warning to capture initialization issues.
	"""

class StabilizationWarning(UserWarning):
	"""
	Custom warning to capture stabilization issues.
	"""
