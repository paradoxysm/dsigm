import os
from setuptools import setup, find_packages

def read(*paths):
    """
	Build a file path from *paths* and return the contents.
	"""
    with open(os.path.join(*paths), 'r') as f:
        return f.read()

setup(
	name='desigm',
	version='0.3.0',
    description='Density-sensitive Evolution-based Self-stabilization of Independent Gaussian Mixtures (DESIGM) Clustering',
    long_description=(read('README.md') + '\n\n'),
	url='http://github.com/paradoxysm/desigm',
	download_url = 'https://github.com/paradoxysm/desigm/archive/0.3.0.tar.gz',
    author='paradoxysm',
	author_email='paradoxysm.dev@gmail.com',
    license='BSD-3-Clause',
    packages=find_packages(),
    install_requires=[
    	'scipy',
		'sklearn',
		'numpy'
    ],
	python_requires='>=3.4, <4',
	classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
		'Topic :: Scientific/Engineering :: Bio-informatics',
        'Topic :: Software Development :: System :: Clustering',
    ],
	keywords=['clustering','python','data-science'],
    zip_safe=True)
