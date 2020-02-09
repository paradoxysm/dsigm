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
	version='0.1.0',
    description='Density-sensitive Evolution-based Self-stabilization of Independent Gaussian Mixtures (DESIGM) Clustering',
    long_description=(read('README.md') + '\n\n'),
	url='http://github.com/paradoxysm/desigm',
    author='paradoxysm',
    license='GPLv3+',
    packages=find_packages(),
    install_requires=[
    	'scipy',
		'sklearn',
		'numpy',
		'tqdm'
    ],
	python_requires='>=2.7, >=3.4, <4',
	classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
		'Topic :: Scientific/Engineering :: Bio-informatics',
        'Topic :: Software Development :: System :: Clustering',
    ],
	keywords='clustering'
    zip_safe=False)
