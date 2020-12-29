#!/usr/bin/env python
# https://github.com/theochem/python-cython-ci-example/blob/master/setup.py
import numpy as np
from setuptools import setup, Extension
from Cython.Build import build_ext

setup(
	name='compose',
	version=1.0,
	description="Compose channels represented in the Kraus decomposition",
	cmdclass={'build_ext': build_ext},
	packages=['krauss_theta', "contract"],
	ext_modules=[Extension(
		'krauss_theta',
		sources=["krauss_theta.pyx", "contract.pyx"],
		include_dirs=[np.get_include()],
		compiler_directives={'language_level' : "3"}
	)],
)
#
# from setuptools import setup
# from Cython.Build import cythonize

# setup(
#     name="My hello app",
#     ext_modules=cythonize("src/*.pyx", include_path=[...]),
# )