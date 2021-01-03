#!/usr/bin/env python
# https://github.com/theochem/python-cython-ci-example/blob/master/setup.py
import os
import numpy as np
from setuptools import setup, Extension
from Cython.Build import build_ext

sources = ["krauss_ptm", "krauss_theta", "pauli", "tracedot"]
ext_modules = [None for __ in sources]
for s in range(len(sources)):
	src = sources[s]
	ext_modules[s] = Extension(src, ["%s.pyx" % (src)], include_dirs=[np.get_include()])

extensions = cythonize(ext_modules, language_level = "3")

os.environ["CFLAGS"] = "-lm -O3 -Wall -ffast-math -march=native -mfpmath=sse -fno-signed-zeros"

setup(
	name='compose',
	cmdclass={'build_ext': build_ext},
	ext_modules=extensions,
)