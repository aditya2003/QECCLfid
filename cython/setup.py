#!/usr/bin/env python
# https://github.com/theochem/python-cython-ci-example/blob/master/setup.py
import numpy as np
from setuptools import setup, Extension
from Cython.Build import build_ext

sources = ["krauss_theta", "pauli", "tracedot"]
ext_modules = [None for __ in sources]
for s in range(len(sources)):
	src = sources[s]
	ext_modules[s] = Extension(src, ["%s.pyx" % (src)], include_dirs=[np.get_include()])
	ext_modules[s].cython_c_in_temp = True

setup(
	name='compose',
	cmdclass={'build_ext': build_ext},
	ext_modules=ext_modules,
)