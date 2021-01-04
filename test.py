import os
import numpy as np
import ctypes as ct
from numpy.ctypeslib import ndpointer
from timeit import default_timer as timer

def KrausToTheta(kraus):
	# Compute the Theta matrix of a channel whose Kraus matrix is given.
	# This is a wrapper for the KrausToTheta function in convert.so.
	dim = kraus.shape[1]
	n_kraus = kraus.shape[0]
	nq = int(np.ceil(np.log2(dim)))
	
	real_kraus = np.real(kraus).reshape(-1).astype(np.float64)
	imag_kraus = np.imag(kraus).reshape(-1).astype(np.float64)
	
	_convert = ct.cdll.LoadLibrary(os.path.abspath("convert.so"))
	_convert.KrausToTheta.argtypes = (
		ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"), # Real part of Kraus
		ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"), # Imgainary part of Kraus
		ct.c_int,  # number of qubits
	)
	# Output is the real part of Theta followed by its imaginary part.
	_convert.KrausToTheta.restype = ndpointer(dtype=ct.c_double, shape=(2 * n_kraus * n_kraus,))
	# Call the backend function.
	theta_out = _convert.KrausToTheta(real_kraus, imag_kraus, nq)
	# print("theta_out\n{}".format(theta_out))
	theta_real = theta_out[ : (n_kraus * n_kraus)].reshape([2, 2, 2, 2] * nq)
	theta_imag = theta_out[(n_kraus * n_kraus) : ].reshape([2, 2, 2, 2] * nq)
	theta = theta_real + 1j * theta_imag
	return theta


def KrausToPTM(kraus):
	# Convert from the Kraus representation to the PTM.
	# This is a wrapper for the KrausToPTM function in convert.so.
	dim = kraus.shape[1]
	n_kraus = kraus.shape[0]
	nq = int(np.ceil(np.log2(dim)))
	
	real_kraus = np.real(kraus).reshape(-1).astype(np.float64)
	imag_kraus = np.imag(kraus).reshape(-1).astype(np.float64)
	
	_convert = ct.cdll.LoadLibrary(os.path.abspath("convert.so"))
	_convert.KrausToPTM.argtypes = (
		ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"), # Real part of Kraus
		ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"), # Imgainary part of Kraus
		ct.c_int,  # number of qubits
	)
	# Output is the flattened PTM.
	_convert.KrausToPTM.restype = ndpointer(dtype=ct.c_double, shape=(n_kraus * n_kraus,))
	# Call the backend function.
	ptm_out = _convert.KrausToPTM(real_kraus, imag_kraus, nq)
	ptm = ptm_out.reshape([4, 4] * nq)
	return ptm

if __name__ == '__main__':
	# Test KrausToTheta function.
	nq = 4
	kraus = np.random.rand(4**nq, 2**nq, 2**nq) + 1j * np.random.rand(4**nq, 2**nq, 2**nq)
	print("Kraus with shape {}\n{}".format(kraus.shape, np.round(kraus, 3)))
	
	click = timer()
	theta = KrausToTheta(kraus)
	print("Theta matrix was constructed in %.3f seconds." % (timer() - click))
	print("Theta\n{}".format(theta))
	
	click = timer()
	ptm = KrausToPTM(kraus)
	print("PTM\n{}".format(ptm))
	print("PTM was constructed in %.3f seconds." % (timer() - click))