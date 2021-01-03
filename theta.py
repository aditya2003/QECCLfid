import numpy as np
import ctypes as ct
from numpy.ctypeslib import ndpointer
from define.QECCLfid.utils import GetNQubitPauli, PauliTensor
from define.QECCLfid.tensor import TensorTranspose, TensorKron, TensorTrace, TraceDot
from define.QECCLfid.contract import ContractTensorNetwork
from timeit import default_timer as timer

def KraussToTheta(kraus):
	# Compute the Theta matrix of a channel whose Kraus matrix is given.
	# This is a wrapper for the KrausToTheta function in convert.so.
	n_kraus = kraus.shape[0]
	dim = kraus.shape[1]
	_convert = ctypes.cdll.LoadLibrary(os.path.abspath("define/QECCLFid/convert.so"))
	_bmark.Benchmark.argtypes(
		ctypes.c_int,  # number of qubits
		ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"), # Real part of Kraus
		ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS") # Imgainary part of Kraus
	)
	# Output is the real part of Theta followed by its imaginary part.
	_bmark.Benchmark.restype = ctypes.POINTER(ctypes.c_double * (2 * n_kraus * dim * dim))

	real_kraus = np.real(kraus).reshape(-1).astype(float64)
	imag_kraus = np.imag(kraus).reshape(-1).astype(float64)
	
	theta_out = _convert.KrausToTheta(nq, real_kraus, imag_kraus)
	theta_flat = ctypes.cast(
		theta_out, ctypes.POINTER(ctypes.c_double * n_kraus * n_kraus)
	).contents

	theta_flat_array = np.ctypeslib.as_array(theta_flat)
	theta_real = theta_flat_array[ : (n_kraus * n_kraus)].reshape([2, 2, 2, 2] * nq)
	theta_imag = theta_flat_array[(n_kraus * n_kraus) : ].reshape([2, 2, 2, 2] * nq)
	theta = theta_real + 1j * theta_imag
	return theta


def ThetaToChiElement(pauli_op_i, pauli_op_j, theta_dict):
	# Convert from the Theta representation to the Chi representation.
	# The "Theta" matrix T of a CPTP map whose chi-matrix is X is defined as:
	# T_ij = \sum_(ij) [ X_ij (P_i o (P_j)^T) ]
	# So we find that
	# Chi_ij = Tr[ (P_i o (P_j)^T) T]
	# Note that [P_i o (P_j)^T] can be expressed as a product of the single qubit Pauli matrices
	# (P^(1)_i)_(r1,c1) (P^(1)_j)_(c(N+1),r(N+1)) x ... x (P^(N)_i)_(c(N),r(N)) (P^(N)_j)_(c(2N),r(2N))
	# We will store T as a Tensor with dimension = (2 * number of qubits) and bond dimension = 4.
	# click = timer()
	nq = pauli_op_i.size
	# ops = [(supp_theta, theta)]
	
	Pj = [((q,), PauliTensor(pauli_op_j[q, np.newaxis])) for q in range(nq)]
	PjT = [((q,), (-1)**(int(pauli_op_j[q] == 2)) * PauliTensor(pauli_op_j[q, np.newaxis])) for q in range(nq)]
	Pi = [((nq + q,), PauliTensor(pauli_op_i[q, np.newaxis])) for q in range(nq)]
	
	(__, chi_elem) = ContractTensorNetwork(theta_dict + PjT + Pi, end_trace=1)
	chi_elem /= 4**nq
	
	# print("Chi element of Pauli op {} = {}".format(pauli_op_i, chi_elem))
	if ((np.real(chi_elem) <= -1E-15) or (np.abs(np.imag(chi_elem)) >= 1E-15)):
		print("Pi\n{}\nPj\n{}".format(pauli_op_i, pauli_op_j))
		print("Chi = %g + i %g" % (np.real(chi_elem), np.imag(chi_elem)))
		exit(0)
	return chi_elem


if __name__ == '__main__':
	# depolarizing channel
	N = 1
	Pauli = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=np.complex128)
	kraus_dp = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
	rate = 0.1
	kraus_dp[0, :, :] = np.sqrt(1 - rate) * Pauli[0, :, :]
	for k in range(1, 4):
		kraus_dp[k, :, :] = np.sqrt(rate/3) * Pauli[k, :, :]
	theta = KraussToTheta(kraus_dp)
	print("Theta\n{}".format(theta))

	# testing theta to chi element
	pauli_op_i = GetNQubitPauli(0, 4)
	pauli_op_j = GetNQubitPauli(0, 4)
	theta = np.random.rand(16, 16)
	supp_theta = (0, 1)
	chi_ij = ThetaToChiElement(pauli_op_i, pauli_op_j, theta, supp_theta)
	print("chi_ij = {}".format(chi_ij))
