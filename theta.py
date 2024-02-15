import os
import numpy as np
import ctypes as ct
from numpy.ctypeslib import ndpointer
from define.QECCLfid.utils import GetNQubitPauli, PauliTensor
from define.QECCLfid.tensor import TensorTranspose, TensorKron, TensorTrace, TraceDot
from define.QECCLfid.contract import ContractTensorNetwork
from timeit import default_timer as timer
from define.QECCLfid.utils import GetNQubitPauli # only for decoding purposes

def KrausToTheta_Python(kraus):
	"""
	Kraus to theta in Python.
	1. Convert from Kraus to Chi
		Convert from the Kraus representation to the chi matrix.
	    chi_ij = \sum_k Tr(E_k P_i) Tr(E^\dag_k P_j)
	           = \sum_(k,l,m) (E_k P_i)_(ll) (E^\dag_k Pj)_(mm)
	           = \sum_(k,l,m,n,p) (E_k)_(ln) (P_i)_(nl) (E*_k)_(pm) (P_j)_(pm)
	2. Convert from Chi to Theta.
		theta = \sum_(ij) chi_ij (P_i o (P_j)^T)
	"""
	dim = kraus.shape[1]
	nq = int(np.ceil(np.log2(dim)))
	npauli = np.power(4, nq, dtype = np.int64)

	paulis = np.zeros((npauli, dim, dim), dtype = np.complex128)
	for p in range(npauli):
		paulis[p, :, :] = PauliTensor(GetNQubitPauli(p, nq)).reshape(dim, dim)

	# theta = np.zeros((npauli, npauli), dtype = np.complex128)
	# for i in range(npauli):
	# 	for j in range(npauli):
	# 		chi_ij = np.einsum('kln,nl,kpm,pm->', kraus, paulis[i, :, :], np.conj(kraus), paulis[j, :, :]) / npauli
	# 		# print("chi[{}, {}] = {}".format(i, j, chi_ij))
	# 		theta = theta + chi_ij * np.kron(paulis[i, :, :], np.transpose(paulis[j, :, :]))
	chi = np.einsum('kln,inl,kpm,jpm->ij', kraus, paulis, np.conj(kraus), paulis) / npauli
	theta = np.einsum('ij,ikl,jmn->kmln', chi, paulis, np.transpose(paulis, axes=(0, 2, 1)))
	theta_reshaped = theta.reshape([2, 2, 2, 2] * nq)
	return theta_reshaped

def KrausToTheta(kraus):
	# Compute the Theta matrix of a channel whose Kraus matrix is given.
	# This is a wrapper for the KrausToTheta function in convert.so.
	dim = kraus.shape[1]
	nq = int(np.ceil(np.log2(dim)))
	nkr = kraus.shape[0]

	# print("dim = {}, nq = {} and nkr = {}".format(dim, nq, nkr))
	
	real_kraus = np.real(kraus).reshape(-1).astype(np.float64)
	imag_kraus = np.imag(kraus).reshape(-1).astype(np.float64)
	
	_convert = ct.cdll.LoadLibrary(os.path.abspath("define/QECCLfid/convert.so"))
	_convert.KrausToTheta.argtypes = (
		ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"), # Real part of Kraus
		ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"), # Imgainary part of Kraus
		ct.c_int,  # number of qubits
		ct.c_long, # number of Kraus operators (can set to -1 if the number of Kraus operators is 4^n)
	)
	# Output is the real part of Theta followed by its imaginary part.
	_convert.KrausToTheta.restype = ndpointer(dtype=ct.c_double, shape=(2 * 4**nq * 4**nq,))
	# Call the backend function.
	theta_out = _convert.KrausToTheta(real_kraus, imag_kraus, nq, nkr)
	print("theta_out\n{}".format(theta_out))
	theta_real = theta_out[ : (4**nq * 4**nq)].reshape([2, 2, 2, 2] * nq)
	theta_imag = theta_out[(4**nq * 4**nq) :].reshape([2, 2, 2, 2] * nq)
	theta = theta_real + 1j * theta_imag
	#
	# Get Theta matrix from pure Python code
	theta_mat_alt = KrausToTheta_Python(kraus) # Only for decoding purposes
	print("theta - theta_mat_alt\n{}".format(theta.reshape(4**nq, 4**nq) -theta_mat_alt.reshape(4**nq, 4**nq))) # Only for decoding purposes
	# print("Tr(theta) = {}".format(np.trace(theta_mat))) # Only for decoding purposes
	return theta


def ThetaToChiElement(pauli_op_i, pauli_op_j, theta_dict):
	# Convert from the Theta representation to the Chi representation.
	# The "Theta" matrix T of a CPTP map whose chi-matrix is X is defined as:
	# T = \sum_(ij) [ X_ij (P_i o (P_j)^T) ]
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
	
	# print("Calling ContractTensorNetwork on\n\ttheta_dict sizes: {}".format([supp for (supp, __) in theta_dict]))
	(__, chi_elem) = ContractTensorNetwork(theta_dict + PjT + Pi, end_trace=1)
	# chi_elem = 0 # For debugging purposes.
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
