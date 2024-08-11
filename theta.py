import os
import numpy as np
import ctypes as ct
from tqdm import tqdm
from numpy.ctypeslib import ndpointer
# from timeit import default_timer as timer # only for decoding purposes
from define.QECCLfid.utils import GetNQubitPauli, PauliTensor
from define.QECCLfid.tensor import TensorTranspose, TensorKron, TensorTrace, TraceDot
from define.QECCLfid.contract import ContractTensorNetwork
from timeit import default_timer as timer
from define.QECCLfid.utils import GetNQubitPauli, PauliTensor

def KrausToTheta_Python(kraus):
	r"""
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

	# start = timer()
	paulis = np.zeros((npauli, dim, dim), dtype = np.complex128)
	for p in range(npauli):
		paulis[p, :, :] = PauliTensor(GetNQubitPauli(p, nq)).reshape(dim, dim)
	# print("Enumerating Paulis took {} seconds.".format(timer() - start))

	# theta = np.zeros((npauli, npauli), dtype = np.complex128)
	# for i in range(npauli):
	# 	for j in range(npauli):
	# 		chi_ij = np.einsum('kln,nl,kpm,pm->', kraus, paulis[i, :, :], np.conj(kraus), paulis[j, :, :]) / npauli
	# 		# print("chi[{}, {}] = {}".format(i, j, chi_ij))
	# 		theta = theta + chi_ij * np.kron(paulis[i, :, :], np.transpose(paulis[j, :, :]))
	start = timer()
	chi = np.einsum('kln,inl,kpm,jpm->ij', kraus, paulis, np.conj(kraus), paulis) / npauli
	# print("Chi computed in {} seconds.\n{}".format(timer() - start, chi))
	theta = np.einsum('ij,ikl,jmn->kmln', chi, paulis, np.transpose(paulis, axes=(0, 2, 1)))
	# print("Theta matrix computed in {} seconds.".format(timer() - start))
	theta_reshaped = theta.reshape([2, 2, 2, 2] * nq)
	return theta_reshaped

def get_theta_pauli_channel(nq, infid):
	# Compute the Theta matrix of a random Pauli channel with fixed infidelity.
	# Recall that
	# Theta = \sum_(ij) chi_ij (P_i o (P_j)^T)
	# So, the Theta matrix of a random Pauli channel is
	# Theta = \sum_(i) chi_ii (P_i o (P_i)^T)
	#       = np.einsum('ii')
	# Note that the chi matrix for a random Pauli channel is a diagonal matrix where all the diagonal entries except for the first one are random.
	# The first diagonal entry is fidelity.
	npauli = np.power(4, nq, dtype = np.uint64)
	chi = np.random.uniform(0, 1, size=(npauli,))
	chi[0] = 1 - infid
	chi[1:] = infid * chi[1:] / np.sum(chi[1:])
	# chi = np.array([0.9, 0.05, 0.03, 0.02]) # only for debugging purposes
	theta = np.zeros((npauli, npauli), dtype = np.complex128)
	for p in range(npauli):
		pauli_op = GetNQubitPauli(p, nq)
		tn_pauli = PauliTensor(pauli_op)
		pauli_mat = tn_pauli.reshape(2**nq, 2**nq)
		theta = theta + chi[p] * np.kron(pauli_mat, pauli_mat.T)
	###########
	# only for debugging purposes
	# (I,X,Y,Z) = (np.eye(2), np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]]))
	# theta = chi[0] * np.kron(I, I) + chi[1] * np.kron(X, X) - chi[2] * np.kron(Y, Y) + chi[3] * np.kron(Z, Z)
	###########
	return (theta, chi)


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
	start = timer()
	theta_out = _convert.KrausToTheta(real_kraus, imag_kraus, nq, nkr)
	# print("theta_out\n{}".format(theta_out))
	theta_real = theta_out[ : (4**nq * 4**nq)].reshape([2, 2, 2, 2] * nq)
	theta_imag = theta_out[(4**nq * 4**nq) :].reshape([2, 2, 2, 2] * nq)
	theta = theta_real + 1j * theta_imag
	# print("Theta matrix computed in {} seconds.".format(timer() - start))
	##########
	# only for debugging purposes
	# (I,X,Y,Z) = (np.eye(2), np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]]))
	# print("chi[0,0] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(I, I))))
	# print("chi[0,1] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(I, X))))
	# print("chi[0,2] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(I, Y))))
	# print("chi[0,3] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(I, Z))))

	# print("chi[1,0] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(X, I))))
	# print("chi[1,1] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(X, X))))
	# print("chi[1,2] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(X, Y))))
	# print("chi[1,3] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(X, Z))))

	# print("chi[2,0] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(Y, I))))
	# print("chi[2,1] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(Y, X))))
	# print("chi[2,2] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(Y, Y))))
	# print("chi[2,3] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(Y, Z))))

	# print("chi[3,0] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(Z, I))))
	# print("chi[3,1] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(Z, X))))
	# print("chi[3,2] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(Z, Y))))
	# print("chi[3,3] = {}".format(np.trace(theta.reshape(4,4) @ np.kron(Z, Z))))
	##########
	#
	# Get Theta matrix from pure Python code
	# start = timer()
	# theta_mat_alt = KrausToTheta_Python(kraus) # Only for decoding purposes
	# print("Theta matrix from the Python code was computed in {} seconds.".format(timer() - start))
	# print("theta - theta_mat_alt\n{}".format(theta.reshape(4**nq, 4**nq) -theta_mat_alt.reshape(4**nq, 4**nq))) # Only for decoding purposes
	# print("IsClose: {}".format(np.allclose(theta, theta_mat_alt)))
	# print("Tr(theta) = {}".format(np.trace(theta_mat_alt))) # Only for decoding purposes
	return theta


def ThetaToChiElement(pauli_op_i, pauli_op_j, kraus_theta_chi_dict):
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
	theta_dict = [(supp_theta, theta) for (__, supp_theta, __, __, theta) in kraus_theta_chi_dict]

	# print("theta_dict\n{}".format([(supp, theta.shape) for (supp, theta) in theta_dict]))
	
	Pj = [((q,), PauliTensor(pauli_op_j[q, np.newaxis])) for q in range(nq)]
	PjT = [((q,), (-1)**(int(pauli_op_j[q] == 2)) * PauliTensor(pauli_op_j[q, np.newaxis])) for q in range(nq)]
	Pi = [((nq + q,), PauliTensor(pauli_op_i[q, np.newaxis])) for q in range(nq)]
	
	# print("Calling ContractTensorNetwork on\n\ttheta_dict sizes: {}".format([supp for (supp, __) in theta_dict + PjT + Pi]))
	(__, chi_elem) = ContractTensorNetwork(theta_dict + PjT + Pi, end_trace=1, verbose=False)
	# chi_elem = 0 # For debugging purposes.
	chi_elem /= 4**nq

	# print("chi[{}, {}] = {}".format(pauli_op_i, pauli_op_j, chi_elem))
	
	# print("Chi element of Pauli op {} = {}".format(pauli_op_i, chi_elem))
	if ((np.real(chi_elem) <= -1E-15) or (np.abs(np.imag(chi_elem)) >= 1E-15)):
		print("Pi\n{}\nPj\n{}".format(pauli_op_i, pauli_op_j))
		print("Chi = %g + i %g" % (np.real(chi_elem), np.imag(chi_elem)))
		exit(0)
	return np.real(chi_elem.item())


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
