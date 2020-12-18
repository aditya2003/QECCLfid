import numpy as np
from define.QECCLfid.utils import GetNQubitPauli, PauliTensor
from define.QECCLfid.tensor import TensorTranspose, TensorKron, TensorTrace, TraceDot
from define.QECCLfid.contract import ContractTensorNetwork
from timeit import default_timer as timer

def ThetaToChiElement(pauli_op_i, pauli_op_j, theta, supp_theta):
	# Convert from the Theta representation to the Chi representation.
	# The "Theta" matrix T of a CPTP map whose chi-matrix is X is defined as:
	# T_ij = \sum_(ij) [ X_ij (P_i o (P_j)^T) ]
	# So we find that
	# Chi_ij = Tr[ (P_i o (P_j)^T) T]
	# Note that [P_i o (P_j)^T] can be expressed as a product of the single qubit Pauli matrices
	# (P^(1)_i)_(r1,c1) (P^(1)_j)_(c(N+1),r(N+1)) x ... x (P^(N)_i)_(c(N),r(N)) (P^(N)_j)_(c(2N),r(2N))
	# We will store T as a Tensor with dimension = (2 * number of qubits) and bond dimension = 4.
	#click = timer()
	nq = pauli_op_i.size
	# print("nq = {}".format(nq))
	#Pi = PauliTensor(pauli_op_i)
	# print("Pi shape = {}".format(Pi.shape))
	#Pj = PauliTensor(pauli_op_j)
	# print("Pj shape = {}".format(Pj.shape))
	#PjT = TensorTranspose(Pj)
	# print("PjT shape = {}".format(PjT.shape))
	#print("Building Paulis took {} seconds".format(timer() - click))
	#click = timer()
	#PioPjT = np.reshape(TensorKron(Pi, PjT), tuple([4, 4] * nq))
	#print("TensorKron took {} seconds".format(timer() - click))
	# print("PioPjT shape = {}".format(PioPjT.shape))
	theta_reshaped = theta.reshape(*[4,4] * len(supp_theta))
	# print("theta_reshaped shape = {}".format(theta_reshaped.shape))
	click = timer()
	ops = [(supp_theta, theta_reshaped)]
	print("pauli_i\n{}\npauli_j\n{}".format(pauli_op_i, pauli_op_j))
	Pi = [((q,), PauliTensor(pauli_op_i[np.newaxis, q])) for q in range(nq)]
	Pj = [((q,), TensorTranspose(PauliTensor(pauli_op_j[np.newaxis, q]))) for q in range(nq)]
	# The below is missing a sign for Y^T = -Y.
	PioPjT = [((q,), np.reshape(PauliTensor(np.concatenate((pauli_op_i[np.newaxis, q], pauli_op_j[np.newaxis, q]))), [4, 4])) for q in range(nq)]
	print("Pi\n{}\nPj\n{}\nPioPjT\n{}".format(Pi, Pj, [op.shape for (sup, op) in PioPjT]))
	(__, chi_elem) = ContractTensorNetwork(ops + PioPjT, end_trace=1)/4**nq
	print("Chi element of Pauli op {} = {}".format(pauli_op_i, chi_elem))
	return chi_elem


def KraussToTheta(kraus):
	# Convert from the Kraus representation to the "Theta" representation.
	# The "Theta" matrix T of a CPTP map whose chi-matrix is X is defined as:
	# T_ij = \sum_(ij) [ X_ij (P_i o (P_j)^T) ]
	# Note that the chi matrix X can be defined using the Kraus matrices {K_k} in the following way
	# X_ij = \sum_k [ <P_i|K_k><K_k|P_j> ]
	# 	   = \sum_k [ Tr(P_i K_k) Tr((K_k)^\dag P_j)]
	# So we find that
	# T = \sum_(ij) [ \sum_k [ Tr(P_i K_k) Tr((K_k)^\dag P_j)] ] (P_i o (P_j)^T) ]
	# We will store T as a Tensor with dimension = (2 * number of qubits) and bond dimension = 4.
	nq = int(np.log2(kraus.shape[1]))
	theta = np.zeros(tuple([4, 4]*nq), dtype = np.complex128)
	for i in range(4**nq):
		for j in range(4**nq):
			Pi = PauliTensor(GetNQubitPauli(i, nq))
			Pj = PauliTensor(GetNQubitPauli(j, nq))
			PjT = TensorTranspose(Pj)
			PioPjT = np.reshape(TensorKron(Pi, PjT), tuple([4, 4] * nq))
			#print("Pi.shape = {}\nPi\n{}".format(Pi.shape, Pi))
			#print("Pj.shape = {}\nPj\n{}".format(Pj.shape, Pj))
			chi_ij = 0 + 0 * 1j
			for k in range(kraus.shape[0]):
				K = np.reshape(kraus[k, :, :], tuple([2, 2]*nq))
				Kdag = np.conj(TensorTranspose(K))
				#print("K.shape = {}\nK\n{}".format(K.shape, K))
				#print("TraceDot(K, Pi) = {}".format(TraceDot(K, Pi)))
				#print("Kdag.shape = {}\nKdag\n{}".format(Kdag.shape, Kdag))
				#print("Pj.shape = {}\nPj\n{}".format(Pj.shape, Pj))
				#print("TraceDot(Kdag, Pj) = {}".format(TraceDot(K, Pj)))
				chi_ij += TraceDot(Pi, K) * TraceDot(Pj, Kdag)
			chi_ij /= 4**nq
			#print("Chi[%d, %d] = %g + i %g" % (i, j, np.real(coeff), np.imag(coeff)))
			theta += chi_ij * PioPjT
	return theta


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
