import numba as nb
import numpy as np
from define.QECCLfid.utils import GetNQubitPauli, PauliTensor
from define.QECCLfid.tensor import TensorTranspose, TensorKron, TensorTrace, TraceDot
from define.QECCLfid.contract import ContractTensorNetwork, TraceNetwork
from timeit import default_timer as timer


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


@nb.guvectorize(["void(complex128[:, :, :], uint16, uint16, complex128[:])"], "(k,n,n),(),()->()")
def IndividualChiElement(kraus, pi, pj, chi_element):
	# Compute the contribution to an element of the chi matrix, from a Kraus operator.
	# In other words, compute [ Tr(P_i K) Tr((K)^\dag P_j) ].
	Paulis = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=nb.complex128)
	nq = int(np.log2(kraus.shape[1]))		
	supports = np.zeros((nq + 1, nq + 1), dtype = nb.uint16)
	operators = np.zeros((4 ** nq + nq * 4), dtype = nb.complex128)
	bond_dimension = 2

	chi_element[0] = 0 + 0 * 1j
	
	for k in range(kraus.shape[0]):
		
		pauli_op_i = GetNQubitPauli(pi, nq)
		Pi_tensor = [((q,), Paulis[pauli_op_i[q], :, :]) for q in range(nq)]

		pauli_op_j = GetNQubitPauli(pj, nq)
		Pj_tensor = [((q,), Paulis[pauli_op_j[q], :, :]) for q in range(nq)]

		# print("Pi_tensor\n{}\nPj_tensor\n{}".format(Pi_tensor, Pj_tensor))
		"""
		supp_K = tuple([q for q in range(nq)])
		K = [(supp_K, np.reshape(kraus[k, :, :], tuple([2, 2]*nq)))]
		
		supp_Kdag = tuple([q for q in range(nq)])
		Kdag = [(supp_Kdag, np.reshape(np.conj(kraus[k, :, :].T), tuple([2, 2]*nq)))]
		
		(__, tr_Pi_K) = ContractTensorNetwork(K + Pi_tensor, end_trace=1)
		(__, tr_Pj_Kdag) = ContractTensorNetwork(Kdag + Pj_tensor, end_trace=1)
		"""

		# form numpy arrays with supports and the respective operators.
		supports[0, 0] = nq
		supports[0, 1:] = np.arange(nq, dtype = nb.uint16)
		supports[1:, 0] = 1
		supports[1:, 1] = np.arange(nq, dtype = nb.uint16)
		
		# Compute Tr[ K Pi ]
		start = kraus[k, :, :].size
		operators[ : start] = np.ravel(kraus[k, :, :])
		for q in range(nq):
			end = start + bond_dimension ** (2 * supports[1 + q, 0])
			operators[start : end] = np.ravel(Paulis[pauli_op_i[q], :, :])
			end = start
		tr_Pi_K = TraceNetwork(supports, bond_dimension, operators)
		
		# Compute Tr[ Kdag Pj ]
		start = kraus[k, :, :].size
		operators[ : start] = np.ravel(np.conj(kraus[k, :, :].T))
		for q in range(nq):
			end = start + bond_dimension ** (2 * supports[1 + q, 0])
			operators[start : end] = np.ravel(Paulis[pauli_op_j[q], :, :])
			end = start
		tr_Pj_Kdag = TraceNetwork(supports, bond_dimension, operators)

		# Compute Tr(P_i K) Tr((K)^\dag P_j)
		chi_element[0] += tr_Pi_K * tr_Pj_Kdag
	


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
	chi = np.zeros((4**nq, 4**nq), dtype = np.complex128)
	theta = np.zeros(tuple([2, 2, 2, 2]*nq), dtype = np.complex128)
	
	# Preparing the Pauli operators.
	# click = timer()
	pauli_tensors = np.zeros(tuple([4**nq] + [2, 2]*nq), dtype = np.complex128)	
	for i in range(4**nq):
		pauli_op_i = GetNQubitPauli(i, nq)
		Pi = [((q,), PauliTensor(pauli_op_i[q, np.newaxis])) for q in range(nq)]
		(__, pauli_tensors[i]) = ContractTensorNetwork(Pi)
	# print("Preparing Pauli tensors took {} seconds.".format(timer() - click))
	
	# click = timer()

	# contributions = np.zeros(kraus.shape[0], dtype = np.complex128)
	for i in range(4**nq):
		Pi_tensor = [(tuple([(nq + q) for q in range(nq)]), pauli_tensors[i])]
		
		for j in range(4**nq):
			pauli_op_j = GetNQubitPauli(j, nq)
			# Pj_tensor = [(tuple(list(range(nq))), pauli_tensors[j])]
			transpose_sign = (-1) ** np.count_nonzero(pauli_op_j == 2)
			PjT_tensor = [(tuple(list(range(nq))), transpose_sign * pauli_tensors[j])]

			click = timer()

			if (i <= j):
				chi[i, j] = IndividualChiElement(kraus, i, j) / (4 ** nq)
			else:
				chi[i, j] = np.conj(chi[j, i])
			
			if (i == j):
				if ((np.real(chi[i, j]) <= -1E-14) or (np.abs(np.imag(chi[i, j])) >= 1E-14)):
					print("Error: Chi[%d, %d] = %g + i %g" % (i, j, np.real(chi[i, j]), np.imag(chi[i, j])))
					exit(0)
			
			(__, PjToPi) = ContractTensorNetwork(PjT_tensor + Pi_tensor, end_trace=0)
			theta += chi[i, j] * PjToPi
			
			print("Chi[%d, %d] = %g + i %g was computed in %g seconds." % (i, j, np.real(chi[i, j]), np.imag(chi[i, j]), timer() - click))
		# print("----")
	
	# print("Theta matrix was computed in {} seconds.".format(timer() - click))
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
