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
	# supp_theta_tensor = np.concatenate((np.array(list(supp_theta), dtype = np.int), nq + np.array(list(supp_theta), dtype = np.int)))
	supp_theta_tensor = [q for q in supp_theta] + [(nq + q) for q in supp_theta]
	# theta_reshaped = theta.reshape([2, 2, 2, 2] * len(supp_theta))
	# print("theta_reshaped supported on {} has shape {}.".format(supp_theta_tensor, theta_reshaped.shape))
	# exit(0);
	# print("pauli_i\n{}\npauli_j\n{}".format(pauli_op_i, pauli_op_j))
	click = timer()
	ops = [(supp_theta_tensor, theta)]
	
	"""
	Pj = list(map(lambda op: TensorTranspose(PauliTensor(op)), pauli_op_j[:, np.newaxis]))
	Pi = list(map(lambda op: PauliTensor(op), pauli_op_i[:, np.newaxis]))
	paulis = Pj + Pi
	PjToPi = [None for __ in range(nq)]
	for i in range(nq):
		PjToPi[i] = ((i,), np.kron(paulis[2*i], paulis[2*i + 1]))
	"""
	# Pi = PauliTensor(pauli_op_i)
	# Pj = PauliTensor(pauli_op_j)
	# PjT = TensorTranspose(Pj)
	# PjToPi = [(tuple(list(range(nq))), np.reshape(TensorKron(PjT, Pi), tuple([4, 4] * nq)))]

	Pj = [((q,), PauliTensor(pauli_op_j[q, np.newaxis])) for q in range(nq)]
	PjT = [((q,), (-1)**(int(pauli_op_j[q] == 2)) * PauliTensor(pauli_op_j[q, np.newaxis])) for q in range(nq)]
	Pi = [((nq + q,), PauliTensor(pauli_op_i[q, np.newaxis])) for q in range(nq)]

	# (__, PjToPi) = ContractTensorNetwork(PjT + Pi, end_trace=0)
	# PjToPi_tensor = [(tuple(list(range(nq))), PjToPi.reshape([4, 4] * nq))]

	(__, chi_elem) = ContractTensorNetwork(ops + PjT + Pi, end_trace=1)
	chi_elem /= 4**nq
	# print("Chi element of Pauli op {} = {}".format(pauli_op_i, chi_elem))
	if ((np.real(chi_elem) <= -1E-15) or (np.abs(np.imag(chi_elem)) >= 1E-15)):
		print("Chi[%d, %d] = %g + i %g" % (i, j, np.real(chi_elem), np.imag(chi_elem)))
		exit(0)
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
	theta = np.zeros(tuple([2, 2, 2, 2]*nq), dtype = np.complex128)
	probs = 0
	# Preparing the Pauli operators.
	click = timer()
	pauli_tensors = np.zeros(tuple([4**nq] + [2, 2]*nq), dtype = np.complex128)
	print("Shape of pauli_tensors: {}".format(pauli_tensors.shape))
	for i in range(4**nq):
		pauli_op_i = GetNQubitPauli(i, nq)
		Pi = [((q,), PauliTensor(pauli_op_i[q, np.newaxis])) for q in range(nq)]
		(__, pauli_tensors[i]) = ContractTensorNetwork(Pi)
	print("Preparing Pauli tensors took {} seconds.".format(timer() - click))
	click = timer()
	for i in range(4**nq):
		#print("Pi: {}".format(GetNQubitPauli(i, nq)))
		#pauli_op_i = GetNQubitPauli(i, nq)
		#Pi = [((nq + q,), PauliTensor(pauli_op_i[q, np.newaxis])) for q in range(nq)]
		#Pi_tensor = [ContractTensorNetwork(Pi)]
		Pi_tensor = [(tuple([(nq + q) for q in range(nq)]), pauli_tensors[i])]
		for j in range(4**nq):
			#print("Pj: {}".format(GetNQubitPauli(j, nq)))
			# Pi = PauliTensor(GetNQubitPauli(i, nq))
			# Pj = PauliTensor(GetNQubitPauli(j, nq))
			# PjT = TensorTranspose(Pj)
			# PjToPi = np.reshape(TensorKron(PjT, Pi), tuple([4, 4] * nq))
			pauli_op_j = GetNQubitPauli(j, nq)
			#Pj = [((q,), PauliTensor(pauli_op_j[q, np.newaxis])) for q in range(nq)]
			#Pj_tensor = [ContractTensorNetwork(Pj)]
			Pj_tensor = [(tuple(list(range(nq))), pauli_tensors[j])]
			transpose_sign = (-1) ** np.count_nonzero(pauli_op_j == 2)
			PjT_tensor = [(tuple(list(range(nq))), transpose_sign * pauli_tensors[j])]
			#PjT = [((q,), (-1)**(int(pauli_op_j[q] == 2)) * PauliTensor(pauli_op_j[q, np.newaxis])) for q in range(nq)]
			
			# print("Pi\n{}".format(Pi))
			# print("PjT\n{}".format(PjT))
			chi_ij = 0 + 0 * 1j
			for k in range(kraus.shape[0]):
				# K = np.reshape(kraus[k, :, :], tuple([2, 2]*nq))
				# Kdag = np.conj(TensorTranspose(K))
				supp_K = tuple(list(nq + np.arange(nq, dtype = np.int)))
				K = [(supp_K, np.reshape(kraus[k, :, :], tuple([2, 2]*nq)))]
				# print("K\n{}".format(K))
				supp_Kdag = tuple(list(np.arange(nq, dtype = np.int)))
				Kdag = [(supp_Kdag, np.reshape(np.conj(kraus[k, :, :].T), tuple([2, 2]*nq)))]
				# print("Kdag\n{}".format(Kdag))
				(__, tr_Pi_K) = ContractTensorNetwork(K + Pi_tensor, end_trace=1)
				(__, tr_Pj_Kdag) = ContractTensorNetwork(Kdag + Pj_tensor, end_trace=1)
				chi_ij += tr_Pi_K * tr_Pj_Kdag
				#print("TraceDot(K, Pi) = {}".format(TraceDot(K, Pi)))
				#print("Kdag.shape = {}\nKdag\n{}".format(Kdag.shape, Kdag))
				#print("Pj.shape = {}\nPj\n{}".format(Pj.shape, Pj))
				#print("TraceDot(Kdag, Pj) = {}".format(TraceDot(K, Pj)))
				# chi_ij += TraceDot(Pi, K) * TraceDot(Pj, Kdag)
			chi_ij /= 4**nq
			if (i == j):
				if ((np.real(chi_ij) <= -1E-15) or (np.abs(np.imag(chi_ij)) >= 1E-15)):
					print("Chi[%d, %d] = %g + i %g" % (i, j, np.real(chi_ij), np.imag(chi_ij)))
					exit(0)
				else:
					probs += np.real(chi_ij)
			(__, PjToPi) = ContractTensorNetwork(PjT_tensor + Pi_tensor, end_trace=0)
			theta += chi_ij * PjToPi
		# print("----")
	print("Sum of chi = {}.".format(probs))
	print("Theta matrix was computed in {} seconds.".format(timer() - click))
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
