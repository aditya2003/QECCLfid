import numpy as np
cimport numpy as np
from define.QECCLfid.utils import PauliTensor
from define.QECCLfid.tensor import TensorTranspose, TensorKron, TensorTrace, TraceDot
from define.QECCLfid.contract import ContractTensorNetwork
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


def KraussToTheta(np.ndarray[np.complex128, ndim=3] kraus):
	# Convert from the Kraus representation to the "Theta" representation.
	# The "Theta" matrix T of a CPTP map whose chi-matrix is X is defined as:
	# T_ij = \sum_(ij) [ X_ij (P_i o (P_j)^T) ]
	# Note that the chi matrix X can be defined using the Kraus matrices {K_k} in the following way
	# X_ij = \sum_k [ <P_i|K_k><K_k|P_j> ]
	# 	   = \sum_k [ Tr(P_i K_k) Tr((K_k)^\dag P_j)]
	# So we find that
	# T = \sum_(ij) [ \sum_k [ Tr(P_i K_k) Tr((K_k)^\dag P_j)] ] (P_i o (P_j)^T) ]
	# We will store T as a Tensor with dimension = (2 * number of qubits) and bond dimension = 4.
	cdef np.ndarray[np.complex128, ndim=3] Paulis = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=nb.complex128)
	
	cdef int nq = int(np.log2(kraus.shape[1]))
	
	cdef double transpose_sign = 0	
	cdef np.ndarray[np.complex128, ndim=2] chi = np.zeros((4**nq, 4**nq), dtype = np.complex128)
	
	cdef np.ndarray[np.complex128, ndim=4 * nq] theta = np.zeros(tuple([2, 2, 2, 2]*nq), dtype = np.complex128)
	cdef np.ndarray[np.complex128, ndim=4 * nq] PjoPi = np.zeros(tuple([2, 2, 2, 2]*nq), dtype = np.complex128)
	cdef np.ndarray[np.complex128, ndim=4 * nq] PjToPi = np.zeros(tuple([2, 2, 2, 2]*nq), dtype = np.complex128)

	cdef np.ndarray[np.int, ndim=1] supp_Kdag = np.arange(nq, dtype = np.int)
	cdef np.ndarray[np.int, ndim=1] supp_K = np.arange(nq, dtype = np.int) + nq
	cdef np.complex128_t tr_Pi_K = 0
	cdef np.complex128_t tr_Pi_K_dag = 0

	# loop variables.
	cdef int i, j, k, q, index, y_count

	# Preparing the Pauli operators.
	# click = timer()
	cdef np.ndarray[np.int, ndim=2] pauli_operators = np.zeros((4**nq, nq), dtype = np.int)	
	cdef np.ndarray[np.int, ndim=1] transpose_signs = 1
	for i in range(4**nq):
		index = i
		y_count = 0
		for q in range(nq):
			pauli_operators[i, nq - q - 1] = index % 4
			index = int(index//4)
			# Count the number of Ys' for computing the transpose.
			if (pauli_operators[i, q] == 2):
				y_count += 1
		transpose_signs[i] = (-1) ** y_count 
	
	# click = timer()
	for i in range(4**nq):
		Pi_tensor = [((q,), Paulis[pauli_operators[i, q], :, :]) for q in range(nq)]
		
		for j in range(4**nq):
			Pj_tensor = [((q,), Paulis[pauli_operators[j, q], :, :]) for q in range(nq)]
			#PjT_tensor = [(tuple(list(range(nq))), transpose_sign * pauli_tensors[j])]

			# click = timer()

			if (i <= j):
				for k in range(kraus.shape[0]):
					K = [(supp_K, np.reshape(kraus[k, :, :], tuple([2, 2]*nq)))]
					Kdag = [(supp_Kdag, np.reshape(np.conj(kraus[k, :, :].T), tuple([2, 2]*nq)))]
					
					(__, tr_Pi_K) = ContractTensorNetwork(K + Pi_tensor, end_trace=1)
					(__, tr_Pj_Kdag) = ContractTensorNetwork(Kdag + Pj_tensor, end_trace=1)
					
					chi[i, j] += tr_Pi_K * tr_Pj_Kdag
				
				chi[i, j] /= 4**nq
			
			else:
				chi[i, j] = np.conj(chi[j, i])
			
			if (i == j):
				if ((np.real(chi[i, j]) <= -1E-14) or (np.abs(np.imag(chi[i, j])) >= 1E-14)):
					print("Error: Chi[%d, %d] = %g + i %g" % (i, j, np.real(chi[i, j]), np.imag(chi[i, j])))
					exit(0)
			
			(__, PjoPi) = ContractTensorNetwork(Pj_tensor + Pi_tensor, end_trace=0)
			PjToPi = PjoPi * transpose_signs[j]
			theta += chi[i, j] * PjToPi
			
			# print("Chi[%d, %d] = %g + i %g was computed in %d seconds." % (i, j, np.real(chi[i, j]), np.imag(chi[i, j]), timer() - click))
		# print("----")
	
	# print("Theta matrix was computed in {} seconds.".format(timer() - click))
	return theta