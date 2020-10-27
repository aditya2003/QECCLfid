import numpy as np
from scipy import linalg as linalg
import random
import scipy as sc
import sys
from define.QECCLfid.multi_qubit_tests import test_get_PTMelem_ij
from define.QECCLfid.utils import extend_gate, Dot, Kron, Dagger, circular_shift
from define.QECCLfid.ptm import get_PTMelem_ij, get_Pauli_tensor, fix_index_after_tensor
from define import randchans as rc
from define import qcode as qc
from define import globalvars as gv


def get_Chielem_ii(krausdict, Pilist, n_qubits):
	r"""
	Calculates the diagonal entry in chi matrix corresponding to each Pauli in Pilist
	Assumes each Pauli in list of Paulis Pilist to be a tensor on n_qubits
	Calculates chi_ii = sum_k |<Pi, A_k>|^2
	where A_k is thr Kraus operator and Pi is the Pauli operator
	Kraus dict has the format ("support": list of kraus ops on the support)
	Assumes qubits
	Assumes kraus ops to be square with dim 2**(number of qubits in support)
	"""
	#     Pres stores the addition of all kraus applications
	#     Pi_term stores result of individual kraus applications to Pi
	chi = np.zeros(len(Pilist), dtype=np.double)
	for i in range(len(Pilist)):
		Pi = get_Pauli_tensor(Pilist[i])
		for key, (support, krausList) in krausdict.items():
			indices = support + tuple(map(lambda x: x + n_qubits, support))
			for kraus in krausList:
				if len(indices) > 0:
					kraus_reshape_dims = [2] * (2 * int(np.log2(kraus.shape[0])))
					indices_Pi = indices[len(indices) // 2 :]
					indices_kraus = range(len(kraus_reshape_dims) // 2)
					#                 Pi_term = np.tensordot(Pi,Dagger(kraus).reshape(kraus_reshape_dims),(indices_Pi,indices_kraus))
					Pi_term = np.tensordot(
						Pi,
						kraus.reshape(kraus_reshape_dims),
						(indices_Pi, indices_kraus),
					)
					Pi_term = fix_index_after_tensor(Pi_term, indices_Pi)
					# Take trace and absolute value
					einsum_inds = list(range(len(Pi_term.shape) // 2)) + list(
						range(len(Pi_term.shape) // 2)
					)
					chi[i] += (
						np.power(np.abs(np.einsum(Pi_term, einsum_inds)), 2)
						/ 4 ** n_qubits
					)
				else:
					# Take trace and absolute value
					if i == 0:
						chi[i] += np.abs(kraus) ** 2
					else:
						chi[i] += 0
	return chi


def get_kraus_ising(J, mu, qcode):
	r"""
	Sub-routine to prepare the dictionary for errors arising due to Ising type interaction
	H = - J \sum_{i} Z_i Z_{i+1} - mu \sum_{i} Z_i
	Returns :
	dict[key] = (support,krauslist)
	where key = number associated to the operation applied (not significant)
	support = tuple describing which qubits the kraus ops act on
	krauslist = krauss ops acting on support
	"""
	ZZ = np.kron(gv.Pauli[3], gv.Pauli[3])
	if qcode.interaction_graph is None:
		# Asssume nearest neighbour in numerically sorted order
		qcode.interaction_graph = np.array([(i,i+1) for i in range(qcode.N-1)],dtype=np.int8)
	Ham = np.zeros(2**qcode.N, dtype = np.double)
	for (i,j) in qcode.interaction_graph :
		Ham = Ham + J * extend_gate([i,j], ZZ, np.arange(qcode.N, dtype=np.int))
	if mu > 0:
		for i in range(qcode.N):
			Ham = Ham + mu * extend_gate([i], gv.Pauli[3], np.arange(qcode.N, dtype=np.int))
	# Ham = np.random.rand(2**qcode.N, 2**qcode.N) + 1j * np.random.rand(2**qcode.N, 2**qcode.N)
	# Ham = Ham + Ham.conj().T
	# Ham = Ham/np.linalg.norm(Ham)
	# print("||H|| = {}".format(np.linalg.norm(Ham)))
	# kraus = rc.RandomUnitary(J, 2**qcode.N, "exp")
	kraus = linalg.expm(-1j * Ham)
	print("Unitarity of Kraus\n{}".format(np.linalg.norm(np.dot(kraus, kraus.conj().T) - np.eye(kraus.shape[0]))))
	kraus_dict = {0:(tuple(range(qcode.N)), [kraus])}
	return kraus_dict


def get_kraus_random(p_error, rotation_angle, qcode, w_thresh=3):
	r"""
	Sub-routine to prepare the dictionary for error eps = sum of unitary errors
	Generates kraus as random multi-qubit unitary operators rotated by rotation_angle
	Probability associated with each weight-k error scales exponentially p_error^k (except weights <= w_thresh)
	Returns :
	dict[key] = (support,krauslist)
	where key = number associated to the operation applied (not significant)
	support = tuple describing which qubits the kraus ops act on
	krauslist = krauss ops acting on support
	"""
	kraus_dict = {}

	kraus_count = 0
	norm_coeff = np.zeros(qcode.N + 1, dtype=np.double)
	for n_q in range(qcode.N, qcode.N + 1):
		if n_q == 0:
			p_q = 1 - p_error
		elif n_q <= w_thresh:
			p_q = np.power(0.1, n_q - 1) * p_error
		else:
			p_q = np.power(p_error, n_q)
		# Number of unitaries of weight n_q is max(1,qcode.N-n_q-1)
		if n_q == 0:
			nops_q = 1
		else:
			nops_q = 1 # For debugging.
			# nops_q = max(1, (qcode.N + n_q) // 2) # Aditya's version
			# nops_q = sc.special.comb(qcode.N, n_q, exact=True) * n_q
		norm_coeff[n_q] += p_q
		for __ in range(nops_q):
			support = tuple(sorted((random.sample(range(qcode.N), n_q))))
			if n_q == 0:
				rand_unitary = 1.0
			else:
				rand_unitary = rc.RandomUnitary(
					rotation_angle, 2 ** n_q, method="exp"
				) # Aditya version
				# rand_unitary = rc.RandomUnitary(
				#     rotation_angle/(n_q), 2 ** n_q, method="exp"
				# )
			kraus_dict[kraus_count] = (support, [rand_unitary * 1 / np.sqrt(nops_q)])
			kraus_count += 1

	norm_coeff /= np.sum(norm_coeff)
	# print("norm_coeff = {}".format(norm_coeff))
	# Renormalize kraus ops
	for key, (support, krauslist) in kraus_dict.items():
		for k in range(len(krauslist)):
			# print("k = {}".format(k))
			kraus_dict[key][1][k] *= np.sqrt(norm_coeff[len(support)])
	return kraus_dict


def get_process_correlated(qcode, kraus_dict):
	r"""
	Generates LS part of the process matrix for Eps = sum of unitary errors
	p_error^k is the probability associated to a k-qubit unitary (except weight <= w_thresh)
	rotation_angle is the angle used for each U = exp(i*rotation_agnle*H)
	return linearized Pauli transfer matrix matrix in LS ordering for T=0
	"""
	nstabs = 2 ** (qcode.N - qcode.K)
	nlogs = 4 ** qcode.K
	ops = qc.GetOperatorsForTLSIndex(qcode, range(nstabs * nlogs))
	ops_tensor = list(map(get_Pauli_tensor, ops))
	process = np.zeros(nstabs * nstabs * nlogs * nlogs, dtype=np.double)
	for i in range(len(ops_tensor)):
		process[i * nstabs * nlogs : (i + 1) * nstabs * nlogs] = get_PTMelem_ij(
			kraus_dict, ops_tensor[i], ops_tensor, qcode.N
		)
		# print("Test for {}\n{}".format(i, test_get_PTMelem_ij(kraus_dict, ops_tensor[i], ops_tensor, qcode.N)))
	return process


def get_process_diagLST(qcode, kraus_dict):
	r"""
	Generates diagonal of the process matrix in LST ordering for Eps = sum of unitary errors
	p_error^k is the probability associated to a k-qubit unitary (except weight<= w_thresh)
	rotation_angle is the angle used for each U = exp(i*rotation_agnle*H)
	return linearized diagonal of the process matrix in LST ordering
	"""
	nstabs = 2 ** (qcode.N - qcode.K)
	nlogs = 4 ** qcode.K
	diag_process = np.zeros(nstabs * nstabs * nlogs, dtype=np.double)
	ops = qc.GetOperatorsForLSTIndex(qcode, range(nstabs * nstabs * nlogs))
	for i in range(len(diag_process)):
		op_tensor = get_Pauli_tensor(ops[i])
		diag_process[i] = get_PTMelem_ij(kraus_dict, op_tensor, [op_tensor], qcode.N)[0]
	return diag_process


def get_chi_diagLST(qcode, kraus_dict):
	r"""
	Generates diagonal of the chi matrix in LST ordering for Eps = sum of unitary errors
	p_error^k is the probability associated to a k-qubit unitary (except weight <= w_thresh)
	rotation_angle is the angle used for each U = exp(i*rotation_agnle*H)
	return linearized diagonal of the chi matrix in LST ordering
	"""
	nstabs = 2 ** (qcode.N - qcode.K)
	nlogs = 4 ** qcode.K
	ops = qc.GetOperatorsForLSTIndex(qcode, range(nstabs * nstabs * nlogs))
	chi = get_Chielem_ii(kraus_dict, ops, qcode.N)
	print("Sum of chi = {}, infid = {}\nElements of chi\n{}".format(np.sum(chi), 1 - chi[0], np.sort(chi)[::-1]))
	return chi

def get_process_chi(qcode, method = "random", *params):
	nstabs = 2 ** (qcode.N - qcode.K)
	nlogs = 4 ** qcode.K
	if method == "random":
		p_error,rotation_angle,w_thresh = params[:3]
		kraus_dict = get_kraus_random(p_error, rotation_angle, qcode, w_thresh)
	elif method == "ising":
		J, mu = params[:2]
		kraus_dict = get_kraus_ising(J, mu, qcode)
	chi = get_chi_diagLST(qcode, kraus_dict)
	process = get_process_correlated(qcode, kraus_dict)

	# Check if the i,j element of the channel is j,i element of the adjoint channel.
	# for key, (support, krauslist) in kraus_dict.items():
	# 	for k in range(len(krauslist)):
	# 		kraus_dict[key][1][k] = Dagger(kraus_dict[key][1][k])
	# process_adj = get_process_correlated(qcode, kraus_dict)
	# print("process - process_adj: {}".format(np.allclose(process.reshape(256, 256), process_adj.reshape(256, 256).T)))

	print("Process[0] = {}".format(process[0]))
	return (process, chi)
