import copy
import numpy as np
from define import globalvars as gv
from define.qcode import GetOperatorsForTLSIndex
from timeit import default_timer as timer
from define.QECCLfid.contract import ContractTensorNetwork
from define.QECCLfid.utils import Dagger, circular_shift, GetNQubitPauli, PauliTensor, ConvertToDecimal


def get_Pauli_tensor(Pauli):
	r"""
	Takes an n-qubit pauli as a list and converts it to the tensor form
	"""
	listOfPaulis = [gv.Pauli[i] for i in Pauli]
	if listOfPaulis:
		Pj = listOfPaulis[0]
		n_qubits = len(listOfPaulis)
		for i in range(1, n_qubits):
			Pj = np.tensordot(Pj, listOfPaulis[i], axes=0)
		indices_Pj = [i for i in range(0, len(Pj.shape), 2)] + [
			i for i in range(1, len(Pj.shape), 2)
		]
	else:
		raise ValueError(f"Invalid Pauli {Pauli} ")
	return np.transpose(Pj, indices_Pj)


def fix_index_after_tensor(tensor, indices_changed):
	r"""
	Tensor product alters the order of indices. This function helps reorder to fix them back.
	"""
	n = len(tensor.shape) - 1
	perm_list = list(range(len(tensor.shape)))
	n_changed = len(indices_changed)
	for i in range(len(indices_changed)):
		index = indices_changed[i]
		perm_list = circular_shift(perm_list, index, n - n_changed + i + 1, "right")
	return np.transpose(tensor, perm_list)

def get_kraus_conj(kraus, Pi, indices):
	# Compute the conjugation of a Pauli with a given Kraus operator.
	# Given K and P, compute K P K^dag.
	kraus_reshape_dims = [2] * (2 * int(np.log2(kraus.shape[0])))
	indices_Pi = indices[len(indices) // 2 :]
	indices_kraus = range(len(kraus_reshape_dims) // 2)
	Pi_int = np.tensordot(
		Pi,
		Dagger(kraus).reshape(kraus_reshape_dims),
		(indices_Pi, indices_kraus),
	)
	Pi_int = fix_index_after_tensor(Pi_int, indices_Pi)
	indices_Pi = indices[: len(indices) // 2]
	indices_kraus = range(len(kraus_reshape_dims))[
		len(kraus_reshape_dims) // 2 :
	]
	Pi_int = np.tensordot(
		Pi_int,
		kraus.reshape(kraus_reshape_dims),
		(indices_Pi, indices_kraus),
	)
	Pi_int = fix_index_after_tensor(Pi_int, indices_Pi)
	return Pi_int


def KrausToPTM(kraus):
	# Compute the PTM for a channel whose Kraus operators are given.
	# Given {K}, compute G given by G_ij = \sum_k Tr[ K_k P_i (K_k)^dag Pj ].
	# print("Function: KrausToPTM")

	nq = int(np.log2(kraus.shape[1]))
	ptm = np.zeros((4**nq, 4**nq), dtype = np.double)
	
	# Preparing the Pauli operators.
	click = timer()
	pauli_tensors = np.zeros(tuple([4**nq] + [2, 2]*nq), dtype = np.complex128)
	
	# print("Shape of pauli_tensors: {}".format(pauli_tensors.shape))
	
	for i in range(4**nq):
		pauli_op_i = GetNQubitPauli(i, nq)
		Pi = [((q,), PauliTensor(pauli_op_i[q, np.newaxis])) for q in range(nq)]
		(__, pauli_tensors[i]) = ContractTensorNetwork(Pi)
	
	# print("Preparing Pauli tensors took {} seconds.".format(timer() - click))
	click = timer()

	for i in range(4**nq):
		Pi_tensor = [(tuple(list(range(nq))), pauli_tensors[i])]
		for j in range(4**nq):
			Pj_tensor = [(tuple(list(range(nq))), pauli_tensors[j])]
			for k in range(kraus.shape[0]):
				supp_K = tuple(list(range(nq)))
				K = [(supp_K, np.reshape(kraus[k, :, :], tuple([2, 2]*nq)))]
				Kdag = [(supp_K, np.reshape(Dagger(kraus[k, :, :]), tuple([2, 2]*nq)))]
				(__, innerprod) = ContractTensorNetwork(K + Pi_tensor + Kdag + Pj_tensor, end_trace = 1)
				ptm[i, j] += np.real(innerprod)/2**nq

	ptm_tensor = ptm.reshape(tuple([2, 2, 2, 2]*nq))
	# print("PTM tensor of dimensions {} was computed in {} seconds.".format(ptm_tensor.shape, timer() - click))
	return ptm_tensor

def PTMAdjointTest(kraus, ptm):
	# Check if the i,j element of the channel is j,i element of the adjoint channel.
	nq = int(np.log2(kraus.shape[1]))
	kraus_adj = np.zeros_like(kraus)
	for k in range(kraus.shape[0]):
		kraus_adj[k, :, :] = Dagger(kraus[k, :, :])
	click = timer()
	ptm_adj = KrausToPTM(kraus_adj)
	# print("Adjoint PTM was constructed in %d seconds." % (timer() - click))
	# print("ptm - ptm_adj: {}".format(np.sum(np.abs(ptm.reshape(4**nq, 4**nq) - ptm_adj.reshape(4**nq, 4**nq).T))))
	return np.allclose(ptm.reshape(4**nq, 4**nq), ptm_adj.reshape(4**nq, 4**nq).T)


def ExtractPTMElement(pauli_op_i, pauli_op_j, ptm, supp_ptm):
	# Compute the PTM element corresponding to a Pair of Pauli operators given a PTM matrix.
	# Given G, we want to compute G_ij = << Pi | G | Pj >> where |P>> is the vector in the Pauli basis corresponding to the Pauli matrix P.
	# In other words, |Pi>> is an indicator vector with a "1" at position x and 0 elsewhere.
	# Here "x" is the position of pauli_op_i in the lexicographic ordering, i.e., if pauli_op_i = (p(0) p(1) ... p(n-1)), then
	# x = 4**(n-1) * p(n-1) + ... + 4 * p1 + p0
	# We want << i(1) i(2) ... i(2n) | A | j(1) j(2) ... j(2n) >>, where i(k) in {0, 1}.
	# We will reshape A to: [2, 2, 2, 2] * supp(A). To specify an element of A, we need two row and two column bits, per support qubit.
	# row_indices = []
	# col_indices = []
	# result = 1
	# for q in [1, n]:
	# 	if supp(A) doesn't have q:
	# 		result *= delta(i(2q), j(2q)) * delta(i(2q + 1), j(2q + 1))
	# if result != 0:
	# 	for q in supp(A):
	#		row_indices.extend([i(2q), i(2q + 1)])
	#		col_indices.extend([j(2q), j(2q + 1)])
	# 	ptm_ij = A[*[row_indices + col_indices]]
	click = timer()
	nq = pauli_op_i.shape[0]
	
	Pi_binary = list(map(int, np.binary_repr(ConvertToDecimal(pauli_op_i, 4), 2 * nq)))
	# print("Pi_binary\n{}".format(Pi_binary))
	Pj_binary = list(map(int, np.binary_repr(ConvertToDecimal(pauli_op_j, 4), 2 * nq)))
	# print("Pj_binary\n{}".format(Pj_binary))
	trivial_action = 1
	for q in range(nq):
		if q not in supp_ptm:
			trivial_action *= int(Pi_binary[2 * q] == Pj_binary[2 * q]) * int(Pi_binary[2 * q + 1] == Pj_binary[2 * q + 1])
	
	row_indices = []
	col_indices = []
	if (trivial_action != 0):
		for i in range(len(supp_ptm)//2):
			q = supp_ptm[i]
			row_indices.extend([Pi_binary[2 * q], Pi_binary[2 * q + 1]])
			col_indices.extend([Pj_binary[2 * q], Pj_binary[2 * q + 1]])
		ptm_ij = ptm[tuple(row_indices + col_indices)]
	else:
		ptm_ij = 0
	"""
	click = timer()
	nq = pauli_op_i.shape[0]
	support_pauli = tuple([q for q in range(nq)] + [(nq + q) for q in range(nq)])
	Pi_indicator = np.zeros((1, 4**nq), dtype = np.double)
	Pi_indicator[0, ConvertToDecimal(pauli_op_i, 4)] = 1
	Pi_tensor = [(support_pauli, Pi_indicator.reshape([1, 2] * 2*nq))]
	Pj_indicator = np.zeros((4**nq, 1), dtype = np.double)
	Pj_indicator[ConvertToDecimal(pauli_op_j, 4), 0] = 1
	Pj_tensor = [(support_pauli, Pj_indicator.reshape([2, 1] * 2*nq))]
	ptm_tensor = [(supp_ptm, ptm)]
	print("Preparing the indicator vectors take {}".format(timer() - click))
	(__, ptm_ij) = ContractTensorNetwork(Pi_tensor + ptm_tensor + Pj_tensor, end_trace=1)
	"""
	return ptm_ij


def ConstructPTM(qcode, kraus_dict):
	r"""
	Generates LS part of the process matrix of the n-qubit Pauli channel.
	"""
	nstabs = 2 ** (qcode.N - qcode.K)
	nlogs = 4 ** qcode.K
	n_maps = len(list(kraus_dict.keys()))
	ptm_dict = [None for __ in range(n_maps)]
	for m in range(n_maps):
		(support, kraus) = kraus_dict[m]
		# print("support = {}".format(support))
		ptm = KrausToPTM(np.array(kraus))
		if (PTMAdjointTest(kraus, ptm) == 0):
			print("PTM adjoint test failed for map {}.".format(m + 1))
			exit(0)
		ptm_support = tuple([q for q in support] + [(qcode.N + q) for q in support])
		ptm_dict[m] = (ptm_support, ptm)
		# print("Map {}:\n{} qubit ptm supported on {} has shape {}.".format(m + 1, len(support), ptm_support, ptm.shape))
	(supp_ptm, ptm_contracted) = ContractTensorNetwork(ptm_dict)
	# print("ptm_contracted supported on {} has shape: {}.".format(supp_ptm, ptm_contracted.shape))
	
	# Derive the full PTM
	# noiseless_qubits = np.setdiff1d([q for q in range(qcode.N)])
	# ContractTensorNetwork()

	(ls_ops, phases) = GetOperatorsForTLSIndex(qcode, range(nstabs * nlogs))
	process = np.zeros((nlogs * nstabs, nlogs * nstabs), dtype=np.double)
	for i in range(nlogs * nstabs):
		for j in range(nlogs * nstabs):
			click = timer()
			process[i, j] = np.real(phases[i] * phases[j] * ExtractPTMElement(ls_ops[i, :], ls_ops[j, :], ptm_contracted, supp_ptm))
			# print("PTM[{}, {}] = {}, was done in {} seconds.".format(i, j, process[i, j], timer() - click))
	return process


def PTM_Element(krausdict, Pi, Pjlist, n_qubits,phasei=None,phasej=None):
	r"""
	Assumes Paulis Pi,Pj to be a tensor on n_qubits
	Calculates Tr(Pj Eps(Pi)) for each Pj in Pjlist
	Kraus dict has the format ("support": list of kraus ops on the support)
	Assumes qubits
	Assumes kraus ops to be square with dim 2**(number of qubits in support)
	"""
	#     Pres stores the addition of all kraus applications
	#     Pi_term stores result of individual kraus applications to Pi
	if phasei is None:
		phasei = 1
	if phasej is None:
		phasej = np.ones(len(Pjlist))
	Pres = np.zeros_like(Pi)
	for key, (support, krausList) in krausdict.items():
		indices = support + tuple(map(lambda x: x + n_qubits, support))
		if len(indices) > 0:
			Pres = np.sum([get_kraus_conj(kraus, Pi, indices) for kraus in krausList], axis=0)
		else:
			Pres = Pi * np.sum([np.power(np.abs(kraus), 2) for kraus in krausList], axis=0)
		Pi = copy.deepcopy(Pres)
	# take dot product with Pj and trace
	trace_vals = np.zeros(len(Pjlist), dtype=np.double)
	indices_Pi = list(range(len(Pi.shape) // 2))
	indices_Pj = list(range(len(Pi.shape) // 2, len(Pi.shape)))
	trace_vals = np.array([np.real(get_trace(Pres, Pjlist[i], indices_Pi, indices_Pj, phase_A=phasei, phase_B=phasej[i]))/(2**n_qubits) for i in range(len(Pjlist))], dtype=np.double)
	# print("trace_vals = {}".format(np.count_nonzero(trace_vals)))
	return trace_vals


def get_trace(A, B, indices_A, indices_B, phase_A=1, phase_B=1):
	# Compute the Trace of [ A . B ] for A and B of similar dimensions.
	A_times_B = np.tensordot(A, B, (indices_A, indices_B))
	# Take trace
	einsum_inds = list(range(len(A_times_B.shape) // 2)) + list(
		range(len(A_times_B.shape) // 2)
	)
	trace = np.einsum(A_times_B, einsum_inds) * phase_A * phase_B
	return trace
