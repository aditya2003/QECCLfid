import os
import copy
import numpy as np
import ctypes as ct
# import more_itertools as mit
from tqdm import tqdm
from pqdm.processes import pqdm
from numpy.ctypeslib import ndpointer
import multiprocessing as mp
from define import globalvars as gv
from timeit import default_timer as timer
from define.qcode import GetOperatorsForTLSIndex
from define.QECCLfid.contract import ContractTensorNetwork
from define.QECCLfid.utils import Dagger, circular_shift, GetNQubitPauli, PauliTensor, ConvertToDecimal

def KrausToPTM_Python(kraus):
	'''
	Convert from Kraus to PTM.
	PTM_ij = Tr( E(P_i) . P_j )
	       = sum_k ( E_k P_i E^dag_k P_j )
	'''
	dim = kraus.shape[1]
	nq = int(np.ceil(np.log2(dim)))
	npauli = np.power(4, nq, dtype = np.int64)

	paulis = np.zeros((npauli, dim, dim), dtype = np.complex128)
	for p in range(npauli):
		paulis[p, :, :] = PauliTensor(GetNQubitPauli(p, nq)).reshape(dim, dim)

	ptm = np.real(einsumt('klm,imn,kpn,jpl->ij', kraus, paulis, np.conj(kraus), paulis)) / dim
	return ptm


def KrausToPTM(kraus):
	# Convert from the Kraus representation to the PTM.
	# This is a wrapper for the KrausToPTM function in convert.so.
	nkr = kraus.shape[0]
	dim = kraus.shape[1]
	nq = int(np.ceil(np.log2(dim)))
	
	real_kraus = np.real(kraus).reshape(-1).astype(np.float64)
	imag_kraus = np.imag(kraus).reshape(-1).astype(np.float64)
	
	_convert = ct.cdll.LoadLibrary(os.path.abspath("define/QECCLfid/convert.so"))
	_convert.KrausToPTM.argtypes = (
		ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"), # Real part of Kraus
		ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"), # Imgainary part of Kraus
		ct.c_int,  # number of qubits
		ct.c_long, # number of Kraus operators (can set to -1 if the number of Kraus operators is 4^n)
	)
	# Output is the flattened PTM.
	_convert.KrausToPTM.restype = ndpointer(dtype=ct.c_double, shape=(4**nq * 4**nq,))
	# Call the backend function.
	ptm_out = _convert.KrausToPTM(real_kraus, imag_kraus, nq, nkr)
	# ptm_python = KrausToPTM_Python(kraus)
	# print("PTM\n{}".format(ptm_out.reshape(4**nq, 4**nq) - ptm_python))
	ptm = np.array(ptm_out).reshape([4, 4] * nq)
	return ptm


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


def PTMAdjointTest(kraus, ptm):
	# Check if the i,j element of the channel is j,i element of the adjoint channel.
	nq = int(np.log2(kraus.shape[1]))
	if (np.abs(ptm[tuple([0, 0] * nq)] - 1) >= 1E-14):
		# Check if PTM[0, 0] is 1.
		return False
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
	# click = timer()
	nq = pauli_op_i.shape[0]

	trivial_action = 1
	for q in range(nq):
		if q not in supp_ptm:
			trivial_action *= int(pauli_op_i[q] == pauli_op_j[q])

	row_indices = []
	col_indices = []
	if (trivial_action != 0):
		for q in supp_ptm:
			row_indices.append(pauli_op_i[q])
			col_indices.append(pauli_op_j[q])
		ptm_ij = ptm[tuple(row_indices + col_indices)]
	else:
		ptm_ij = 0

	# print("PTM element extracted for Pi = {} and Pj = {} is {}.".format(pauli_op_i, pauli_op_j, ptm_ij))
	return ptm_ij

def chi_to_ptm_pauli(pauli_chi_mat, nq):
	# Convert a chi matrix of a Pauli channel to its PTM representation.
	# ptm[i,j] = 1/2^n Tr ( P_i . E( P_j ) )
	# 		   = 1/2^n sum_kl chi_kl Tr ( P_i . P_k . P_j . P_l )
	# 		   = 1/2^n sum_kl chi_kl 
	# 		   = 1/2^n sum_akl chi_kl ( P_i . P_k . P_j . P_l )_aa
	# 		   = 1/2^n sum_abcd sum_kl chi_kl ( (P_i)ab (P_k)bc (P_j)cd (P_l)da )
	# Note that the ptm for a Pauli channel is a diagonal matrix.
	# 		   = 1/2^n sum_abcd sum_k chi_kk ( (P_i)ab (P_k)bc (P_j)cd (P_k)da )
	# 		   = 1/2^n sum_abcd sum_k chi_k ( (P_i)ab (P_k)bc (P_j)cd (P_k)da )
	# 		   = 1/2^n np.einsum('k,iab,kbc,jcd,kda->ij', chi, P, P, P, P)
	npauli = np.power(4, nq, dtype = np.uint64)
	paulis = np.zeros((npauli, 2**nq, 2**nq), dtype = np.complex128)
	for p in range(npauli):
		pauli_op = GetNQubitPauli(p, nq)
		tn_pauli = PauliTensor(pauli_op)
		paulis[p, :, :] = tn_pauli.reshape(2**nq, 2**nq)
	pauli_ptm_mat = 1 / np.power(2, nq) * np.real(np.einsum('k,iab,kbc,jcd,kda->ij', pauli_chi_mat, paulis, paulis, paulis, paulis, optimize="greedy"))
	return pauli_ptm_mat

def ComputePTMElement(kraus_dict, ls_ops, phases, ns_indices, nstabs, nlogs):
	# Compute an element of the n-qubit PTM when there is only one Kraus operator.
	ptm_elements = np.zeros(ns_indices.size, dtype = np.double)
	for k in range(ns_indices.size):
		ns_index = ns_indices[k]
		# Compute the Pauli operators for which we need to compute the PTM element
		(ns_i, ns_j) = (ns_index // (nlogs * nstabs), ns_index % (nlogs * nstabs))
		pauli_op_i = ls_ops[ns_i, :]
		pauli_op_j = ls_ops[ns_j, :]

		(kraus_support, kraus) = kraus_dict[0]
		nqubits = pauli_op_i.size
		# Pauli operators as tensors
		PauliMats = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=np.complex128)
		pauli_op_i_network = [((q,), PauliMats[pauli_op_i[q], :, :]) for q in range(nqubits) if pauli_op_i[q] > 0]
		pauli_op_j_network = [((q,), PauliMats[pauli_op_j[q], :, :]) for q in range(nqubits) if pauli_op_j[q] > 0]
		# forming the tensor network
		kraus_unpacked = kraus.reshape([2, 2] * len(kraus_support))
		kraus_dag_unpacked = kraus.conj().T.reshape([2, 2] * len(kraus_support))
		network = [(kraus_support, kraus_unpacked)] + pauli_op_i_network + [(kraus_support, kraus_dag_unpacked)] + pauli_op_j_network
		(__, ptm_ij) = ContractTensorNetwork(network, end_trace=1, use_einsum=1)
		ptm_elements[k] = np.real(ptm_ij * phases[ns_i] * phases[ns_j])
		# print("ptm element corresponding to the normalizer element {} = {}".format(k, ptm_elements[k]))
	return ptm_elements

def ConstructPTM(qcode, kraus_theta_chi_dict, compose_with_pauli=0):
	r"""
	Generates LS part of the process matrix of the n-qubit channel.
	"""
	nstabs = 2 ** (qcode.N - qcode.K)
	nlogs = 4 ** qcode.K
	n_maps = len(kraus_theta_chi_dict)
	(ls_ops, phases) = GetOperatorsForTLSIndex(qcode, range(nstabs * nlogs))
	process = np.zeros((nlogs * nstabs, nlogs * nstabs), dtype=np.double)
	
	if (n_maps > 1):
		ptm_dict = []
		for (supp, __, kraus, chi_pauli, __) in kraus_theta_chi_dict:
			ptm_mat = KrausToPTM(kraus)
			if (compose_with_pauli == 1):
				# Compose the PTMs with the Pauli channels that need to be composed.
				# These were generated while constructing the chi matrix.
				ptm_pauli_chan = chi_to_ptm_pauli(chi_pauli, len(supp))
				ptm_mat = ptm_mat.reshape(4**len(supp), 4**len(supp)) @ ptm_pauli_chan
			ptm_dict.append((supp, ptm_mat.reshape([4, 4] * len(supp))))

		click = timer()
		(supp_ptm, ptm_contracted) = ContractTensorNetwork(ptm_dict, end_trace=0)
		print("\033[2mPTM tensor network was contracted in %d seconds.\033[0m" % (timer() - click))
		

		# print("Individual PTMs\n{}".format(ptm_dict))
		# print("Number of nonzero elements in the contracted PTM\n{}".format(np.count_nonzero(ptm_contracted)))
		
		for k in tqdm(range(nlogs * nstabs * nlogs * nstabs), ascii = True, desc = "PTM elements: ", colour = "CYAN"):
		# for k in range(nlogs * nstabs * nlogs * nstabs):
			(i, j) = (k // (nlogs * nstabs), k % (nlogs * nstabs))
			process[i, j] = np.real(phases[i] * phases[j] * ExtractPTMElement(ls_ops[i, :], ls_ops[j, :], ptm_contracted, supp_ptm))

	else:
		kraus_dict = [(supp, kraus) for (supp, __, kraus, __, __) in kraus_theta_chi_dict]
		# for k in tqdm(range(nlogs * nstabs * nlogs * nstabs), ascii = True, desc = "PTM elements: ", colour = "CYAN"):
		# 	(i, j) = (k // (nlogs * nstabs), k % (nlogs * nstabs))
		# 	process[i, j] = np.real(phases[i] * phases[j] * ComputePTMElement(kraus_dict, ls_ops[i, :], ls_ops[j, :]))
		n_cores = mp.cpu_count()
		chunks = np.array_split(np.arange(nlogs * nstabs * nlogs * nstabs, dtype=int), n_cores)
		args = [(kraus_dict, ls_ops, phases, ch, nstabs, nlogs) for ch in chunks]
		process_list = pqdm(args, ComputePTMElement, n_jobs = n_cores, ascii=True, colour='CYAN', desc = "PTM elements", argument_type = 'args')
		# process_list = []
		# for ag in args:
		# 	process_list.append(ComputePTMElement(*ag))
		process = 1 / np.power(2.0, qcode.N) * np.array(np.concatenate(tuple(process_list))).reshape(nstabs * nlogs, nstabs * nlogs)

	return process

"""
@nb.jit(fastmath=True,error_model="numpy",cache=True,parallel=True)
def ComputePTMElements(nlogs, nstabs, ls_ops, phases, supp_ptm, ptm_contracted):
	# Compute the elements of a n-qubit PTM which is expressed as a tensor network of several k-qubit PTMs.
	ptm_LS = np.zeros((nlogs * nstabs, nlogs * nstabs), dtype=np.double)
	for i in range(nlogs * nstabs):
		for j in range(nlogs * nstabs):
			ptm_LS[i, j] = np.real(phases[i] * phases[j] * ExtractPTMElement(ls_ops[i, :], ls_ops[j, :], ptm_contracted, supp_ptm))
	return ptm_LS
"""

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
	n_maps = len(krausdict)
	for key in range(n_maps):
		(support,krausList) = krausdict[key]
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