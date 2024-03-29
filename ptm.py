import os
import copy
import numpy as np
import ctypes as ct
import more_itertools as mit
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
	ptm = ptm_out.reshape([4, 4] * nq)
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


def ConstructPTM_Partial(kraus_chans):
	# Construct PTM for some maps.
	# for m in tqdm(range(map_start, map_end), ascii = True, desc = "Core %d" % (core + 1), position = core, colour = "yellow"):
	n_maps = len(kraus_chans)
	ptm_chans = [None for __ in range(n_maps)]
	for m in range(n_maps):
		# click = timer()
		ptm_chans[m] = KrausToPTM(np.array(kraus_chans[m]))
		# print("\033[2mPTM for map %d was constructed in %.2f seconds.\033[0m" % (m + 1, timer() - click))
	return ptm_chans


def ConstructPTM(qcode, kraus_dict, n_cores = None):
	r"""
	Generates LS part of the process matrix of the n-qubit channel.
	"""
	nstabs = 2 ** (qcode.N - qcode.K)
	nlogs = 4 ** qcode.K
	n_maps = len(kraus_dict)
	ptm_channels_sizes = [0 for __ in range(n_maps)]
	for m in range(n_maps):
		(support, __) = kraus_dict[m]
		ptm_channels_sizes[m] = 16 ** len(support)
	ptm_channels = mp.Array(ct.c_double, sum(ptm_channels_sizes))

	if (n_cores is None):
		n_cores = mp.cpu_count()

	n_cores = min(n_cores, n_maps)

	# Using pqdm: https://pqdm.readthedocs.io/en/latest/usage.html
	kr_ops=[kr_op for (supp, kr_op) in kraus_dict]
	kr_ops_chunks = [list(chunk) for chunk in mit.divide(n_cores, kr_ops)]
	ptms_list_chunks = pqdm(kr_ops_chunks, ConstructPTM_Partial, n_jobs = n_cores, ascii=True, colour = "CYAN", desc = "PTM elements", argument_type = 'args')
	ptms_list = [ptms for p in range(n_cores) for ptms in ptms_list_chunks[p]]
	ptm_dict = list(zip([supp for (supp, __) in kraus_dict], ptms_list))
	
	# print("PTMs\n{}".format(ptm_dict))

	click = timer()
	(supp_ptm, ptm_contracted) = ContractTensorNetwork(ptm_dict, end_trace=0)
	print("\033[2mPTM tensor network was contracted in %d seconds.\033[0m" % (timer() - click))

	# print("Individual PTMs\n{}".format(ptm_dict))
	# print("Number of nonzero elements in the contracted PTM\n{}".format(np.count_nonzero(ptm_contracted)))
	
	(ls_ops, phases) = GetOperatorsForTLSIndex(qcode, range(nstabs * nlogs))
	process = np.zeros((nlogs * nstabs, nlogs * nstabs), dtype=np.double)
	
	for k in tqdm(range(nlogs * nstabs * nlogs * nstabs), ascii = True, desc = "PTM elements: ", colour = "CYAN"):
	# for k in range(nlogs * nstabs * nlogs * nstabs):
		(i, j) = (k // (nlogs * nstabs), k % (nlogs * nstabs))
		process[i, j] = np.real(phases[i] * phases[j] * ExtractPTMElement(ls_ops[i, :], ls_ops[j, :], ptm_contracted, supp_ptm))
	
	# process = ComputePTMElements(nlogs, nstabs, ls_ops, phases, np.array(supp_ptm, dtype = np.int8), ptm_contracted)
	# print("process\n{}".format(process))
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