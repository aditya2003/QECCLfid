import numpy as np
from scipy.special import comb
from define.qcode import GetOperatorsForTLSIndex, GetOperatorsForLSTIndex, PrepareSyndromeLookUp
from define.QECCLfid.chi import Chi_Element_Diag
from define.QECCLfid.ptm import PTM_Element, get_Pauli_tensor
from define.QECCLfid.utils import Dot, Kron, Dagger, circular_shift
# for debugging
from define.QECCLfid.backup import get_Chielem_ii

def get_process_correlated(qcode, kraus_dict):
	r"""
	Generates LS part of the process matrix for Eps = sum of unitary errors
	p_error^k is the probability associated to a k-qubit unitary (except weight <= w_thresh)
	rotation_angle is the angle used for each U = exp(i*rotation_agnle*H)
	return linearized Pauli transfer matrix matrix in LS ordering for T=0
	"""
	nstabs = 2 ** (qcode.N - qcode.K)
	nlogs = 4 ** qcode.K
	(ops, phases) = GetOperatorsForTLSIndex(qcode, range(nstabs * nlogs))
	ops_tensor = list(map(get_Pauli_tensor, ops))
	process = np.zeros(nstabs * nstabs * nlogs * nlogs, dtype=np.double)
	for i in range(len(ops_tensor)):
		process[i * nstabs * nlogs : (i + 1) * nstabs * nlogs] = PTM_Element(
			kraus_dict, ops_tensor[i], ops_tensor, qcode.N, phases[i], phases
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
	(ops, phases) = GetOperatorsForLSTIndex(qcode, range(nstabs * nstabs * nlogs))
	for i in range(len(diag_process)):
		op_tensor = get_Pauli_tensor(ops[i]) * phases[i]
		diag_process[i] = PTM_Element(kraus_dict, op_tensor, [op_tensor], qcode.N)[0]
	return diag_process


def get_chi_diagLST(qcode, kraus_dict):
	r"""
	Generates diagonal of the chi matrix in LST ordering for Eps = sum of unitary errors
	p_error^k is the probability associated to a k-qubit unitary (except weight <= w_thresh)
	rotation_angle is the angle used for each U = exp(i*rotation_angle*H)
	return linearized diagonal of the chi matrix in LST ordering.
	"""
	nstabs = 2 ** (qcode.N - qcode.K)
	nlogs = 4 ** qcode.K
	(ops, __) = GetOperatorsForLSTIndex(qcode, range(nstabs * nstabs * nlogs))
	# ops = [[0, 0, 0, 1, 0, 0, 0]] # only for debugging.
	chi = get_Chielem_ii(kraus_dict, ops, qcode.N)
	# chi = get_Chielem_broadcast(kraus_dict, ops, qcode.N)
	# chi = Chi_Element_Diag(kraus_dict, ops, qcode.N)
	print("Sum of chi = {}, infid = {}\nElements of chi\n{}".format(np.sum(chi), 1 - chi[0], np.sort(chi)[::-1]))
	return chi

def NoiseReconstruction(qcode, kraus_dict, max_weight=None):
	r"""
	Compute the diagonal elements of the Chi matrix, i.e., Pauli error probabilities.
	We don't want all the diagonal entries; only a fraction "x" of these.
	For a given fraction "x", we will choose x * 4^N errors, picking the low weight ones before one of a higher weight.
	Amongst errors of the same weight, we will simply choose a random

	chi matrix in LST ordering.
	"""
	if max_weight is None:
		max_weight = qcode.N//2 + 1
	if qcode.group_by_weight is None:
		PrepareSyndromeLookUp(qcode)
	n_errors_weight = [qcode.group_by_weight[w].size for w in range(max_weight + 1)]
	nrops = np.zeros((np.sum(n_errors_weight, dtype = np.int), qcode.N), dtype = np.int8)
	filled = 0
	for w in range(max_weight + 1):
		(nrops[filled : (filled + n_errors_weight[w]), :], __) = GetOperatorsForLSTIndex(qcode, qcode.group_by_weight[w])
		filled += n_errors_weight[w]
	# nrops = np.array([[0, 0, 0, 1, 0, 0, 0]], dtype = np.int8) # only for debugging.
	# In the chi matrix, fill the entries corresponding to nrops with the reconstruction data.
	chi_partial = Chi_Element_Diag(kraus_dict, nrops)
	chi = np.zeros(4**qcode.N, dtype = np.double)
	start = 0
	for w in range(max_weight + 1):
		end = start + n_errors_weight[w]
		chi[qcode.group_by_weight[w]] = chi_partial[start:end]
		start = end
	print("Sum of chi = {}, infid = {}\nElements of chi\n{}".format(np.sum(chi), 1 - chi[0], np.sort(chi)[::-1]))
	return chi
