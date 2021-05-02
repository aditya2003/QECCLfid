import numpy as np
from scipy.special import comb
from define.qcode import GetOperatorsForTLSIndex, GetOperatorsForLSTIndex, PrepareSyndromeLookUp
from define.QECCLfid.chi import Chi_Element_Diag
from define.QECCLfid.ptm import PTM_Element, get_Pauli_tensor, ExtractPTMElement
from define.QECCLfid.utils import Dot, Kron, Dagger, circular_shift
# for debugging
from define.QECCLfid.backup import get_Chielem_ii, get_PTMelem_ij

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
		process[i * nstabs * nlogs : (i + 1) * nstabs * nlogs] = get_PTMelem_ij(
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
	chi = get_Chielem_ii(kraus_dict, ops, qcode.N)
	# chi = get_Chielem_broadcast(kraus_dict, ops, qcode.N)
	# chi = Chi_Element_Diag(kraus_dict, ops, qcode.N)
	print("Sum of chi = {}, infid = {}\nElements of chi\n{}".format(np.sum(chi), 1 - chi[0], np.sort(chi)[::-1]))
	return chi
