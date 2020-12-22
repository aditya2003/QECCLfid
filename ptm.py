import numpy as np
from define.QECCLfid.utils import Dagger, circular_shift
from define import globalvars as gv


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

def ApplyChannel(kraus, pauli):
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
		Pi = Pres
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