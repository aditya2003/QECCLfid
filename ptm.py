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


def get_PTMelem_ij(krausdict, Pi, Pjlist, n_qubits,phasei=None,phasej=None):
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
		for kraus in krausList:
			if len(indices) > 0:
				kraus_reshape_dims = [2] * (2 * int(np.log2(kraus.shape[0])))
				indices_Pi = indices[len(indices) // 2 :]
				indices_kraus = range(len(kraus_reshape_dims) // 2)
				Pi_term = np.tensordot(
					Pi,
					Dagger(kraus).reshape(kraus_reshape_dims),
					(indices_Pi, indices_kraus),
				)
				Pi_term = fix_index_after_tensor(Pi_term, indices_Pi)
				indices_Pi = indices[: len(indices) // 2]
				indices_kraus = range(len(kraus_reshape_dims))[
					len(kraus_reshape_dims) // 2 :
				]
				Pi_term = np.tensordot(
					Pi_term,
					kraus.reshape(kraus_reshape_dims),
					(indices_Pi, indices_kraus),
				)
				Pi_term = fix_index_after_tensor(Pi_term, indices_Pi)
				Pres += Pi_term
			else:
				Pres = Pi * (np.abs(kraus) ** 2)
	# take dot product with Pj and trace
	trace_vals = np.zeros(len(Pjlist), dtype=np.double)
	indices_Pi = list(range(len(Pi.shape) // 2))
	indices_Pj = list(range(len(Pi.shape) // 2, len(Pi.shape)))
	for i in range(len(Pjlist)):
		Pj = Pjlist[i]
		Pres_times_Pj = np.tensordot(Pres, Pj, (indices_Pi, indices_Pj))
		# Take trace
		einsum_inds = list(range(len(Pres_times_Pj.shape) // 2)) + list(
			range(len(Pres_times_Pj.shape) // 2)
		)
		raw_trace = np.einsum(Pres_times_Pj, einsum_inds)*phasei*phasej[i]
		# if np.abs(np.imag(raw_trace)) > 1E-15:
		# 	print("raw_trace {}: {}".format(i, raw_trace))
		trace_vals[i] = np.real(raw_trace) / 2 ** n_qubits
	return trace_vals
