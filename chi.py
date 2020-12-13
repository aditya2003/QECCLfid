import numpy as np
from define.QECCLfid.ptm import fix_index_after_tensor, get_Pauli_tensor

def get_chi_kraus(kraus, Pi, indices_Pi, n_qubits):
	# Compute the addition to the Chi element from a given Kraus operator.
	kraus_reshape_dims = [2] * (2 * int(np.log2(kraus.shape[0])))
	indices_kraus = range(len(kraus_reshape_dims) // 2)
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
	# print("einsum_inds_old = {}".format(einsum_inds))
	contrib = (
		np.power(np.abs(np.einsum(Pi_term, einsum_inds)), 2)
		/ 4 ** n_qubits
	)
	return contrib

def Chi_Element_Diag(krausdict, Pilist, n_qubits):
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
			indices_Pi = indices[len(indices) // 2 :]
			if (len(indices) > 0):
				chi[i] += np.sum([get_chi_kraus(kraus, Pi, indices_Pi, n_qubits) for kraus in krausList])
			else:
				if i == 0:
					chi[i] += np.abs(krausList[0]) ** 2
	return chi