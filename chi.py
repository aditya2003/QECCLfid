import time
import numpy as np
import ctypes as ct
import multiprocessing as mp
from define.QECCLfid.contract import ContractTensorNetwork
from define.QECCLfid.theta import KraussToTheta,ThetaToChiElement
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


def Chi_Element_Diag(krausdict, paulis, n_cores=None):
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
	thetadict = [None for __ in krausdict]
	for key, (support, krauslist) in krausdict.items():
		nq = len(support)
		# print("Shape of Kraus list : {}".format(np.array(krauslist).shape))
		thetadict[key] = (support, KraussToTheta(np.array(krauslist)).reshape([4, 4]*nq))
		# print("support = {} and Theta matrix shape = {}".format(support, thetadict[key][1].shape))
	(supp_theta, theta_contracted) = ContractTensorNetwork(thetadict)
	print("theta_contracted supported on {} has shape: {}.".format(supp_theta, theta_contracted.shape))
	theta_contracted_reshaped = theta_contracted.reshape([2, 2, 2, 2]*len(supp_theta))
	# (supp_theta, theta_contracted) = thetadict[0] # only for debugging purposes.

	chi_diag = np.zeros(paulis.shape[0], dtype = np.double)
	for i in range(paulis.shape[0]):
		chi_diag[i] = np.real(ThetaToChiElement(paulis[i, :], paulis[i, :], theta_contracted_reshaped, supp_theta))

	print("Pauli error probabilities:\n{}".format(chi_diag))
	return chi_diag