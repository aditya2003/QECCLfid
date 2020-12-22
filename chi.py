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


def Chi_Element_Diag_Partial(start, end, theta_channels, krausdict, paulis):
	# Compute the theta matrix for a subset of Kraus maps.
	nq = paulis.shape[1]
	for m in range(start, end):
		(support, kraus) = krausdict[m]
		theta_support = tuple([q for q in support] + [(nq + q) for q in support])
		theta = KraussToTheta(np.array(kraus))
		theta_channels.put((m, theta_support, theta))
	return None



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
	if n_cores is None:
		n_cores = mp.cpu_count()
	n_maps = len(list(krausdict.keys()))
	chunk = int(np.ceil(n_maps/n_cores))
	theta_channels = mp.Queue()
	processes = []
	
	for p in range(n_cores):
		start = p * chunk
		end = min((p + 1) * chunk, n_maps)
		processes.append(mp.Process(target = Chi_Element_Diag_Partial, args = (start, end, theta_channels, krausdict, paulis)))
	
	for p in range(n_cores):
		processes[p].start()
	
	for p in range(n_cores):
		processes[p].join()
	
	# Gather the results
	theta_dict = [None for __ in krausdict]
	while not theta_channels.empty():
		(map_index, theta_support, theta) = theta_channels.get()
		theta_dict[map_index] = (theta_support, theta)

	"""
	nq = paulis.shape[1]
	theta_dict = [None for __ in krausdict]
	for m in krausdict:
		(support, kraus) = krausdict[m]
		# print("Shape of Kraus list : {}".format(np.array(kraus).shape))
		theta_dict[m] = (tuple([q for q in support] + [(nq + q) for q in support]), KraussToTheta(np.array(kraus)))
		# print("support = {} and Theta matrix shape = {}".format(support, theta_dict[m][1].shape))
	"""

	(supp_theta, theta_contracted) = ContractTensorNetwork(theta_dict)
	print("theta_contracted supported on {} has shape: {}.".format(supp_theta, theta_contracted.shape))
	theta_contracted_reshaped = theta_contracted
	# theta_contracted_reshaped = theta_contracted.reshape([2, 2, 2, 2]*len(supp_theta))
	# (supp_theta, theta_contracted) = theta_dict[0] # only for debugging purposes.

	chi_diag = np.zeros(paulis.shape[0], dtype = np.double)
	for i in range(paulis.shape[0]):
		chi_diag[i] = np.real(ThetaToChiElement(paulis[i, :], paulis[i, :], theta_contracted, supp_theta))

	print("Pauli error probabilities:\n{}".format(chi_diag))
	return chi_diag