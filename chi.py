import time
import numpy as np
import ctypes as ct
import multiprocessing as mp
from timeit import default_timer as timer
from define.QECCLfid.contract import ContractTensorNetwork
from define.QECCLfid.theta import KraussToTheta,ThetaToChiElement
from define.qcode import GetOperatorsForLSTIndex, PrepareSyndromeLookUp
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


def Chi_Element_Diag_Partial(map_start, map_end, mem_start, theta_channels, krausdict, paulis):
	# Compute the theta matrix for a subset of Kraus maps.
	nq = paulis.shape[1]
	for m in range(map_start, map_end):
		(support, kraus) = krausdict[m]
		click = timer()
		theta = KraussToTheta(np.array(kraus))
		print("Theta matrix for map {} was computed in {} seconds.".format(m + 1, timer() - click))
		# theta_channels[m] = (theta_support, theta)
		mem_inter = mem_start + 16 ** len(support)
		theta_channels[mem_start : mem_inter] = np.real(np.reshape(theta, -1))
		mem_end = mem_inter + 16 ** len(support)
		theta_channels[mem_inter : mem_end] = np.imag(np.reshape(theta, -1))
		mem_start = mem_end
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
	size_theta_contracted = [0 for __ in range(n_maps)]
	for m in range(n_maps):
		(support, kraus) = krausdict[m]
		size_theta_contracted[m] = 2 * 16 ** len(support)
	theta_channels = mp.Array(ct.c_double, sum(size_theta_contracted))

	chunk = int(np.ceil(n_maps/n_cores))
	processes = []

	for p in range(n_cores):
		map_start = p * chunk
		map_end = min((p + 1) * chunk, n_maps)
		mem_start = sum(size_theta_contracted[:map_start])
		# Chi_Element_Diag_Partial(start, end, theta_dict, krausdict, paulis)
		processes.append(mp.Process(target = Chi_Element_Diag_Partial, args = (map_start, map_end, mem_start, theta_channels, krausdict, paulis)))

	for p in range(n_cores):
		processes[p].start()

	for p in range(n_cores):
		processes[p].join()

	# Gather the results
	theta_dict = [None for __ in krausdict]
	nq = paulis.shape[1]
	for m in range(n_maps):
		(support, kraus) = krausdict[m]
		theta_support = tuple([q for q in support] + [(nq + q) for q in support])
		
		mem_start = sum(size_theta_contracted[:m])
		mem_inter = mem_start + 16 ** len(support)
		theta_real = np.reshape(theta_channels[mem_start : mem_inter], [2, 2] * len(theta_support))

		mem_end = mem_inter + 16 ** len(support)
		theta_imag = np.reshape(theta_channels[mem_inter : mem_end], [2, 2] * len(theta_support))
		mem_start = mem_end

		theta = theta_real + 1j * theta_imag

		theta_dict[m] = (theta_support, theta)

	# print("All theta matrices are done.")

	click = timer()
	(supp_theta, theta_contracted) = ContractTensorNetwork(theta_dict)
	print("Theta tensor network was contracted in {} seconds.".format(timer() - click))

	chi_diag = np.zeros(paulis.shape[0], dtype = np.double)
	for i in range(paulis.shape[0]):
		# click = timer()
		chi_diag[i] = np.real(ThetaToChiElement(paulis[i, :], paulis[i, :], theta_contracted, supp_theta))
		# print("Chi[{}, {}] was computed in {} seconds.".format(i, i, timer() - click))

	# print("Pauli error probabilities:\n{}".format(chi_diag))
	return chi_diag


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
	
	# In the chi matrix, fill the entries corresponding to nrops with the reconstruction data.
	chi_partial = Chi_Element_Diag(kraus_dict, nrops) # For debugging only.
	chi = np.zeros(4**qcode.N, dtype = np.double)
	start = 0
	for w in range(max_weight + 1):
		end = start + n_errors_weight[w]
		chi[qcode.group_by_weight[w]] = chi_partial[start:end]
		start = end
	print("Budget of chi = {}, infid = {}\nElements of chi\n{}".format(np.sum(chi), 1 - chi[0], np.sort(chi)[::-1]))
	return chi
