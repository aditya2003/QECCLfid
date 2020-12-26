import time
import numpy as np
import ctypes as ct
from tqdm import tqdm
from sys import getsizeof
import multiprocessing as mp
from psutil import virtual_memory
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
		theta_channels[mem_start : mem_inter] = np.real(np.ravel(theta))
		mem_end = mem_inter + 16 ** len(support)
		theta_channels[mem_inter : mem_end] = np.imag(np.ravel(theta))
		mem_start = mem_end
	return None


def Theta_Chi_Partial(core, start, stop, mp_chi, paulis, theta_dict):
	# Compute the chi matrix elements for a list of Paulis
	for i in tqdm(range(start, stop), ascii=True, desc="Core %d" % (core + 1), position=core+1):
		mp_chi[i] = np.real(ThetaToChiElement(paulis[i, :], paulis[i, :], theta_dict))
	print("") # to account for the lines used by tqdm.
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

	# If the memory required by a theta matrix is X, then we are limited to RAM/X cores, where RAM is the total physical memory required.
	ram = virtual_memory().total/1E9
	theta_mem_size = getsizeof(theta_dict)/1E9
	n_cores_ram = int(ram/theta_mem_size)

	if (n_cores is None):
		n_cores = mp.cpu_count()

	if (n_cores_ram < n_cores):
		n_cores = min(n_cores, n_cores_ram)
		print("Downsizing to {} cores since the total RAM available is only {} GB and each process needs {} GB.".format(n_cores, ram, theta_mem_size))

	mp_chi = mp.Array(ct.c_double, paulis.shape[0])
	
	chunk = int(np.ceil(paulis.shape[0]/n_cores))
	processes = []
	for p in range(n_cores):
		start = p * chunk
		stop = min((p + 1) * chunk, paulis.shape[0])
		processes.append(mp.Process(target=Theta_Chi_Partial, args = (p, start, stop, mp_chi, paulis, theta_dict)))
	for p in range(n_cores):
		processes[p].start()
	for p in range(n_cores):
		processes[p].join()

	# print("Pauli error probabilities:\n{}".format(chi_diag))
	chi_diag = np.array(mp_chi[:], dtype = np.double)
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
	chi_partial = Chi_Element_Diag(kraus_dict, nrops)
	chi = np.zeros(4**qcode.N, dtype = np.double)
	start = 0
	for w in range(max_weight + 1):
		end = start + n_errors_weight[w]
		chi[qcode.group_by_weight[w]] = chi_partial[start:end]
		start = end
	print("Budget of chi = {}, infid = {}\nElements of chi\n{}".format(np.sum(chi), 1 - chi[0], np.sort(chi)[::-1]))
	return chi
