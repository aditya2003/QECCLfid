import time
import numpy as np
import ctypes as ct
from tqdm import tqdm
#from pqdm.processes import pqdm
from sys import getsizeof
import multiprocessing as mp
from psutil import virtual_memory
from timeit import default_timer as timer
from define.QECCLfid.ptm import fix_index_after_tensor
from define.QECCLfid.contract import ContractTensorNetwork
from define.QECCLfid.theta import KrausToTheta, ThetaToChiElement, get_theta_pauli_channel
from define.qcode import GetOperatorsForLSTIndex, PrepareSyndromeLookUp


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
		# print("Computing theta matrix for map {} supported on {}".format(m, support))
		click = timer()
		theta = KrausToTheta(np.array(kraus))
		print("\033[2mTheta matrix for map %d was computed in %.2f seconds.\033[0m" % (m + 1, timer() - click))
		# theta_channels[m] = (theta_support, theta)
		mem_inter = mem_start + 16 ** len(support)
		theta_channels[mem_start : mem_inter] = np.real(np.reshape(theta, -1))
		mem_end = mem_inter + 16 ** len(support)
		theta_channels[mem_inter : mem_end] = np.imag(np.reshape(theta, -1))
		mem_start = mem_end
	return None


def Theta_to_Chi_Elements(paulis_chunk, pauli_chunk_indices, kraus_theta_chi_dict, chi_diag):
	# Compute the chi matrix elements for a list of Paulis
	start = timer()
	for p in range(paulis_chunk.shape[0]):
		chi_diag[pauli_chunk_indices[p]] = np.real(ThetaToChiElement(paulis_chunk[p, :], paulis_chunk[p, :], kraus_theta_chi_dict))
		# print("[{}s]: Chi element for p = {}.".format(timer() - start, p))
	
	# print("[{}s]: {} Chi elements computed.".format(timer() - start, pauli_chunk_indices.size, p))
	return None


def Chi_Element_Diag(kraus_dict, paulis, compose_with_pauli_rate=0, n_cores=None):
	r"""
	Calculates the diagonal entry in chi matrix corresponding to each Pauli in Pilist
	Assumes each Pauli in list of Paulis Pilist to be a tensor on n_qubits
	Calculates chi_ii = sum_k |<Pi, A_k>|^2
	where A_k is thr Kraus operator and Pi is the Pauli operator
	Kraus dict has the format ("support": list of kraus ops on the support)
	Assumes qubits
	Assumes kraus ops to be square with dim 2**(number of qubits in support)
	"""
	# Gather the results
	kraus_theta_chi_dict = [None for __ in kraus_dict]
	nq = paulis.shape[1]
	n_maps = len(kraus_dict)
	chi_pauli = None
	for m in range(n_maps):
		(support, kraus) = kraus_dict[m]
		click = timer()
		theta = KrausToTheta(kraus)
		if (compose_with_pauli_rate > 0):
			# Compose the k-qubit channel with a random Pauli channel whose single-qubit infidelity (r0) is provided.
			# The infidelity of the channel is computed as follows.
			# r = 1 - (1 - r0)^k, where r is the infidelity of the k-qubit map.
			# This is particularly useful for approximating a random CPTP map with the composition of a random Unitary map and a random Pauli channel
			# The composition is simple in the Theta matrix picture since we just have to multiply the respective Theta matrices.
			infid = 1 - np.power(1 - compose_with_pauli_rate, len(support))
			(theta_pauli, chi_pauli) = get_theta_pauli_channel(len(support), infid)
			# print("map {} with theta\n{}\nand theta_pauli\n{}".format(m, theta, theta_pauli))
			theta = theta.reshape(4**len(support), 4**len(support)) @ theta_pauli
		# print("\033[2mTheta matrix for map %d was computed in %.2f seconds.\033[0m" % (m + 1, timer() - click))
		theta_support = tuple([q for q in support] + [(nq + q) for q in support])
		kraus_theta_chi_dict[m] = (support, theta_support, kraus, chi_pauli, theta.reshape([2, 2, 2, 2] * len(support)))

	# print("All theta matrices are done.\n{}".format(kraus_theta_chi_dict))

	# If the memory required by a theta matrix is X, then we are limited to RAM/X cores, where RAM is the total physical memory required.
	ram = virtual_memory().total
	theta_mem_size = getsizeof(kraus_theta_chi_dict)
	n_cores_ram = int(ram/theta_mem_size)

	if (n_cores is None):
		n_cores = mp.cpu_count()

	if (n_cores_ram < n_cores):
		n_cores = min(n_cores, n_cores_ram)
		print("Downsizing to {} cores since the total RAM available is only {} GB and each process needs {} GB.".format(n_cores, ram, theta_mem_size))

	npauli = paulis.shape[0]
	pauli_chunks = np.array_split(paulis, n_cores, axis=0)
	pauli_indices = np.array_split(np.arange(npauli, dtype=int), n_cores)

	chi_diag = mp.Array('d', range(npauli))
	
	processes = []
	for p in range(n_cores):
		processes.append(mp.Process(target=Theta_to_Chi_Elements, args=(pauli_chunks[p], pauli_indices[p], kraus_theta_chi_dict, chi_diag)))

	for p in range(n_cores):
		processes[p].start()

	for p in range(n_cores):
		processes[p].join()
	
	# Using pqdm: https://pqdm.readthedocs.io/en/latest/usage.html
	# print("paulis[::{}, :] = {}".format(n_cores, np.array_split(paulis, n_cores, axis=0)))
	# args=list(zip(np.array_split(paulis, n_cores, axis=0), [kraus_theta_chi_dict for __ in range(n_cores)]))
	# print("Args\n{}".format(args))
	# chi_diag_elements_chunks = pqdm(args, Theta_to_Chi_Elements, n_jobs = n_cores, ascii=True, colour='CYAN', desc = "Chi Matrix elements", argument_type = 'args')
	
	###########
	# Only for debugging purposes
	# chi_diag_elements_chunks = []
	# for i in range(n_cores):
	# 	output_core = Theta_to_Chi_Elements(*args[i])
	# 	print("Output from core {}\n{}".format(i, output_core))
	# 	chi_diag_elements_chunks.append(output_core)
	###########

	# print("chi_diag_elements_chunks\n{}".format(chi_diag_elements_chunks))
	# chi_diag = np.concatenate(tuple(chi_diag_elements_chunks))
	# print("Pauli error probabilities:\n{}".format(chi_diag))
	return (np.array(chi_diag), kraus_theta_chi_dict)

def KraussToChi(kraus_dict, nrops):
	# Compute the Chi matrix from the Kraus representation when we have only one Kraus operator.
	# In this case we don't need to compose various chi-matrices.
	assert len(kraus_dict) == 1
	(kraus_support, kraus) = kraus_dict[0]
	support_size = len(kraus_support)
	
	PauliMats = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=np.complex128)
	(npauli, nqubits) = nrops.shape

	chi = np.zeros(npauli, dtype = np.double)
	start = timer()
	for p in range(npauli):
		network = [(kraus_support, kraus.reshape([2, 2] * support_size))] + [((q,), PauliMats[nrops[p, q], :, :]) for q in range(nqubits) if nrops[p, q] > 0]
		(__, chi_left) = ContractTensorNetwork(network, end_trace=1, use_einsum=1)
		network = [(kraus_support, kraus.conj().T.reshape([2, 2] * support_size))] + [((q,), PauliMats[nrops[p, q], :, :]) for q in range(nqubits) if nrops[p, q] > 0]
		(__, chi_right) = ContractTensorNetwork(network, end_trace=1, use_einsum=1)
		chi[p] = np.real(chi_left * chi_right)
		# if (p == 0):
		# 	print("kraus.reshape(4,4) = {}".format(kraus.reshape(4,4)))
		# 	print("Chi_0,0 = {} = {} x {} = {}".format(chi_left, np.trace(kraus.reshape(4,4)), chi_right,np.trace(kraus.conj().T.reshape(4,4))))
	chi = chi / np.power(4, support_size)
	# print("Chi computed in {} seconds.\n{}".format(timer() - start, chi))
	return chi

def NoiseReconstruction(qcode, kraus_dict, compose_with_pauli_rate=0, max_weight=None):
	r"""
	Compute the diagonal elements of the Chi matrix, i.e., Pauli error probabilities.
	We don't want all the diagonal entries; only a fraction "x" of these.
	For a given fraction "x", we will choose x * 4^N errors, picking the low weight ones before one of a higher weight.
	Amongst errors of the same weight, we will simply choose a random

	chi matrix in LST ordering.
	"""
	if max_weight is None:
		max_weight = qcode.N
	if qcode.group_by_weight is None:
		PrepareSyndromeLookUp(qcode)
	# n_errors_weight = [qcode.group_by_weight[w].size for w in range(max_weight + 1)]
	# nrops = np.zeros((np.sum(n_errors_weight, dtype = np.int64), qcode.N), dtype = np.int8)
	# filled = 0
	# for w in range(max_weight + 1):
	# 	(nrops[filled : (filled + n_errors_weight[w]), :], __) = GetOperatorsForLSTIndex(qcode, qcode.group_by_weight[w])
	# 	filled += n_errors_weight[w]
	
	if (len(kraus_dict) == 1):
		# print("Special method for {} Kraus operator.".format(len(kraus_dict)))
		chi = KraussToChi(kraus_dict, qcode.PauliOperatorsLST)
		kraus_theta_chi_dict = [(supp, None, kraus, None, None) for (supp, kraus) in kraus_dict]
	else:
		# In the chi matrix, fill the entries corresponding to nrops with the reconstruction data.
		(chi, kraus_theta_chi_dict) = Chi_Element_Diag(kraus_dict, qcode.PauliOperatorsLST, compose_with_pauli_rate=compose_with_pauli_rate)
	# print("chi matrix diagonal entries\n{}".format(chi))

	# Compute different statistical properties of the probability distribution over weight w errors.
	weight_stats = np.zeros((max_weight + 1, 4), dtype = np.double)
	for w in range(max_weight + 1):
		weight_stats[w, 0] = np.sum(chi[qcode.group_by_weight[w]]) # Total probability of weight w errors
		weight_stats[w, 1] = np.min(chi[qcode.group_by_weight[w]]) # Minimum probability of weight w errors
		weight_stats[w, 2] = np.max(chi[qcode.group_by_weight[w]]) # Maximum probability of weight w errors
		weight_stats[w, 3] = np.median(chi[qcode.group_by_weight[w]]) # Median probability of weight w errors
	
	# Convert total probability of weight - w errors into percentages.
	weight_stats[:, 0] = weight_stats[:, 0] / np.sum(weight_stats[:, 0]) * 100
	for w in range(max_weight + 1):
		print("\033[2mFraction of weight %d errors: %.3e %%. %g <= p_%d <= %g. Median: %g\033[0m" % (w, weight_stats[w, 0], weight_stats[w, 1], w, weight_stats[w, 2], weight_stats[w, 3]))
	
	atol = 1E-10
	if ((np.any(np.real(chi) < -atol)) or (np.real(np.sum(chi)) >= 1 + atol)):
		print("Invalid chi matrix: chi[0,0] = {} and sum of chi = {}.".format(chi[0], np.sum(chi)))
		# Save the Kraus operators for investigation
		filename = "problematic_kraus_%s.txt" % (time.time())
		# print("kraus_dict\n{}".format(kraus_dict))
		with open(filename, "w") as fp:
			fp.write("Invalid chi matrix: chi[0,0] = {} and sum of chi = {}\n.".format(chi[0], np.sum(chi)))
			fp.write("Budget of chi excluded = {} and infid = {}.\n".format(1 - np.sum(chi), 1 - chi[0]))
			for k in kraus_dict:
				(supp, op) = kraus_dict[k]
				fp.write("supp: {}\n{}\n======\n".format(supp, np.array_str(op)))
	print("\033[2mBudget of chi excluded = %.2e and infid = %.2e.\033[0m" % (1 - np.sum(chi), 1 - chi[0]))
	return (chi, kraus_theta_chi_dict)