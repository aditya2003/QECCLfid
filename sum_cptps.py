import random
import numpy as np
from define.randchans import RandomUnitary
from define.QECCLfid.utils import SamplePoisson

def HermitianConjugate(M):
	return M.conj().T

def StineToKraus(U):
	# Compute the Krauss operators for the input quantum channel, which is represented in the Stinespring dialation
	# The Krauss operator T_k is given by: <a|T_k|b> = <a e_k|U|b e_0> , where {|e_i>} is a basis for the environment and |a>, |b> are basis vectors of the system
	# Note that a k-qubit channel needs to be generated from a unitary matrix on 3*k qubits where 2*k qubits represent the environment.
	nq = int(np.log2(U.shape[0]))//3
	# environment = np.eye(4**nq)[:,:,np.newaxis]
	# system = np.eye(2**nq)[:,:,np.newaxis]
	krauss = np.zeros((4**nq, 2**nq, 2**nq), dtype=np.complex128)
	for r in range(2**nq):
		for c in range(2**nq):
			krauss[:, r, c] = U[r * 4**nq + np.arange(4**nq, dtype = np.int), c * 4**nq]
	return krauss

def SumCptps(rotation_angle, qcode, cutoff = 3, n_maps = 3):
	r"""
	Sub-routine to prepare the dictionary for error eps = sum of cptp maps
	Generates Kraus by using Stine to Kraus
	of random multi-qubit unitary operators rotated by rotation_angle/2**|support|
	Probability associated with each weight-k error scales exponentially p_error^k (except weights <= w_thresh)
	Input :
	rotation_angle = the angle to rotate the Stinespring unitary by
	qcode = QEC
	n_maps = number of maps to be considered for the sum
	Returns :
	dict[key] = (support,krauslist)
	where key = number associated to the operation applied (not significant)
	support = tuple describing which qubits the kraus ops act on
	krauslist = krauss ops acting on support
	"""
	# print("Sum of CPTP maps:\ncutoff = {}, n_maps = {}".format(cutoff, n_maps))
	kraus_dict = {m:None for m in range(n_maps)}
	n_nontrivial_maps = 0
	for m in range(n_maps):
		# n_q = SamplePoisson(mean = 1, cutoff=cutoff)
		n_q = 1 # for debugging only
		# support = tuple(sorted((random.sample(range(qcode.N), n_q))))
		support = (1, 3) # for debugging only
		if n_q == 0:
			rand_unitary = 1.0
			kraus_dict[m] = (support,[rand_unitary])
		else:
			rand_unitary = RandomUnitary(rotation_angle/8**n_q, 8**n_q)
			kraus = StineToKraus(rand_unitary)
			KrausTest(kraus)
			kraus_dict[m] = (support, kraus)
			n_nontrivial_maps += 1

	# Remove identity channels
	non_trivial_channels = {m:None for m in range(n_nontrivial_maps)}
	supports = [None for __ in range(n_nontrivial_maps)]
	n_nontrivial_maps = 0
	for m in range(n_maps):
		(support, kraus) = kraus_dict[m]
		if len(support) > 0:
			non_trivial_channels[n_nontrivial_maps] = (support, kraus)
			supports[n_nontrivial_maps] = support
			n_nontrivial_maps += 1

	print("Random channel generated with the following {} interactions\n{}.".format(n_nontrivial_maps, supports))
	return non_trivial_channels

def KrausTest(kraus):
	# Given a set of Kraus operators {K_i}, check if \sum_i [ (K_i)^dag K_i ] = I.
	total = np.zeros((kraus.shape[1], kraus.shape[2]), dtype = np.complex128)
	for k in range(kraus.shape[0]):
		# print("||K_{} - diag(K_{})||_2 = {}".format(k, k, np.linalg.norm(kraus[k, :, :] - np.diag(np.diag(kraus[k, :, :])))))
		total = total + np.dot(HermitianConjugate(kraus[k, :, :]), kraus[k, :, :])
	success = 0
	if np.allclose(total, np.eye(kraus.shape[1], dtype=np.complex128)):
		success = 1
		# print("Kraus test passed.")
	else:
		print("sum_i [ (K_i)^dag K_i ]\n{}".format(total))
		print("Kraus test failed.")
		exit(0)
	return success
