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
			krauss[:, r, c] = U[(r - 1) * 4**nq + (np.arange(4**nq, dtype = np.int) - 1), (c - 1) * 4**nq]
	return krauss

def SumCptps(rotation_angle, qcode, cutoff = 3, n_maps = 3):
	r"""
	Sub-routine to prepare the dictionary for error eps = sum of cptp maps
	Generates kraus by using stine to kraus
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
	print("Sum of CPTP maps:\ncutoff = {}, n_maps = {}".format(cutoff, n_maps))
	kraus_dict = {}
	prob_maps = np.array([1/n_maps]*n_maps) # Probabilites associated to the above CPTP maps
	cutoff = 3 # Cutoff number of qubits for poissson distribution
	cptp_map_count = 0
	for __ in range(n_maps):
		n_q = SamplePoisson(mean = 1, cutoff=cutoff)
		support = tuple(sorted((random.sample(range(qcode.N), n_q))))
		
		print("support of size {}\n{}".format(n_q, support))
		
		if n_q == 0:
			rand_unitary = 1.0
			kraus_dict[cptp_map_count] = (support,[rand_unitary])
		else:
			rand_unitary = RandomUnitary(
				rotation_angle/(2**(3*n_q)), 2**(3*n_q)
			)
			kraus_dict[cptp_map_count] = (support, StineToKraus(rand_unitary))
		
		# print("U on {} qubits\n{}".format(n_q, rand_unitary))

		cptp_map_count += 1
	
	# Multiplying kraus by their respective probabilities
	for key, (support, krauslist) in kraus_dict.items():
		for k in range(len(krauslist)):
			# print("k = {}".format(k))
			kraus_dict[key][1][k] *= np.sqrt(prob_maps[key])
	print("Kruas dictionary prepared.")
	return kraus_dict