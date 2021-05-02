import random
import numpy as np
# import scipy as sc
from define.QECCLfid.utils import extend_gate
from define import randchans as rc

def SumUnitaries(p_error, rotation_angle, qcode, w_thresh=3):
	r"""
	Sub-routine to prepare the dictionary for error eps = sum of unitary errors
	Generates kraus as random multi-qubit unitary operators rotated by rotation_angle
	Probability associated with each weight-k error scales exponentially p_error^k (except weights <= w_thresh)
	Returns :
	dict[key] = (support,krauslist)
	where key = number associated to the operation applied (not significant)
	support = tuple describing which qubits the kraus ops act on
	krauslist = krauss ops acting on support
	"""
	kraus_dict = {}

	kraus_count = 0
	norm_coeff = np.zeros(qcode.N + 1, dtype=np.double)
	for n_q in range(qcode.N, qcode.N + 1):
		if n_q == 0:
			p_q = 1 - p_error
		elif n_q <= w_thresh:
			p_q = np.power(0.1, n_q - 1) * p_error
		else:
			p_q = np.power(p_error, n_q)
		# Number of unitaries of weight n_q is max(1,qcode.N-n_q-1)
		if n_q == 0:
			nops_q = 1
		else:
			# nops_q = 1 # For debugging.
			# nops_q = max(1, (qcode.N + n_q) // 2) # Aditya's version
			nops_q = max(1, qcode.N - n_q - 1)
			# nops_q = sc.special.comb(qcode.N, n_q, exact=True) * n_q
		norm_coeff[n_q] += p_q
		for __ in range(nops_q):
			support = tuple(sorted((random.sample(range(qcode.N), n_q))))
			if n_q == 0:
				rand_unitary = np.array([[1.0]])
			else:
				rand_unitary = rc.RandomUnitary(rotation_angle/(2**n_q), 2 ** n_q, method="exp")
			kraus_dict[kraus_count] = (support, np.array([rand_unitary * 1 / np.sqrt(nops_q)]))
			kraus_count += 1

	norm_coeff /= np.sum(norm_coeff)
	# print("norm_coeff = {}".format(norm_coeff))
	# Renormalize kraus ops
	for key, (support, krauslist) in kraus_dict.items():
		# print("Map: {}: support: {}, krauslist shape: {}".format(key, support, np.shape(krauslist)))
		for k in range(len(krauslist)):
			# print("k = {}".format(k))
			kraus_dict[key][1][k] *= np.sqrt(norm_coeff[len(support)])
	
	supports = [kraus_dict[key][0] for key in kraus_dict]
	print("Range of interactions : {}".format(supports))
	
	return kraus_dict

