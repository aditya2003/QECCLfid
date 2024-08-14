import random
import numpy as np
# import scipy as sp # only for debugging purposes.
from define.QECCLfid.utils import SamplePoisson, get_interactions, RandomSupport, GenerateSupport
from define.randchans import RandomUnitary
# from define import globalvars as gv

def HermitianConjugate(M):
	return M.conj().T

def StineToKraus(U):
	# Compute the Krauss operators for the input quantum channel, which is represented in the Stinespring dialation
	# The Krauss operator T_k is given by: <a|T_k|b> = <a e_k|U|b e_0> , where {|e_i>} is a basis for the environment and |a>, |b> are basis vectors of the system
	# Note that a k-qubit channel needs to be generated from a unitary matrix on 3*k qubits where 2*k qubits represent the environment.
	# We can do this using tensor contraction tools:
	# T_k = <e_k|U|e_0>
	#     = env[k]^\dag . U . env[0]
	# Assuming U is full rank:
	# sys-env = np.reshape(U, [2^n, 2^n, 4^n, 4^n])
	# T_k = eye-env[:, :, k, 0]
	# Note that a k-qubit channel needs to be generated from a unitary matrix on 3*k qubits where 2*k qubits represent the environment.
	nq = int(np.log2(U.shape[0]))//3
	sys_env_split = np.reshape(U, [2**nq, 4**nq, 2**nq, 4**nq])
	kraus = np.einsum('ikj->kij', sys_env_split[:, :, :, 0])
	return kraus

def CorrelatedCPTP(rotation_angle, qcode, cutoff = 3, n_maps = 3, mean = 1, isUnitary = 0):
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
	# print("Sum of CPTP maps:\nmean = {}, cutoff = {}, n_maps = {}".format(mean, cutoff, n_maps))
	interaction_range = get_interactions(n_maps, mean, cutoff)
	# supports = [(q, ) for q in range(qcode.N)] + GenerateSupport(qcode.N, interaction_range, cutoff=cutoff)
	# supports = [(q, ) for q in range(qcode.N)] + RandomSupport(qcode.N, interaction_range)
	supports = [(q, ) for q in range(qcode.N)] + [(i, j) for i in range(1, qcode.N // 2 + 1) for j in range(i)] + RandomSupport(qcode.N, interaction_range)
	# supports = RandomSupport(qcode.N, interaction_range)

	# supports = [(0, 5, 4), (5, 1, 3), (3, 0, 6), (0, 5, 1), (1, 4, 0), (6,), (1, 3), (4,)]
	# # If the Kraus list is empty, then append the identity error on some qubit.
	if len(supports) == 0:
		print("!!! Warning: no maps were generated. Forcing random single qubit interactions.")
		supports = [(q, ) for q in range(qcode.N)]
	
	interaction_range = [len(supp) for supp in supports]
	# print("Range of interactions : {}".format(interaction_range))
	
	non_trivial_channels = [None for ir in interaction_range]
	for m in range(len(interaction_range)):
		n_q = interaction_range[m]

		# For random Unitary channels: set the only Kraus operator to be the random Unitary on n_q qubits.
		if (isUnitary == 1):
			# print("Rotation angle for map {} is {}.".format(m, rotation_angle))
			# rand_unitary = RandomUnitary(rotation_angle / np.power(2, n_q), np.power(2, n_q), method="exp")
			# rand_unitary = RandomUnitary(rotation_angle / n_q, np.power(2, n_q), method="exp")
			# scaling_factor = np.power(0.1, np.random.normal(n_q-1, 1))
			rand_unitary = RandomUnitary(rotation_angle * np.random.uniform(1, 10), np.power(2, n_q), method="exp")
			kraus = rand_unitary[np.newaxis, :, :]
			#############
			# Only for debugging purposes:
			# print("Map {} is a rotation on {} about the Z axis".format(m, supports[m]))
			# unitary_mat = sp.linalg.expm(-1j * 0.3 * np.array([[1, 0], [0, -1]], dtype = np.complex128))
			# kraus = unitary_mat[np.newaxis, :, :]
			#############
		else:
			if (n_q <= 2):
				rand_unitary = RandomUnitary(rotation_angle * np.random.uniform(10,100), np.power(8, n_q), method="exp")
			else:
				rand_unitary = RandomUnitary(rotation_angle, np.power(8, n_q), method="exp")
			
			kraus = StineToKraus(rand_unitary)

		if (KrausTest(kraus) == 0):
			print("Kraus test failed for the following channel.\n{}".format(kraus))

		non_trivial_channels[m] = (supports[m], kraus)
	print("Random channel generated with the following {} interactions\n{}.".format(len(interaction_range), supports))

	return non_trivial_channels


def HamiltonainTwoBody(time, qcode, cutoff = 3, n_maps = 3, mean = 1):
	r"""
	Generates random 2-body interaction Hamiltonians along with the supports to ensure that each qubit participates in the highest number of interactions.
	Input :
	time = Time for which Hamiltonian evolution takes place. This is a handle on the noise strength. We will mutiply each Hamiltonian by this time factor.
	qcode = QEC
	n_maps = number of 2-body interactions to be considered
	Returns :
	dict[key] = (support, Hamiltonains)
	where key = number associated to the operation applied (not significant)
	support = tuple describing which qubits the kraus ops act on
	Hamiltonians = 4 x 4 Hermitian matrices that serve as interactions.
	"""
	supports = [(i, j) for j in range(i, qcode.N) for j in range(qcode.N)]
	n_interactions = len(supports)
	interaction_hamiltonians = {m:None for m in range(n_interactions)}
	for m in range(n_interactions):
		ham = RandomHermitian(4) * time
		interaction_hamiltonians[m] = (supports[m], ham)
	print("Random Hamiltonain with {} 2-body interactions:\n{}.".format(n_interactions, supports))

	return interaction_hamiltonians


def KrausTest(kraus):
	# Given a set of Kraus operators {K_i}, check if \sum_i [ (K_i)^dag K_i ] = I.
	total = np.zeros((kraus.shape[1], kraus.shape[2]), dtype = np.complex128)
	for k in range(kraus.shape[0]):
		# print("||K_{} - diag(K_{})||_2 = {}".format(k, k, np.linalg.norm(kraus[k, :, :] - np.diag(np.diag(kraus[k, :, :])))))
		total = total + np.dot(HermitianConjugate(kraus[k, :, :]), kraus[k, :, :])
	success = 0
	if np.allclose(total, np.eye(kraus.shape[1], dtype=np.complex128), atol=1E-7):
		success = 1
		# print("Kraus test passed.")
	else:
		print("sum_i [ (K_i)^dag K_i ]\n{}".format(total))
	return success

if __name__ == "__main__":
	# Test the channel generation code.
	# from define import qcode as qec
	# qcode = qec.QuantumErrorCorrectingCode("Steane")
	# channels = CorrelatedCPTP(0.1, qcode, cutoff = 3, n_maps = 10)
	nmaps = 8
	nqubits = 7
	interaction_range = []
	for m in range(nmaps):
		n_q = SamplePoisson(mean=1, cutoff=4)
		if (n_q > 0):
			interaction_range.append(n_q)
	nmaps = len(interaction_range)
	supports = GenerateSupport(nqubits, interaction_range)
	print("interaction range: {}\nSupport = {}".format(interaction_range, supports))