import random
import numpy as np
import cvxpy as cp
from define.QECCLfid.utils import SamplePoisson
from define.randchans import RandomUnitary

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


def GenerateSupport(nmaps, nqubits, qubit_occupancies):
	r"""
	Generates supports for maps such that
	1. Each qubit participates in at least one maps
	2. Every map has support given by the number of qubits in qubit_occupancies list
	3. Allocation optimized such that each qubit participates roughly in nmaps_per_qubit maps
	returns a list of tuples with support indicies for each map
	"""
	# A matrix of variables, where each row corresponds to an interaction while each column to a qubit.
	# The (i,j) entry of this matrix is 1 if the i-th interaction involves the j-th qubit.
	mat = cp.Variable(shape=(nmaps,nqubits), boolean = True)
	
	# These are hard constraints.
	constraints = []
	# Each qubit to be part of at least one map.
	col_sums = cp.sum(mat, axis=0, keepdims=True)
	constraints.append(col_sums >= 1)
	# Each interaction must involve a fixed number of qubits.
	row_sums = cp.sum(mat, axis=1)
	constraints.append(row_sums == qubit_occupancies)
	
	# Objective function to place a penalty on the number of interactions per qubit.
	objective = cp.Minimize(cp.norm(col_sums, "fro"))
	
	# Solve the optimization problem.
	problem = cp.Problem(objective,constraints)
	# print("Available solvers\n{}".format(cp.installed_solvers()))
	problem.solve(solver='ECOS_BB', verbose=False)
	print("problem\n{}".format(problem.status))
	if ("optimal" in problem.status):
	    supports = [tuple(np.nonzero(np.round(row).astype(np.int))[0]) for row in mat.value]
	else:
	    print("Qubit allocation to maps infeasible.")
	    supports = None
	
	return supports


def CorrelatedCPTP(rotation_angle, qcode, cutoff = 3, n_maps = 3):
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
	n_nontrivial_maps = 0
	interaction_range = []
	for m in range(n_maps):
		n_q = SamplePoisson(mean = 1, cutoff=cutoff)
		# n_q = 3 # Only for decoding purposes.
		if n_q != 0:
			interaction_range.append(n_q)
			n_nontrivial_maps += 1
	print("Range of interactions : {}".format(interaction_range))

	# If the Kraus list is empty, then append the identity error on some qubit.
	if n_nontrivial_maps == 0:
		non_trivial_channels = {0: ((0,), [np.eye(2, dtype = np.complex128)])}
	else:
		# nmaps_per_qubit = max(0.1 * n_nontrivial_maps, 1)
		supports = GenerateSupport(n_nontrivial_maps, qcode.N, interaction_range)
		if (supports is None):
			supports = []
			for m in range(n_nontrivial_maps):
				support = tuple((random.sample(range(qcode.N), interaction_range[m])))
				supports.append(support)
		# supports = [(0, 1), (1, 2), (2, 3), (3, 4)] # Only for decoding purposes.
		# supports = [(0, 1, 2)] # Only for decoding purposes.

		non_trivial_channels = {m:None for m in range(n_nontrivial_maps)}
		for m in range(n_nontrivial_maps):
			n_q = interaction_range[m]
			rand_unitary = RandomUnitary(rotation_angle/8**n_q, 8**n_q)
			kraus = StineToKraus(rand_unitary)

			if (KrausTest(kraus) == 0):
				print("Kraus test failed for the following channel.\n{}".format(kraus))

			non_trivial_channels[m] = (supports[m], kraus)
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
	return success

if __name__ == "__main__":
	# Test the channel generation code.
	from define import qcode as qec
	qcode = qec.QuantumErrorCorrectingCode("Steane")
	channels = CorrelatedCPTP(0.1, qcode, cutoff = 3, n_maps = 10)
