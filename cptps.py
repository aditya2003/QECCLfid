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

def generate_support(nmaps, nqubits, nmaps_per_qubit, qubit_occupancies):
	r"""
	Generates supports for maps such that
	1. Each qubit participates in at least one maps
	2. Every map has support given by the number of qubits in qubit_occupancies list
	3. Allocation optimized such that each qubit participates roughly in nmaps_per_qubit maps
	returns a list of tuples with support indicies for each map
	"""
	supports = None
	mat = cp.Variable(shape=(nmaps,nqubits), boolean = True)
	constraints = []
	col_sums = cp.sum(mat, axis=0, keepdims=True)
	constraints.append(col_sums >= 1)
	row_sums = cp.sum(mat, axis=1)
	constraints.append(row_sums == qubit_occupancies)
	objective = cp.Minimize(cp.norm(col_sums-nmaps_per_qubit,"inf"))
	problem = cp.Problem(objective,constraints)
	problem.solve()
	if problem.status not in ["infeasible", "unbounded"]:
	    supports = [tuple(np.nonzero(row)[0]) for row in mat.value]
	else:
	    print("Qubit allocation to maps infeasible.")
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
	qubit_occupancies = []
	for m in range(n_maps):
		n_q = SamplePoisson(mean = 1, cutoff=cutoff)
		if n_q == 0:
			continue
		else:
			qubit_occupancies.append(n_q)
			n_nontrivial_maps += 1
	# print("Qubit occupancies : {}".format(qubit_occupancies))
	# If the Kraus list is empty, then append the identity error on some qubit.
	if n_nontrivial_maps == 0:
		non_trivial_channels = {0: ((0,), [np.eye(2, dtype = np.complex128)])}
	else:
		nmaps_per_qubit = max(0.1*n_nontrivial_maps,1)
		supports = generate_support(n_nontrivial_maps, qcode.N, nmaps_per_qubit, qubit_occupancies)
		if supports is not None:
			non_trivial_channels = {m:None for m in range(n_nontrivial_maps)}
			for m in range(n_nontrivial_maps):
				n_q = qubit_occupancies[m]
				rand_unitary = RandomUnitary(rotation_angle/8**n_q, 8**n_q)
				kraus = StineToKraus(rand_unitary)
				KrausTest(kraus)
				non_trivial_channels[m] = (supports[m], kraus)
			print("Random channel generated with the following {} interactions\n{}.".format(n_nontrivial_maps, supports))
		else:
			print("Qubit allocation failed for a total of {} non-trivial map(s) with each of {} qubits required to participate in {} map(s)".format(n_nontrivial_maps, qcode.N, nmaps_per_qubit))
			exit(0)
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

if __name__ == "__main__":
	# import sys
	# import os
	# from define import qcode as qec
	# codename = sys.argv[1]
	# qcode = qec.QuantumErrorCorrectingCode("%s" % (codename))
	# channels = CorrelatedCPTP(0.1, qcode, cutoff = 3, n_maps = 10)
