import numpy as np
from scipy.linalg import expm
import define.globalvars as gv
from define.qcode import PauliOperatorToMatrix, ComputeCorrectableIndices, PrepareSyndromeLookUp

def AdversarialRotKraus(qcode, angle, ratio):
	# Design an adversarial error model to show performance degradation for QEC in the RC setting.
	# The adversarial error model is a background unitary (rotation about an arbitrary non Pauli axis) and a rotation about an axis defined by the sum of a few correctable errors.
	kraus_dict = {}
	n_corr = 10
	syndrome = 1
	nstabs = 2**(qcode.N - qcode.K)
	if qcode.Paulis_correctable is None:
		PrepareSyndromeLookUp(qcode)
		ComputeCorrectableIndices(qcode)

	# Rotate about a Pauli axis which is sum of n_corr correctable and uncorrectable errors of least weight
	pauli_indices_corr = []
	pauli_indices_uncorr = []
	# Add correctable indices
	for w in range(qcode.N):
		is_correctable_errors = np.in1d(qcode.group_by_weight[w], qcode.PauliCorrectableIndices)
		correctable_errors = qcode.group_by_weight[w][np.nonzero(is_correctable_errors)]
		uncorrectable_errors = qcode.group_by_weight[w][np.nonzero(1 - is_correctable_errors)]
		correctable_errors_synd = correctable_errors[np.where(np.mod(correctable_errors, nstabs) == syndrome)]
		uncorrectable_errors_synd = uncorrectable_errors[np.where(np.mod(uncorrectable_errors, nstabs) == syndrome)]
		pauli_indices_corr.append(correctable_errors_synd)
		pauli_indices_uncorr.append(uncorrectable_errors_synd)

	weights_corr = 1
	weights_uncorr = ratio * 1
	
	corr_indices_taken = np.concatenate(pauli_indices_corr)[: n_corr]
	uncorr_indices_taken = np.concatenate(pauli_indices_uncorr)[: n_corr]
	paulis = qcode.PauliOperatorsLST[np.concatenate([corr_indices_taken, uncorr_indices_taken])]

	print("Paulis taken : {}".format(paulis))
	support = tuple(range(qcode.N))
	axis_rot = np.zeros((2**len(support), 2**len(support)), dtype = np.complex128 )
	for i in range(len(corr_indices_taken)):
		# axis_rot += PauliOperatorToMatrix(paulis[i, list(support)])
		axis_rot += weights_corr * PauliOperatorToMatrix(paulis[i, list(support)])
	offset = len(corr_indices_taken)
	for i in range(len(uncorr_indices_taken)):
		axis_rot += weights_uncorr * PauliOperatorToMatrix(paulis[i + offset, list(support)])
	
	kraus_dict[0] = (support, [expm(-1j * angle * np.pi/2 * axis_rot)])
	# for q in range(qcode.N):
	# 	kraus_dict[q+1] = ((q,), [expm(-1j * angle * np.pi * gv.Pauli[2])])
	return kraus_dict
