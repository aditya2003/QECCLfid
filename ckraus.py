import numpy as np
from scipy.linalg import expm
import define.globalvars as gv
from define.qcode import PauliOperatorToMatrix, ComputeCorrectableIndices, PrepareSyndromeLookUp

def AdversarialRotKraus(qcode, angle):
	# Design an adversarial error model to show performance degradation for QEC in the RC setting.
	# The adversarial error model is a background unitary (rotation about an arbitrary non Pauli axis) and a rotation about an axis defined by the sum of a few correctable errors.
	# support = (0, 2, 4, 6)
	kraus_dict = {}
	# n_corr = 2
	# syndrome = 1
	# nstabs = 2**(qcode.N - qcode.K)
	# if qcode.Paulis_correctable is None:
	# 	PrepareSyndromeLookUp(qcode)
	# 	ComputeCorrectableIndices(qcode)
	# paulis = qcode.Paulis_correctable[(syndrome*nstabs):(syndrome*nstabs + n_corr)]
	# print("Paulis taken : {}".format(paulis))
	# axis_rot = np.zeros((2**len(support), 2**len(support)), dtype = np.complex128)
	# for i in range(n_corr):
	# 	axis_rot += PauliOperatorToMatrix(paulis[i,list(support)])
	# kraus_dict[0] = (support, [expm(-1j * angle * np.pi * axis_rot)])
	for q in range(qcode.N):
		kraus_dict[q] = ((q,), [expm(-1j * angle * np.pi * gv.Pauli[3])])
	return kraus_dict