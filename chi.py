import time
import numpy as np
import ctypes as ct
import multiprocessing as mp
from define.QECCLfid.contract import ContractTensorNetwork
from define.QECCLfid.theta import KraussToTheta,ThetaToChiElement
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


def PartialDiagChi(coreid, paulis, theta, supp_theta, start, stop, chi_diag_mp):
	# Compute the diagonal Chi matrix elements corresponding to the given Pauli operators.
	# Note that the diagonal elements of the Chi matrix are probabilities.
	print("Core: {}\nPauli operators from {} to {}.".format(coreid, start, stop))
	for i in range(start, stop):
		# print("Core {} paulis[{}]\n{}".format(coreid, i, paulis[i, :]))
		# time.sleep(1)
		chi_diag_mp[i] = np.real(ThetaToChiElement(paulis[i, :], paulis[i, :], theta, supp_theta))
	return None


def Chi_Element_Diag(krausdict, paulis, n_qubits, n_cores=None):
	r"""
	Calculates the diagonal entry in chi matrix corresponding to each Pauli in Pilist
	Assumes each Pauli in list of Paulis Pilist to be a tensor on n_qubits
	Calculates chi_ii = sum_k |<Pi, A_k>|^2
	where A_k is thr Kraus operator and Pi is the Pauli operator
	Kraus dict has the format ("support": list of kraus ops on the support)
	Assumes qubits
	Assumes kraus ops to be square with dim 2**(number of qubits in support)
	"""
	#     Pres stores the addition of all kraus applications
	#     Pi_term stores result of individual kraus applications to Pi
	thetadict = []
	for key, (support, krauslist) in krausdict.items():
		print("Shape of Kraus list : {}".format(np.array(krauslist).shape))
		thetadict.append((support, KraussToTheta(np.array(krauslist))))
	# (supp_theta, theta_contracted) = ContractTensorNetwork(thetadict)
	(supp_theta, theta_contracted) = thetadict[0] # only for debugging purposes.

	# We want to parallelize the computation of chi matrix entries.
	n_errors = paulis.shape[0]
	chi_diag_mp = mp.Array(ct.c_double, n_errors)
	n_cores = 1 # for debugging purposes only.
	if n_cores is None:
		n_cores = mp.cpu_count()
	chunk = int(np.ceil(n_errors / np.float(n_cores)))
	
	print("Chunk size: {}".format(chunk))

	processes = []
	for p in range(n_cores):
		processes.append(mp.Process(target=PartialDiagChi, args=(p, paulis, theta_contracted, supp_theta, p * chunk, min((p + 1) * chunk, n_errors), chi_diag_mp)))
	for p in range(n_cores):
		processes[p].start()
	for p in range(n_cores):
		processes[p].join()

	chi_diag = np.array(chi_diag_mp, dtype = np.double)
	print("Pauli error probabilities:\n{}".format(chi_diag))
	return chi_diag