import numpy as np
from timeit import default_timer as timer
from define.QECCLfid.utils import Dagger
from define.QECCLfid.utils import Kron
from define.QECCLfid.ising import Ising
from define.QECCLfid.sum_unitaries import SumUnitaries
from define.QECCLfid.cptps import CorrelatedCPTP
from define.QECCLfid.ckraus import AdversarialRotKraus
from define.QECCLfid.multi_qubit_kraus import get_process_correlated, get_chi_diagLST
from define.QECCLfid.chi import NoiseReconstruction
from define.QECCLfid.ptm import ConstructPTM


def GetProcessChi(qcode, method = "sum_unitaries", *params):
	nstabs = 2 ** (qcode.N - qcode.K)
	nlogs = 4 ** qcode.K

	if method == "sum_unitaries":
		(p_error, rotation_angle, w_thresh) = params[:3]
		kraus_dict = SumUnitaries(p_error, rotation_angle, qcode, w_thresh)

	elif method == "ising":
		J, mu, time = params[:3]
		kraus_dict = Ising(J, mu, time, qcode)

	elif method == "corr_cptp":
		(angle, cutoff, n_maps, mean) = params[:4]
		kraus_dict = CorrelatedCPTP(angle, qcode, cutoff = int(cutoff), n_maps = int(n_maps), mean = mean, isUnitary = 0)

	elif method == "corr_unitary":
		(angle, cutoff, n_maps, mean) = params[:4]
		kraus_dict = CorrelatedCPTP(angle, qcode, cutoff = int(cutoff), n_maps = int(n_maps), mean = mean, isUnitary = 1)

	elif method == "correctable_kraus":
		(angle, ratio) = params[:2]
		kraus_dict = AdversarialRotKraus(qcode, angle, ratio)
	else:
		pass

	# Prepare the adjoint channel for PTM checks.
	# kraus_dict_adj = AdjointChannel(kraus_dict)

	click = timer()
	print("\033[2mKraus operators done in %d seconds.\033[0m" % (timer() - click))
	# chi = NoiseReconstruction(qcode, kraus_dict)
	if len(kraus_dict) > 1:
		chi = NoiseReconstruction(qcode, kraus_dict)
	else:
		chi = get_chi_diagLST(qcode, kraus_dict)
	# chi = np.zeros(4**qcode.N, dtype = np.double) # only for debugging
	print("\033[2mCHI was constructed in %d seconds.\033[0m" % (timer() - click))

	click = timer()
	if len(kraus_dict) > 1:
		ptm = ConstructPTM(qcode, kraus_dict)
	else:
		ptm = get_process_correlated(qcode, kraus_dict).reshape(2**(qcode.N + qcode.K), 2**(qcode.N + qcode.K))
	print("\033[2mPTM was constructed in %d seconds.\033[0m" % (timer() - click))
	print("\033[2mProcess[0, 0] = {}\033[0m".format(ptm[0, 0]))

	# if (CHI_PTM_Tests(chi, ptm, kraus_dict, kraus_dict_adj, qcode, compare_against_old = 0) == 0):
	# 	print("PTM test failed.")
	# 	exit(0)

	# Load the interactions
	interactions = []
	for m in kraus_dict:
		interactions.append(kraus_dict[m][0])
	misc_info = (interactions, 1 - chi[0], 1 - np.sum(chi), np.linalg.norm(ptm - np.diag(np.diag(ptm))))
	return (ptm.reshape(-1), chi, misc_info)


def CHI_PTM_Tests(chi, ptm, kraus_dict, kraus_dict_adj, qcode, compare_against_old = 0):
	# Tests for the PTM and CHI matrices.
	success = 1
	atol = 1E-8
	n_maps = len(kraus_dict)

	print("Process[0, 0] = {}".format(ptm[0, 0]))
	if (np.abs(ptm[0, 0] - 1) >= atol):
		success = 0

	print("||PTM - Diag(PTM)||_2 = {}".format(np.linalg.norm(ptm - np.diag(np.diag(ptm)))))

	click = timer()
	ptm_adj = ConstructPTM(qcode, kraus_dict_adj)
	print("Adjoint PTM was constructed in %d seconds." % (timer() - click))
	diff = np.linalg.norm(ptm - ptm_adj.T)
	if (diff >= atol):
		success = 0
	print("||PTM - PTM_adj.T||_2 = {}.".format(diff))

	if (compare_against_old == 1):
		# for debugging, compare with old version.
		print("========")
		print("OLD CHI AND PROCESS MATRICES")
		if (n_maps == 1):
			chi_old = get_chi_diagLST(qcode, kraus_dict)
			print("Old chi matrix was computed in %d seconds." % (timer() - click))
			diff = np.linalg.norm(chi - chi_old)
			print("||CHI - CHI_old||_2 = {}".format(diff))
			if (diff >= atol):
				success = 0
			print("\033[2mInfidelity = %.4e.\033[0m" % (1 - chi[0]))

		click = timer()
		ptm_old = get_process_correlated(qcode, kraus_dict).reshape(256, 256)
		print("Old PTM was constructed in %d seconds." % (timer() - click))
		diff = np.linalg.norm(ptm - ptm_old)
		print("||PTM - PTM_old||_2 = {}".format(diff))
		if (diff >= atol):
			success = 0

		# Check if the i,j element of the channel is j,i element of the adjoint channel.
		click = timer()
		ptm_adj_old = get_process_correlated(qcode, kraus_dict_adj).reshape(256, 256)
		print("Old adjoint PTM was constructed in %d seconds." % (timer() - click))
		diff = np.linalg.norm(ptm_old - ptm_adj_old.T)
		print("||PTM_old - PTM_adj_old.T||_2 = {}".format(diff))
		if (diff >= atol):
			success = 0
		print("========")

		# (mismatches,) = np.nonzero(np.abs(np.diag(ptm) - np.diag(ptm_adj.T)) >= 1E-14)
		# print("Closeness of diagonal of PTM and PTM adjoint: {}.".format(np.allclose(np.diag(ptm), np.diag(ptm_adj))))
		# disagreements = zip(mismatches, np.diag(ptm)[mismatches], np.diag(ptm_adj.T)[mismatches])
		# print("{:<8} {:<8} {:<8}".format("P", "PTM", "PTM*"))
		# for (p, ptm_ii, ptm_adj_ii) in disagreements:
		# 	print("{:<8} {:<8} {:<8}".format("%d" % p, "%.5f" % ptm_ii, "%.5f" % ptm_adj_ii))
	return success


def AdjointChannel(kraus_dict):
	# Compute the adjoint channel.
	n_maps = len(kraus_dict)
	kraus_dict_adj = {}
	for key, (support, krauslist) in kraus_dict.items():
		krauslist_adj = [Dagger(K) for K in krauslist]
		kraus_dict_adj[n_maps - key - 1] = (support, krauslist_adj)
	return kraus_dict_adj
