import numpy as np
from define.QECCLfid.ising import Ising
from define.QECCLfid.utils import Dagger
from timeit import default_timer as timer
from define.QECCLfid.sum_cptps import SumCptps
from define.QECCLfid.sum_unitaries import SumUnitaries
from define.QECCLfid.multi_qubit_kraus import get_process_correlated, get_chi_diagLST, NoiseReconstruction
from define.QECCLfid.ptm import ConstructPTM


def get_process_chi(qcode, method = "sum_unitaries", *params):
	nstabs = 2 ** (qcode.N - qcode.K)
	nlogs = 4 ** qcode.K
	if method == "sum_unitaries":
		p_error,rotation_angle,w_thresh = params[:3]
		kraus_dict = SumUnitaries(p_error, rotation_angle, qcode, w_thresh)
	elif method == "ising":
		J, mu, time = params[:3]
		kraus_dict = Ising(J, mu, time, qcode)
	elif method == "sum_cptps":
		(angle, cutoff, n_maps) = params[:3]
		kraus_dict = SumCptps(angle, qcode, cutoff = int(cutoff), n_maps = int(n_maps))
	else:
		pass
	# click = timer()
	# print("NEW")
	# chi = NoiseReconstruction(qcode, kraus_dict)
	# print("Done in %d seconds." % (timer() - click))
	chi = None
	click = timer()
	ptm = ConstructPTM(qcode, kraus_dict)
	print("PTM was constructed in %d seconds." % (timer() - click))
	click = timer()
	# for debugging, compare with old version.
	# print("OLD")
	# chi = get_chi_diagLST(qcode, kraus_dict)
	# print("Done in %d seconds." % (timer() - click))
	# print("\033[2mInfidelity = %.4e.\033[0m" % (1 - chi[0]))
	print("====")
	print("OLD")
	click = timer()
	ptm_old = get_process_correlated(qcode, kraus_dict).reshape(256, 256)
	print("Old PTM was constructed in %d seconds." % (timer() - click))
	print("||PTM - Diag(PTM)||_2 = {}".format(np.linalg.norm(ptm_old - np.diag(np.diag(ptm_old)))))
	# Check if the i,j element of the channel is j,i element of the adjoint channel.
	for key, (support, krauslist) in kraus_dict.items():
		for k in range(len(krauslist)):
			kraus_dict[key][1][k] = Dagger(kraus_dict[key][1][k])
	click = timer()
	ptm_adj_old = get_process_correlated(qcode, kraus_dict).reshape(256, 256)
	print("Old adjoint PTM was constructed in %d seconds." % (timer() - click))
	print("ptm_old - ptm_adj_old: {}".format(np.linalg.norm(ptm_old - ptm_adj_old.T)))
	click = timer()
	print("====")

	click = timer()
	ptm_adj = ConstructPTM(qcode, kraus_dict)
	print("Adjoint PTM was constructed in %d seconds." % (timer() - click))
	print("ptm - ptm_adj: {}".format(np.linalg.norm(ptm - ptm_adj.T)))
	print("Process[0, 0] = {}".format(ptm[0, 0]))
	print("||PTM - Diag(PTM)||_2 = {}".format(np.linalg.norm(ptm - np.diag(np.diag(ptm)))))
	exit(0) # for debugging only
	return (ptm, chi)
