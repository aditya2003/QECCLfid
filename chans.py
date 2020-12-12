from define.QECCLfid.sum_unitaries import SumUnitaries
from define.QECCLfid.ising import Ising
from define.QECCLfid.sum_cptps import SumCptps
from define.QECCLfid.multi_qubit_kraus import get_process_correlated, get_chi_diagLST

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
	chi = get_chi_diagLST(qcode, kraus_dict)
	process = get_process_correlated(qcode, kraus_dict)
	# Check if the i,j element of the channel is j,i element of the adjoint channel.
	# for key, (support, krauslist) in kraus_dict.items():
	# 	for k in range(len(krauslist)):
	# 		kraus_dict[key][1][k] = Dagger(kraus_dict[key][1][k])
	# process_adj = get_process_correlated(qcode, kraus_dict)
	# print("process - process_adj: {}".format(np.allclose(process.reshape(256, 256), process_adj.reshape(256, 256).T)))
	# print("Process[0] = {}".format(process[0]))
	print("\033[2mInfidelity = %.4e.\033[0m" % (1 - chi[0]))
	return (process, chi)