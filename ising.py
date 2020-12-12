import numpy as np
from scipy.linalg import expm
from define.QECCLfid.utils import extend_gate
from define import globalvars as gv

def Ising(J, mu, time, qcode):
	r"""
	Sub-routine to prepare the dictionary for errors arising due to Ising type interaction
	https://en.wikipedia.org/wiki/Transverse-field_Ising_model
	H = - J \sum_{i} Z_i Z_{i+1} - mu \sum_{i} X_i
	Returns :
	dict[key] = (support,krauslist)
	where key = number associated to the operation applied (not significant)
	support = tuple describing which qubits the kraus ops act on
	krauslist = krauss ops acting on support
	"""
	ZZ = np.kron(gv.Pauli[3], gv.Pauli[3])
	if qcode.interaction_graph is None:
		if qcode.name == "Steane":
			# Color code triangle graph
			connections = [(0,5),(0,1),(1,6),(5,6),(4,5),(3,4),(3,6),(2,3),(1,2)]
			connections_rev = [(y,x) for (x,y) in connections]
			qcode.interaction_graph = np.array(connections + connections_rev)
		else:
			# Asssume nearest neighbour in numerically sorted order
			qcode.interaction_graph = np.array([(i,(i+1)%qcode.N) for i in range(qcode.N)],dtype=np.int8)

	Ham = np.zeros(2**qcode.N, dtype = np.double)
	for (i,j) in qcode.interaction_graph :
		Ham = Ham + J * extend_gate([i,j], ZZ, np.arange(qcode.N, dtype=np.int))
	if mu > 0:
		for i in range(qcode.N):
			Ham = Ham + mu * extend_gate([i], gv.Pauli[1], np.arange(qcode.N, dtype=np.int))
	kraus = expm(-1j * time * Ham)
	# print("Unitarity of Kraus\n{}".format(np.linalg.norm(np.dot(kraus, kraus.conj().T) - np.eye(kraus.shape[0]))))
	kraus_dict = {0:(tuple(range(qcode.N)), [kraus])}
	return kraus_dict