import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm
from timeit import default_timer as timer
from define.QECCLfid.utils import extend_operator, get_interactions, RandomSupport, GenerateSupport, check_hermiticity
from define.randchans import RandomHermitian
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
	# J *= -1 # This creats an antiferromagnetic ground state.
	# ZZ = np.kron(gv.Pauli[3], gv.Pauli[3])
	YY = np.kron(gv.Pauli[3], gv.Pauli[3])
	if qcode.interaction_graph is None:
		qcode.interaction_graph = np.array([(i, (i + 1) % qcode.N) for i in range(qcode.N)], dtype=np.int8)
		# if qcode.name == "Steane":
		# 	# Color code triangle graph
		# 	connections = [(0,5),(0,1),(1,6),(5,6),(4,5),(3,4),(3,6),(2,3),(1,2)]
		# 	connections_rev = [(y,x) for (x,y) in connections]
		# 	qcode.interaction_graph = np.array(connections + connections_rev)
		# else:
		# 	# Asssume nearest neighbour in numerically sorted order
		# 	qcode.interaction_graph = np.array([(i, (i + 1) % qcode.N) for i in range(qcode.N)], dtype=np.int8)

	Ham = np.zeros(2**qcode.N, dtype = np.double)
	for (i,j) in qcode.interaction_graph :
		# Ham = Ham + J * extend_gate([i,j], ZZ, np.arange(qcode.N, dtype=np.int64))
		Ham = Ham + J * extend_gate([i,j], YY, np.arange(qcode.N, dtype=np.int64))
	if mu > 0:
		for i in range(qcode.N):
			Ham = Ham + mu * extend_gate([i], gv.Pauli[1], np.arange(qcode.N, dtype=np.int64))
	kraus = expm(-1j * time * Ham)
	# print("Unitarity of Kraus\n{}".format(np.linalg.norm(np.dot(kraus, kraus.conj().T) - np.eye(kraus.shape[0]))))
	kraus_dict = {0:(tuple(range(qcode.N)), [kraus])}
	return kraus_dict

def CG1DModel(n_factors, angle, mean_correlation_length, cutoff, nqubits):
	r"""
	
	Noise model corresponding to the coarse grained 1D error model in https://arxiv.org/pdf/2303.00780.pdf.
	
	We will choose some random subsets of qubits using the get_interactions and the RandomSupport functions in utils.
	Once we identify some subsets of qubits, we will define a random local Hamiltonian on each subset.
	The sum of the random local Hamiltonians is the Hamiltonian of the entrie system: H.
	Exponentiating H as exp(i H t) gives us an unitary error model.

	Returns : a dictionary of size 1 which contains (support, krauslist)
	support = tuple describing which qubits the kraus ops act on
	krauslist = krauss ops acting on support
	"""
	dim = np.power(2, nqubits, dtype = int)
	# interaction_range = get_interactions(n_factors, mean_correlation_length, cutoff)
	# supports = RandomSupport(nqubits, interaction_range)
	# supports = [(q, ) for q in range(nqubits)] + GenerateSupport(nqubits, interaction_range, cutoff=cutoff)
	supports = [(0,), (1,), (2,), (3,), (4,), (5,), (6,)] # only for debugging purposes
	interaction_range = [len(supp) for supp in supports]

	print("CG1D Model describing a Hamiltonian acting on {}.".format(supports))

	# Compute the global Hamiltonain H as a sum of random local terms.
	H = np.zeros((dim, dim), dtype = np.complex128)
	for m in range(len(interaction_range)):
		dim = np.power(2, interaction_range[m], dtype = int)
		# local_term = RandomHermitian(dim)
		# extended_operator = extend_operator(np.array(supports[m], dtype=int), local_term, nqubits)
		
		local_term = np.array([[1, 0], [0, -1]], dtype = np.complex128) # Z rotation on qubit m. Only for debugging purposes.
		extended_operator = extend_operator(np.array(supports[m], dtype=int), local_term, nqubits)
				
		H = H + extended_operator

	# check_hermiticity(H, "Hermiticity of H: H - H^dag")

	# Exponentiate the Hamiltonian
	start = timer()
	angle = 0.3
	kraus = expm(-1j * angle * H)
	print("Unitarity of Kraus\n{}".format(np.linalg.norm(np.dot(kraus, kraus.conj().T) - np.eye(kraus.shape[0]))))
	kraus_dict = [(tuple(list(range(nqubits))), kraus[np.newaxis, :, :])]
	# print("Kraus of dimensions {} corresponding to the CG1D was computed in {} seconds.".format(kraus.shape, timer() - start))
	return kraus_dict