import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm
from timeit import default_timer as timer
from define.QECCLfid.utils import extend_operator, get_interactions, RandomSupport, GenerateSupport, check_hermiticity, PauliTensor
from define.randchans import RandomHermitian, RandomPauli
from define import globalvars as gv
from define.qcode import PrepareSyndromeLookUp

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
	if qcode.group_by_weight is None:
		PrepareSyndromeLookUp(qcode)

	dim = np.power(2, qcode.N, dtype = int)
	H = np.zeros((dim, dim), dtype = np.complex128)
	biases = [0, mu, J]
	for w in range(1, 3):
		local_terms = qcode.PauliOperatorsLST[qcode.group_by_weight[w]]
		weight_contribution = np.zeros((dim, dim), dtype = np.complex128)
		print("Weight {} terms get multiplied by {}".format(w, biases[w]))
		for i in range(local_terms.shape[0]):
			pauli_op = local_terms[i, :]
			weight_contribution = weight_contribution + PauliTensor(pauli_op).reshape(dim, dim)

		# Normalize H to norm 1.
		weight_contribution = weight_contribution / np.linalg.norm(weight_contribution)
		
		H = H + biases[w] * weight_contribution

	kraus = expm(-1j * time * H)
	kraus_dict = [(tuple(list(range(qcode.N))), kraus[np.newaxis, :, :])]
	return kraus_dict


def CG1DModelPauli(angle, cutoff, qcode):
	r"""
	
	Unitary error U derived from exponentiating a Hamiltonian H: U = exp(i H theta)
	where H is a sum of local terms: H = h1 + h2 + ... + hM
		  M : cutoff
		  h1 = sum of single body Pauli terms with uniform random Coefficients
		  h2 = sum of two-body Pauli terms with uniform random Coefficients
		  ...
		  hM = sum of M-body Pauli terms
	"""
	
	if qcode.group_by_weight is None:
		PrepareSyndromeLookUp(qcode)

	dim = np.power(2, qcode.N, dtype = int)
	H = np.zeros((dim, dim), dtype = np.complex128)
	for w in range(1, cutoff + 1):
		# print("Weight {} terms".format(w))
		local_terms = qcode.PauliOperatorsLST[qcode.group_by_weight[w]]
		weight_contribution = np.zeros((dim, dim), dtype = np.complex128)
		coeffs = np.random.uniform(size=(local_terms.shape[0],))
		for i in range(local_terms.shape[0]):
			pauli_op = local_terms[i, :]
			# print("c = {}, P = {}".format(coeffs[i], pauli_op))
			weight_contribution = weight_contribution + coeffs[i] * PauliTensor(pauli_op).reshape(dim, dim)

		# Normalize the total contribution from weight-w terms to norm 1.
		weight_contribution = weight_contribution / np.linalg.norm(weight_contribution)
		H = H + 1/w * weight_contribution
	
	kraus = expm(-1j * angle * H)
	kraus_dict = [(tuple(list(range(qcode.N))), kraus[np.newaxis, :, :])]
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
	interaction_range = get_interactions(n_factors, mean_correlation_length, cutoff)
	supports = [(q, ) for q in range(nqubits)] + RandomSupport(nqubits, interaction_range)
	# supports = [(q, ) for q in range(nqubits)] + [(i, j) for i in range(nqubits) for j in range(i)]
	 # + RandomSupport(nqubits, interaction_range)
	# supports = [(q, ) for q in range(nqubits)] + [(i, j) for i in range(nqubits) for j in range(i)]
	# supports = GenerateSupport(nqubits, interaction_range, cutoff=cutoff)
	# supports = [(0, 1), (1, 2, 3)] # only for debugging purposes
	interaction_range = [len(supp) for supp in supports]

	print("CG1D Model describing a Hamiltonian acting on {}.".format(supports))

	# local_terms = [np.loadtxt("/home/pavi/Documents/IQC/chbank/cg1d/test/test_cg1d_split/h1.txt"), np.loadtxt("/home/pavi/Documents/IQC/chbank/cg1d/test/test_cg1d_split/h2.txt")]
	# Compute the global Hamiltonain H as a sum of random local terms.
	H = np.zeros((dim, dim), dtype = np.complex128)
	for m in range(len(interaction_range)):
		dim = np.power(2, interaction_range[m], dtype = int)
		# local_term = RandomHermitian(dim)
		local_term = RandomPauli(interaction_range[m])
		# local_term = local_terms[m]
		# local_term = np.array([[1, 0], [0, -1]], dtype = np.complex128) # Z rotation on qubit m. Only for debugging purposes.
		extended_operator = extend_operator(np.array(supports[m], dtype=int), local_term, nqubits)
				
		H = H + angle * extended_operator

	# check_hermiticity(H, "Hermiticity of H: H - H^dag")
	# Exponentiate the Hamiltonian
	# start = timer()
	# angle = 0.3 # only for debugging purposes
	# kraus = expm(-1j * angle * H) # only for debugging purposes
	# print("Unitarity of Kraus\n{}".format(np.linalg.norm(np.dot(kraus, kraus.conj().T) - np.eye(kraus.shape[0]))))
	kraus = expm(-1j * H)
	kraus_dict = [(tuple(list(range(nqubits))), kraus[np.newaxis, :, :])]
	# print("Kraus of dimensions {} corresponding to the CG1D was computed in {} seconds.".format(kraus.shape, timer() - start))
	return kraus_dict