import random
import numpy as np
import cvxpy as cp
from collections import deque
from functools import reduce
from define.QECCLfid.contract import OptimalEinsum
import string
# from numba import njit

def ConvertToDecimal(digits, base):
	# Convert a number represented in a given base to its decimal form (base 10).
	decimal = np.sum(digits[::-1] * np.power(base, np.arange(digits.size, dtype = np.int64), dtype = np.int64))
	return decimal

def PauliTensor(pauli_op):
	# Convert a Pauli in operator form to a tensor.
	# The tensor product of A B .. Z is given by simply putting the rows indices of A, B, ..., Z together, followed by their column indices.
	# Each qubit index q can be assigned a pair of labels for the row and columns of the Pauli matrix on q: C[2q], C[2q + 1].
	# Pauli matrices
	characters = string.printable[10:]
	# replace the following line with the variable from globalvars.py
	Pauli = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=np.complex128)
	nq = len(pauli_op)
	labels = ",".join(["%s%s" % (characters[2 * q], characters[2 * q + 1]) for q in range(nq)])
	ops = [Pauli[pauli_op[q], :, :] for q in range(nq)]
	kn_indices = ["%s" % (characters[2 * q]) for q in range(nq)]
	kn_indices += ["%s" % (characters[2 * q + 1]) for q in range(nq)]
	kn_label = "".join(kn_indices)
	scheme = "%s->%s" % (labels, kn_label)
	pauli_tensor = OptimalEinsum(scheme, ops)
	return pauli_tensor

# @njit("uint8[:](uint64, uint64)")
def GetNQubitPauli(ind, nq):
	# Compute the n-qubit Pauli that is at position 'i' in an ordering based on [I, X, Y, Z].
	# We will express the input number in base 4^n - 1.
	pauli = np.zeros(nq, dtype = np.int64)
	for i in range(nq):
		pauli[i] = ind % 4
		ind = int(ind//4)
	return pauli[::-1]

def SamplePoisson(mean, cutoff=None):
	# Sample a number from a Poisson distribution.
	# If the random variable takes a value above a cutoff, sample again until it returns a value below the cutoff.
	if (cutoff == None):
		rv_val = np.random.poisson(lam = mean)
	else:
		rv_val = cutoff + 1
		while (rv_val > cutoff):
			rv_val = np.random.poisson(lam = mean)
	return rv_val

def GenerateSupport(nqubits, interaction_ranges, cutoff=4):
	r"""
	Generates supports for maps such that
	1. Each qubit participates in at least one maps
	2. Every map has support given by the number of qubits in interaction_ranges list
	3. Allocation optimized such that each qubit participates roughly in nmaps_per_qubit maps
	returns a list of tuples with support indicies for each map
	"""
	nmaps = len(interaction_ranges)
	# A matrix of variables, where each row corresponds to an interaction while each column to a qubit.
	# The (i,j) entry of this matrix is 1 if the i-th interaction involves the j-th qubit.
	mat = cp.Variable(shape=(nmaps, nqubits), boolean = True)

	# These are hard constraints.
	constraints = []
	# Each qubit to be part of at least one map.
	col_sums = cp.sum(mat, axis=0, keepdims=True)
	constraints.append(col_sums >= 1)
	# Each interaction must involve a fixed number of qubits +/- 1.
	row_sums = cp.sum(mat, axis=1)
	constraints.append(row_sums <= [min(r + 1, cutoff) for r in interaction_ranges])
	constraints.append(row_sums >= [max(1, r - 1) for r in interaction_ranges])

	# Objective function to place a penalty on the number of interactions per qubit.
	# objective = cp.Minimize(cp.norm(col_sums, "fro"))
	objective = cp.Minimize(cp.norm(col_sums, "inf"))

	# Solve the optimization problem.
	problem = cp.Problem(objective,constraints)
	problem.solve(solver = 'ECOS_BB', verbose=False)

	if ("optimal" in problem.status):
		if (not (problem.status == "optimal")):
			print("\033[2mWarning: The problem status is \"{}\".\033[0m".format(problem.status))
		supports = [tuple(np.nonzero(np.round(row).astype(np.int64))[0]) for row in mat.value]
	else:
		print("\033[2mQubit allocation to maps infeasible.\033[0m")
		# Choose random subsets of qubits of the specified sizes.
		supports = []
		for m in range(nmaps):
			support = tuple((random.sample(range(nqubits), interaction_ranges[m])))
			supports.append(support)

	return supports

def RandomSupport(nqubits, interaction_ranges):
	# Choose random subsets of qubits of the specified sizes.
	supports = []
	for m in range(len(interaction_ranges)):
		support = tuple((random.sample(range(nqubits), interaction_ranges[m])))
		supports.append(support)
	return supports

def get_interactions(n_maps, mean, cutoff):
	# Sample the sizes of interactions that constitute a channel.
	# The sizes are to be samples from a Poisson distribution with a fixed mean.
	interaction_range = []
	for m in range(n_maps):
		n_q = SamplePoisson(mean = mean, cutoff=cutoff)
		if n_q != 0:
			interaction_range.append(n_q)
	return interaction_range

def check_hermiticity(M, message):
	# Check if M - M^dag = 0
	print("{}: {}".format(message, M - M.conj().T))
	return None

def extend_operator(support_qubits, operator, nqubits):
	r"""
	# Given an operator defined on a subspace, extend it to a larger dimension.
	Given some support S and local hamiltonian h
	1. Compute the complementary support S'
	2. do h -> h_1 = h \otimes I_{complementary space}
	3. h_1 is supported on S + S'
	4. Compute a permutation to S + S' to set it to the order we want: P = argsort(S + S')
	5. Reshape h_1 to [2, 2] * N
	6. Apply Permutation P to the row indices and the same permutation to the column
		Applying np.transpose(h_1.reshape([2,2]*N), [P + P])
	7. Reshape back.
	"""
	dim = np.power(2, nqubits, dtype=int)
	# check_hermiticity(operator, "Hermiticity of the local operator")

	# Extend the operator by padding an identity matrix of the appropriate dimension.
	complement_qubits = np.setdiff1d(range(nqubits), support_qubits)
	complement_dim = np.power(2, nqubits - len(support_qubits), dtype = int)
	extended_unordered = np.kron(operator, np.eye(complement_dim) / np.sqrt(complement_dim)).reshape([2, 2] * nqubits)
	# check_hermiticity(extended_unordered.reshape(dim, dim), "Hermiticity of the extended unordered operator")
	
	# print("axes labels = {}".format(np.concatenate((support_qubits, complement_qubits, nqubits + support_qubits, complement_qubits + nqubits))))
	# Order the axes according to the qubits in the system.
	unordered_axes = np.concatenate((support_qubits, complement_qubits, nqubits + support_qubits, complement_qubits + nqubits))
	ordering = np.argsort(unordered_axes)
	extended_operator = extended_unordered.transpose(ordering)
	
	# check_hermiticity(extended_operator.reshape(dim, dim), "Hermiticity of the extended operator")
	return extended_operator.reshape(dim, dim)

def extend_gate(support, mat, extended_support):
	r"""
	Extend a gate supported on some qubits in support, to a set of qubit labels, by padding identities.

	Let the given gate :math:`G` have support on

	.. math::
		:nowrap:

		\begin{gather*}
		\vec{s} = s_{1} s_{2} \ldots s_{m}
		\end{gather*}

	and the extended support be

	.. math::
		:nowrap:

		\begin{gather*}
		\vec{e} = e_{1} e_{2} \ldots e_{N}
		\end{gather*}

	- Let :math:`\Gamma \gets \mathbb{I}`

	- For each :math:`i` in 1 to :math:`m`

			- Let :math:`j~\gets` index of :math:`s_{i}` of :math:`\vec{e}`

			- For :math:`k = j-1` to :math:`i`, do

			- Let :math:`d \gets i - (j-1)`

			- Let :math:`\Gamma \gets \Gamma \cdot SWAP_{k, k+d}.`

	- return

	.. math::
		:nowrap:

		\begin{gather*}
		\Gamma \cdot (G \otimes \mathbb{I}^{\otimes (N - m)}) \cdot \Gamma^{-1}
		\end{gather*}
	"""
	SWAP = np.array(
		[[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
	)
	#     print("Extend {} on {} to {}".format(mat, support, extended_support))
	support_range = np.array(
		[(np.argwhere(extended_support == s)).item() for s in support], dtype=np.int64
	)
	# print("support_range = {}".format(support_range))
	if (np.ptp(support_range)).item() < 2:
		# print(
		#     "M = I_(%d) o G o I_(%d)"
		#     % (np.min(support_range), extended_support.size - 1 - np.max(support_range))
		# )
		return Kron(
			np.identity(2 ** np.min(support_range)),
			Dot(SWAP, mat, SWAP.T) if (support_range[0] > support_range[-1]) else mat,
			np.identity(2 ** (extended_support.size - 1 - np.max(support_range))),
		)
	swap = np.identity(2 ** extended_support.size, dtype=np.double)
	for i in range(len(support)):
		j = (np.argwhere(extended_support == support[i])).item()
		d = np.sign(i - j)
		# print("bring {} to {} in direction {}".format(j, i, d))
		if not (d == 0):
			for k in range(j, i, d):
				swap = Dot(
					swap,
					extend_gate(
						[extended_support[k], extended_support[k + d]],
						SWAP,
						extended_support,
					),
				)
	# print("M = G o I_(%d)" % (extended_support.size - len(support)))
	return Dot(
		swap,
		Kron(mat, np.identity(2 ** (extended_support.size - len(support)))),
		swap.T,
	)


def Dagger(x):
	return np.conj(np.transpose(x))


def Kron(*mats):
	"""
	Extend the standard Kronecker product, to a list of matrices, where the Kronecker product is recursively taken from left to right.
	"""
	if len(mats) < 2:
		return mats[0]
	return np.kron(mats[0], Kron(*(mats[1:])))


def Dot(*mats):
	"""
	Extend the standard Dot product, to a list of matrices, where the Kronecker product is recursively taken from left to right.
	"""
	if len(mats) < 2:
		return mats[0]
	return np.dot(mats[0], Dot(*(mats[1:])))


def circular_shift(li, start, end, direction="right"):
	r"""
	Circular shifts a part of the list `li` between start and end indices
	"""
	d = deque(li[start : end + 1])
	if direction == "right":
		d.rotate(1)
	else:
		d.rotate(-1)
	return li[:start] + list(d) + li[end + 1 :]


def GetErrorBudget(probs, nPaulis):
	# Get the total probability of leading non-trivial Pauli decays.
	if probs.ndim == 1:
		chan_probs = probs
	else:
		chan_probs = reduce(np.kron, probs)
	budget = np.sum(np.sort(chan_probs[1:])[-nPaulis:])
	return budget/(1-chan_probs[0])



def inner_sym(sym1, sym2, d=2):
	r"""
	Gives the symplectic inner product modulo 'd'
	Assumption : sym1,sym2 are dictionary with keys "sx","sz" and values as binary list
	Eg: For XZY : sym = {"sx":[1,0,1],"sz":[0,1,1]}
	"""
	return np.mod(
		np.inner(sym1["sx"], sym2["sz"]) - np.inner(sym1["sz"], sym2["sx"]), d
	)


def prod_sym(sym1, sym2, d=2):
	r"""
	Gives the product of two Paulis
	In symplectic form this corresponds to addition modulo 'd'
	Assumption : sym1,sym2 are dictionary with keys "sx","sz" and values as binary list
	Eg: For XZY : sym = {"sx":[1,0,1],"sz":[0,1,1]}
	"""
	return {
		"sx": [np.mod(sum(x), d) for x in zip(sym1["sx"], sym2["sx"])],
		"sz": [np.mod(sum(x), d) for x in zip(sym1["sz"], sym2["sz"])],
	}


def convert_Pauli_to_symplectic(listOfPaulis):
	r"""
	Return a dictionary with symplectic components for a list of Paulis
	Example : Pauli Labelling : 0-I,1-X,2-Y,3-Z
	Input : List of Paulis [3,0,1,2]
	Output : {"sx":[0,0,1,1],"sz":[1,0,0,1]}
	"""
	listSx = list(map(lambda x: 1 if x == 1 or x == 2 else 0, listOfPaulis))
	listSz = list(map(lambda x: 1 if x == 2 or x == 3 else 0, listOfPaulis))
	return {"sx": listSx, "sz": listSz}


def convert_symplectic_to_Pauli(sym):
	r"""
	Takes a dictionary with symplectic components and returns a list of Paulis
	Example : Pauli Labelling : 0-I,1-X,2-Y,3-Z
	Input : {"sx":[0,0,1,1],"sz":[1,0,0,1]}
	Output : List of Paulis [3,0,1,2]
	"""
	ListSx = sym["sx"]
	ListSz = sym["sz"]
	listOfPaulis = []
	for i in range(len(ListSx)):
		if ListSx[i] == 0 and ListSz[i] == 0:
			listOfPaulis.append(0)
		elif ListSx[i] == 1 and ListSz[i] == 0:
			listOfPaulis.append(1)
		elif ListSx[i] == 1 and ListSz[i] == 1:
			listOfPaulis.append(2)
		else:
			listOfPaulis.append(3)
	return listOfPaulis


def get_syndrome(pauli, Q):
	r"""
	Returns the syndrome string for a given pauli
	"""
	syndr = [inner_sym(pauli, S) for S in Q.SSym]
	return "".join([str(elem) for elem in syndr])


def GetChiElements(pauli_operators, chimat, iscorr):
	"""
	Get elements of the Chi matrix corresponding to a set of Pauli operators.
	"""
	nqubits = pauli_operators.shape[1]
	if iscorr == 0:
		return GetErrorProbabilities(
			pauli_operators, np.tile(chimat, [nqubits, 1, 1]), 2
		)
	if iscorr == 1:
		return pauliprobs
	# In this case the noise is a tensor product of single qubit chi matrices.
	chipart = np.zeros(
		(pauli_operators.shape[0], pauli_operators.shape[0]), dtype=np.complex128
	)
	for i in range(operator_probs.shape[0]):
		for j in range(operator_probs.shape[0]):
			chipart[i, j] = 0
	return None


def GetErrorProbabilities(pauli_operators, pauliprobs, iscorr):
	"""
	Compute the probabilities of a set of Pauli operators.
	"""
	nqubits = pauli_operators.shape[1]
	if iscorr == 0:
		# In this case, the noise channel on n qubits is the tensor product of identical copies of a single qubit channel. Hence only one 4-component Pauli probability vector is given.
		return GetErrorProbabilities(pauli_operators, np.tile(pauliprobs, [nqubits, 1]), 2)
	if iscorr == 1:
		return pauliprobs
	# In this case, the noise channel on n qubits is the tensor product of non-identical copies of a single qubit channel. Hence n 4-component Pauli probability vectors are given.
	operator_probs = np.zeros(pauli_operators.shape[0], dtype=np.double)
	for i in range(operator_probs.shape[0]):
		operator_probs[i] = np.prod(
			[pauliprobs[q][pauli_operators[i, q]] for q in range(nqubits)]
		)
	return operator_probs


if __name__ == '__main__':
	# Testing GetNQubitPauli
	GetNQubitPauli(172, 4)

	# Testing PauliTensor
	N = 3
	pauli_op = np.random.randint(0, high=4, size=(N,))
	print("Pauli operator: {}".format(pauli_op))
	tn_pauli = PauliTensor(pauli_op)
	Pauli = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=np.complex128)
	np_pauli = Kron(*Pauli[pauli_op, :, :])
	print("PauliTensor - Numpy = {}".format(np.allclose(tn_pauli.reshape(2**N, 2**N), np_pauli)))
