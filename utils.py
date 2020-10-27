import numpy as np
from collections import deque
from functools import reduce

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
		[np.asscalar(np.argwhere(extended_support == s)) for s in support], dtype=np.int
	)
	# print("support_range = {}".format(support_range))
	if np.asscalar(np.ptp(support_range)) < 2:
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
		j = np.asscalar(np.argwhere(extended_support == support[i]))
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
