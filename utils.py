import numpy as np


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
        return GetErrorProbabilities(
            pauli_operators, np.tile(pauliprobs, [nqubits, 1]), 2
        )
    if iscorr == 1:
        return pauliprobs
    # In this case, the noise channel on n qubits is the tensor product of non-identical copies of a single qubit channel. Hence n 4-component Pauli probability vectors are given.
    operator_probs = np.zeros(pauli_operators.shape[0], dtype=np.double)
    for i in range(operator_probs.shape[0]):
        operator_probs[i] = np.prod(
            [pauliprobs[q][pauli_operators[i, q]] for q in range(nqubits)]
        )
    return operator_probs
