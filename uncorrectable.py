import numpy as np
import itertools as it
from scipy.special import comb


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


def checkmembershipInNormalizer(PauliSym, Q, type="stab"):
    r"""
    Checks if a given PauliSym commutes with all of the stabilizers or logicals depending on the input
    Input : Pauli In symplectic form, qcode, Normalizer of "stab" or "logical"
    Example : For XZY, sym = {"sx":[1,0,1],"sz":[0,1,1]}
    Output : True if Pauli is in N(S) or N(L), False otherwise
    """
    if type == "stab":
        return not (any(list(map(lambda sym: inner_sym(sym, PauliSym), Q.SSym))))
    elif type == "logical":
        return not (any(list(map(lambda sym: inner_sym(sym, PauliSym), Q.LSym))))
    else:
        print("Invalid type")


def FindMaxClique(prodInNSminusS, nNodes):
    r"""
    Takes a list of all Paulis in N(S) and returns a maximum clique of correctable errors
    Input : prodInNSminusS : List of tuples (i,j) such that product of Pauli_i and Pauli_j is in N(S)\S
            nNodes : Total number of errors considered
    Output : One possible maximal list of correctable errors
    """
    import networkx as nx
    from networkx.algorithms.approximation import clique

    G = nx.complete_graph(nNodes)
    for (i, j) in prodInNSminusS:
        G.remove_edge(i, j)
    return clique.max_clique(G)


def GenPauliWtK(Q, k):
    r"""
    Generates all pauli errors of weight 'k' for Q.N qubits
    Input : Q - QECC, k - weight of errors desired
    Output : List of Paulis in their symplectic form
    """
    n = Q.N
    bitstrings = []
    for bits in it.combinations(range(n), k):
        s = ["0"] * n
        for bit in bits:
            s[bit] = "1"
        bitstrings.append(s)

    Paulis = list(
        map(
            convert_Pauli_to_symplectic,
            it.chain(
                *list(
                    map(
                        lambda s: list(
                            it.product(
                                *list(map(lambda x: [1, 2, 3] if x == "1" else [0], s))
                            )
                        ),
                        bitstrings,
                    )
                )
            ),
        )
    )

    return Paulis


def AdjustToLevel(pi, qcode, levels):
    r"""
    Takes the correctable errors probability at the lowest level
    and calculates its equivalent for a concatenated code
    """
    t = (qcode.D - 1) // 2
    for i in range(1, levels):
        sum = pi
        for k in range(1, t + 1):
            sum = sum + comb(qcode.N, k) * (pi ** k) * ((1 - pi) ** (qcode.N - k))
        pi = sum
    # print("pi = {}, levels = {}".format(pi, levels))
    return pi


def get_syndrome(pauli, Q):
    r"""
    Returns the syndrome string for a given pauli
    """
    syndr = [inner_sym(pauli, S) for S in Q.SSym]
    return "".join([str(elem) for elem in syndr])


def ComputeUnCorrProb(pauliProbs, qcode, method=None):
    r"""
    Given a list of Pauli probabilities corresponding to a noise process and a list of qcodes,
    it estimates uncorrectable probability using the chosen method. The default method is
    calculating correctable errors using minimum weight.
    Input : list of probabilities for various noise process, qcode, number of levels, method = "minwt" or "maxclique"
    """
    if method == None:
        # Can insert fancy selection here later
        method = "minwt"
    if method == "minwt":
        return ComputeUnCorrProbUsingMinWt(pauliProbs, qcode)
    elif method == "maxclique":
        return ComputeUnCorrProbUsingClique(pauliProbs, qcode)
    else:
        print("Invalid method. Use 'maxclique' or 'minwt'")
    return None


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


def ComputeUnCorrProbUsingMinWt(pauliProbs, qcode):
    r"""
    Generates all Paulis of weight 0,1 and 2
    Assigns syndromes with their correspoding lowest weight errors
    """
    if qcode.PauliCorrectableIndices is None:
        ComputeCorrectableIndices(qcode, method="minwt")
    if pauliProbs.ndim == 1:
        probs = pauliProbs
    else:
        probs = {
            qcode.PauliCorrectableIndices[p]: np.prod(
                [
                    pauliProbs[q, qcode.Paulis_correctable[p][q]]
                    for q in range(pauliProbs.shape[0])
                ]
            )
            for p in range(len(qcode.PauliCorrectableIndices))
        }
    return 1 - sum([probs[p] for p in qcode.PauliCorrectableIndices])


def ComputeUnCorrProbUsingClique(pauliProbs, qcode):
    r"""
    Generates all Paulis of weight k and k+1,checks their membership in N(S),
    generates the clique from the set not in N(S) and returns a list of pauli errors
    that are correctable
    """
    if qcode.PauliCorrectableIndices is None:
        ComputeCorrectableIndices(qcode, method="clique")
        # print("Correctable 1 and 2 qubit errors : {}".format(qcode.Paulis_correctable))
    if pauliProbs.ndim == 1:
        probs = pauliProbs
    else:
        probs = {
            qcode.PauliCorrectableIndices[p]: np.prod(
                [
                    pauliProbs[q, qcode.Paulis_correctable[p, q]]
                    for q in range(pauliProbs.shape[0])
                ]
            )
            for p in range(len(qcode.PauliCorrectableIndices))
        }
    return 1 - sum([probs[p] for p in qcode.PauliCorrectableIndices])


def ComputeCorrectableIndices(qcode, method="minwt"):
    r"""
    Compute the indices of correctable errors in a code.
    """
    if method == "minwt":
        k = 1
        id_error = [{"sx": [0] * qcode.N, "sz": [0] * qcode.N}]
        # PauliWtKandk1 = id_error + GenPauliWtK(qcode, k) + GenPauliWtK(qcode, k + 1)
        # dict_syndr = {}
        # total_syndromes = 2 ** (qcode.N - qcode.K)
        # syndrome_count = 0
        # for pauli in PauliWtKandk1:
        #     syndrome = get_syndrome(pauli, qcode)
        #     if syndrome not in dict_syndr:
        #         dict_syndr[syndrome] = pauli
        #         syndrome_count += 1
        #         if syndrome_count == total_syndromes:
        #             break
        minwt_reps = list(map(convert_Pauli_to_symplectic, qcode.lookup[:, 2:]))
        degeneracies = [
            prod_sym(unique_rep, stab)
            for unique_rep in minwt_reps
            for stab in qcode.SGroupSym
        ]
        qcode.Paulis_correctable = list(map(convert_symplectic_to_Pauli, degeneracies))
        qcode.PauliCorrectableIndices = list(
            map(lambda op: qcode.GetPositionInLST(op), qcode.Paulis_correctable)
        )
        # print("Correctable 1 and 2 qubit errors : {}".format(qcode.Paulis_correctable))
    else:
        k = 1
        id_error = [{"sx": [0] * qcode.N, "sz": [0] * qcode.N}]
        PauliWtKandk1 = id_error + GenPauliWtK(qcode, k) + GenPauliWtK(qcode, k + 1)

        # print("Number of 1 and 2 qubit errors: ", len(PauliWtKandk1))
        prodInNSminusS = []
        for i in range(len(PauliWtKandk1)):
            PauliE = PauliWtKandk1[i]
            for j in range(i + 1, len(PauliWtKandk1)):
                PauliF = PauliWtKandk1[j]
                PauliProdEF = prod_sym(PauliE, PauliF)
                if checkmembershipInNormalizer(PauliProdEF, qcode, "stab"):
                    if not checkmembershipInNormalizer(PauliProdEF, qcode, "logical"):
                        prodInNSminusS.append((i, j))
        cliqueG = FindMaxClique(prodInNSminusS, len(PauliWtKandk1))
        qcode.Paulis_correctable = list(
            map(
                convert_symplectic_to_Pauli,
                list(map(PauliWtKandk1.__getitem__, list(cliqueG))),
            )
        )
        qcode.PauliCorrectableIndices = list(
            map(lambda op: qcode.GetPositionInLST(op), qcode.Paulis_correctable)
        )
    return None
