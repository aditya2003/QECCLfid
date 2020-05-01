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
    return pi


def get_syndrome(pauli, Q):
    r"""
    Returns the syndrome string for a given pauli
    """
    syndr = [inner_sym(pauli, S) for S in Q.SSym]
    return "".join([str(elem) for elem in syndr])


def ComputeUnCorrProb(pauliProbsList, qcode, levels=1, method=None):
    r"""
    Given a list of Pauli probabilities corresponding to a noise process and a list of qcodes,
    it estimates uncorrectable probability using the chosen method. The default method is
    calculating correctable errors using minimum weight.
    Input : list of probabilities for various noise process, qcode, number of levels, method = "minwt" or "maxclique"
    """
    if method == None:
        # Can insert fancy selection here later
        method = "minwt"
    uncorrectable_probs = np.zeros(len(pauliProbsList))
    if method == "minwt":
        for i in range(len(pauliProbsList)):
            pauliProbs = pauliProbsList[i]
            uncorrectable_probs[i] = ComputeUnCorrProbUsingMinWt(
                pauliProbs, qcode, levels
            )
        return uncorrectable_probs
    elif method == "maxclique":
        for i in range(pauliProbsList.shape(0)):
            pauliProbs = pauliProbsList[i]
            uncorrectable_probs[i] = ComputeUnCorrProbUsingClique(
                pauliProbs, qcode, levels
            )
        return uncorrectable_probs
    else:
        print("Invalid method. Use 'maxclique' or 'minwt'")
    return None


def ComputeUnCorrProbUsingMinWt(pauliProbs, qcode, levels=1):
    r"""
    Generates all Paulis of weight 0,1 and 2
    Assigns syndromes with their correspoding lowest weight errors
    """
    if qcode.PauliCorrectableIndices is None:
        k = 1
        id_error = [{"sx": [0] * qcode.N, "sz": [0] * qcode.N}]
        PauliWtKandk1 = id_error + GenPauliWtK(qcode, k) + GenPauliWtK(qcode, k + 1)
        dict_syndr = {}
        total_syndromes = 2 ** (qcode.N - qcode.K)
        syndrome_count = 0
        for pauli in PauliWtKandk1:
            syndrome = get_syndrome(pauli, qcode)
            if syndrome not in dict_syndr:
                dict_syndr[syndrome] = pauli
                syndrome_count += 1
                if syndrome_count == total_syndromes:
                    break

        qcode.Paulis_correctable = list(
            map(convert_symplectic_to_Pauli, list(dict_syndr.values()))
        )
        qcode.PauliCorrectableIndices = list(
            map(lambda op: qcode.GetPositionInLST(op), qcode.Paulis_correctable)
        )
        # print("Correctable 1 and 2 qubit errors : {}".format(qcode.Paulis_correctable))

        if pauliProbs.shape[0] == 4 ** qcode.N:
            probs = pauliProbs
        else:
            probs = {
                qcode.PauliCorrectableIndices[p]: np.prod(
                    pauliProbs[qcode.Paulis_correctable[p]]
                )
                for p in range(len(qcode.PauliCorrectableIndices))
            }
        return 1 - AdjustToLevel(
            sum([probs[p] for p in qcode.PauliCorrectableIndices]), qcode, levels
        )


def ComputeUnCorrProbUsingClique(pauliProbs, qcode, levels=1):
    r"""
    Generates all Paulis of weight k and k+1,checks their membership in N(S),
    generates the clique from the set not in N(S) and returns a list of pauli errors
    that are correctable
    """
    if qcode.PauliCorrectableIndices is None:
        k = 1
        id_error = [{"sx": [0] * qcode.N, "sz": [0] * qcode.N}]
        PauliWtKandk1 = id_error + GenPauliWtK(qcode, k) + GenPauliWtK(qcode, k + 1)

        print("Number of 1 and 2 qubit errors: ", len(PauliWtKandk1))
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
        # print("Correctable 1 and 2 qubit errors : {}".format(qcode.Paulis_correctable))
    if pauliProbs.shape[0] == 4 ** qcode.N:
        probs = pauliProbs
    else:
        probs = {
            qcode.PauliCorrectableIndices[p]: np.prod(
                pauliProbs[qcode.Paulis_correctable[p]]
            )
            for p in range(len(qcode.PauliCorrectableIndices))
        }
    # print(
    #     "probs: {}\ntotal: {}\n====".format(
    #         probs, sum([probs[p] for p in qcode.PauliCorrectableIndices])
    #     )
    # )
    return 1 - AdjustToLevel(
        sum([probs[p] for p in qcode.PauliCorrectableIndices]), qcode, levels
    )
