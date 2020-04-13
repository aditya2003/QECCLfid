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


def checkNSmembership(PauliSym, Q):
    r"""
    Checks if a given PauliSym commutes with all of the stabilizers
    Input : Pauli In symplectic form
    Example : For XZY, sym = {"sx":[1,0,1],"sz":[0,1,1]}
    Output : True if Pauli is in N(S), False otherwise
    """
    return not (any(list(map(lambda sym: inner_sym(sym, PauliSym), Q.SSym))))


def FindMaxClique(prodInNS, nNodes):
    r"""
    Takes a list of all Paulis in N(S) and returns a maximum clique of correctable errors
    Input : ProdInNS : List of tuples (i,j) such that product of Pauli_i and Pauli_j is in N(S)
            nNodes : Total number of errors considered
    Output : One possible maximal list of correctable errors
    """
    import networkx as nx
    from networkx.algorithms.approximation import clique

    G = nx.complete_graph(nNodes)
    for (i, j) in prodInNS:
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


def ComputeUnCorrProb(qcode, pauliProbs, levels=1):
    r"""
    Generates all Paulis of weight k and k+1,checks their membership in N(S),
    generates the clique from the set not in N(S) and returns a list of pauli errors
    that are correctable
    """
    k = 1
    PauliWtKandk1 = GenPauliWtK(qcode, k) + GenPauliWtK(qcode, k + 1)
    print("Number of 1 and 2 qubit errors: ", len(PauliWtKandk1))
    prodInNS = []
    for i in range(len(PauliWtKandk1)):
        PauliE = PauliWtKandk1[i]
        for j in range(i + 1, len(PauliWtKandk1)):
            PauliF = PauliWtKandk1[j]
            PauliProdEF = prod_sym(PauliE, PauliF)
            if checkNSmembership(PauliProdEF, qcode):
                prodInNS.append((i, j))
    cliqueG = FindMaxClique(prodInNS, len(PauliWtKandk1))
    Paulis_correctable = list(
        map(
            convert_symplectic_to_Pauli,
            list(map(PauliWtKandk1.__getitem__, list(cliqueG))),
        )
    )
    PaulisCorrectableIndices = list(
        map(lambda op: qcode.GetPositionInLST(op), Paulis_correctable)
    )
    print(
        "Number of correctable 1 and 2 qubit errors : ", len(PaulisCorrectableIndices)
    )

    return 1 - AdjustToLevel(pauliProbs[PaulisCorrectableIndices].sum(), qcode, levels)


# if __name__ == "__main__":
#     qcode = qc.QuantumErrorCorrectingCode("7qc_cyclic")
#     qc.Load(qcode)
#     qc.populate_symplectic(qcode)
#     pauliProbs = np.random.rand(4 ** qcode.N)
#     UncorrProb = ComputeUnCorrProb(qcode, pauliProbs / pauliProbs.sum())
#     print("Upper bound on probability of uncorrecatble errors is : ", UncorrProb)
