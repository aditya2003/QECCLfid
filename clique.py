import numpy as np
import itertools as it
from scipy.special import comb


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


def ComputeCorrectableIndices(qcode, method="minwt"):
    r"""
    Compute the indices of correctable errors in a code.
    """
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


def ComputeUncorrProbs(pauliProbs, qcode):
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
