import numpy as np
from define.QECCLfid import utils as ut
from define import qcode as qc


def ComputeDecoderDegeneracies(qcode):
    """
    Compute decoder degenracies.
    """
    nlogs = 4 ** qcode.K
    nstabs = 2 ** (qcode.N - qcode.K)
    unique_reps = {}
    for l in range(nlogs):
        unique_reps.update({l: np.nonzero(qcode.lookup[:, 0] == l)[0]})
    qcode.decoder_degens = {}
    for l in range(nlogs):
        qcode.decoder_degens.update(
            {
                l: np.array(
                    list(
                        map(
                            lambda t: l * (nstabs * nstabs)
                            + range(nstabs) * nstabs
                            + t,
                            unique_reps[l],
                        )
                    ),
                    dtype=np.int,
                ).ravel()
            }
        )
    return None


def ComputeResiduals(L_s, pauli_probs, qcode):
    """
    Compute decoder degeneracies
    """
    ordering = np.array(([0, 3], [1, 2]), dtype=np.int)
    nstabs = 2 ** (qcode.N - qcode.K)
    probls = 0
    for t in range(nstabs):
        L_t = qcode.lookup[t, 0]
        pos_L = qc.PauliProduct(
            np.array([L_t], dtype=np.int), np.array([L_s], dtype=np.int)
        )[0]
        indices = pos_L * nstabs * nstabs + np.arange(nstabs) * nstabs + t
        probls += np.sum(pauli_probs[indices])
    return probls


def ComputeUncorrProbs(probs, qcode, nlevels):
    r"""
    Generates all Paulis of weight 0,1 and 2
    Assigns syndromes with their correspoding lowest weight errors
    """
    # print("Probs = {}".format(probs))
    if qcode.PauliCorrectableIndices is None:
        ComputeCorrectableIndices(qcode)
    if qcode.decoder_degens is None:
        ComputeDecoderDegeneracies(qcode)
    if probs.ndim == 2:
        pauli_probs = np.prod(
            probs[range(qcode.N), qcode.PauliOperatorsLST[:, range(qcode.N)]], axis=1
        )
    else:
        pauli_probs = probs
        # print(
        #     "P\n{}\nsum(P) = {}\nP(I) = {}".format(
        #         pauli_probs, np.sum(pauli_probs), pauli_probs[0]
        #     )
        # )
    coset_probs = np.zeros(4, dtype=np.double)
    for l in range(4):
        # inds = qcode.decoder_degens[l]
        # coset_probs[l] = np.sum(pauli_probs[inds])
        coset_probs[l] = ComputeResiduals(l, pauli_probs, qcode)
    # Here we can use the details of the level-2 code if they're not the same as the level-1 code.
    correctable_probabilities = np.zeros(nlevels, dtype=np.double)
    for l in range(nlevels):
        if l == 0:
            # Get level-0 contribution -- this is just the probability of I.
            correctable_probabilities[0] = pauli_probs[0]
        elif l == 1:
            # Get level-1 contribution
            # These are all errors that are corected by the level-1 code.
            correctable_probabilities[1] = np.sum(
                pauli_probs[qcode.PauliCorrectableIndices]
            )
            # correctable_probabilities[1] = 1 - sum(
            #     [pauli_probs[p] for p in qcode.PauliCorrectableIndices]
            # )
        elif l == 2:
            # This is computed in two steps.
            # First we need to account for errors that are completely removed by the level-1 code.
            correctable_probabilities[2] = np.power(
                correctable_probabilities[1], qcode.N
            )
            # Then we need to account for errors that are mapped to a correctable pattern of logical errors, for the level-2 code to correct.
            # correctable_probabilities[2] = np.sum(
            #     np.prod(coset_probs[qcode.Paulis_correctable[0:]], axis=1)
            # )
            correctable_probabilities[2] += np.sum(
                np.prod(coset_probs[qcode.Paulis_correctable[1:]], axis=1)
            )
        else:
            pass
    # print("uncorrectable_probabilities\n{}".format(1 - correctable_probabilities))
    return 1 - correctable_probabilities


def ComputeCorrectableIndices(qcode):
    r"""
    Compute the indices of correctable errors in a code.
    """
    minwt_reps = list(map(ut.convert_Pauli_to_symplectic, qcode.lookup[:, 2:]))
    degeneracies = [
        ut.prod_sym(unique_rep, stab)
        for unique_rep in minwt_reps
        for stab in qcode.SGroupSym
    ]
    qcode.Paulis_correctable = np.array(
        list(map(ut.convert_symplectic_to_Pauli, degeneracies)), dtype=np.int
    )
    qcode.PauliCorrectableIndices = np.array(
        list(map(lambda op: qcode.GetPositionInLST(op), qcode.Paulis_correctable)),
        dtype=np.int,
    )
    # print("Pauli correctable indices : {}".format(list(qcode.PauliCorrectableIndices)))
    return None
