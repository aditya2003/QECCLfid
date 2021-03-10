import numpy as np
from define.QECCLfid import utils as ut
from define import qcode as qc
from define.decoder import CompleteDecoderKnowledge


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


# Compute residuals done differently
# def ComputeResidualLogicalProbabilities(pauli_probs, qcode, lookup=None):
#     """
#     Given the probability vector for all Paulis, compute the probability
#     of obtaining a residual logical, assuming the decoder acts as given by the lookup.
#     """
#     if lookup is None:
#         lookup = qcode.lookup
#     prob_logicals = np.zeros(4 ** qcode.K, dtype=np.double)
#     nstabs = 2 ** (qcode.N - qcode.K)
#     nlogs = 4 ** qcode.K
#     ordering = np.array([[0, 3], [1, 2]], dtype=np.int8)
#     mult = np.array(
#         [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]], dtype=np.int8
#     )
#
#     for l in range(nlogs):
#         lgens = np.array(
#             list(map(np.int8, np.binary_repr(l, width=(2 * qcode.K)))), dtype=np.int8
#         )
#         for t in range(nstabs):
#             correction_logical = int(lookup[t, 0])
#             product_logical = mult[ordering[lgens[0], lgens[1]], correction_logical]
#             for s in range(nstabs):
#                 prob_logicals[product_logical] += pauli_probs[
#                     ordering[lgens[0], lgens[1]] * nstabs * nstabs + s * nstabs + t
#                 ]
#     return prob_logicals


def ComputeResiduals(L_s, pauli_probs, qcode, lookup=None):
    """
    Compute decoder degeneracies
    """
    if lookup is None:
        lookup = qcode.lookup
    ordering = np.array(([0, 3], [1, 2]), dtype=np.int)
    nstabs = 2 ** (qcode.N - qcode.K)
    probls = 0
    for t in range(nstabs):
        L_t = int(lookup[t, 0])
        pos_L = qc.PauliProduct(
            np.array([L_t], dtype=np.int), np.array([L_s], dtype=np.int)
        )[0]
        indices = pos_L * nstabs * nstabs + np.arange(nstabs) * nstabs + t
        probls += np.sum(pauli_probs[indices])
    return probls


def ComputeUncorrProbs(probs, qcodes, nlevels, leading_fraction, recompute=None):
    r"""
    Computes uncorr
    """
    # print("Probs = {}".format(probs))
    for qcode in qcodes:
        if (recompute == True):
            if qcode.PauliCorrectableIndices is None:
                qc.ComputeCorrectableIndices(qcode)
        else:
            if (recompute == True):
                qc.ComputeCorrectableIndices(qcode)
        # print("Pure errors\n{}".format(qcode.T))
        # print("Correctable error representatives for the {} code:\n{}\nCorrectable errors\n{}".format(qcode.name, qcode.lookup, qcode.Paulis_correctable))
        if qcode.decoder_degens is None:
            ComputeDecoderDegeneracies(qcode)
    if probs.ndim == 2:
        pauli_probs = np.prod(
            probs[range(qcodes[0].N), qcodes[0].PauliOperatorsLST[:, range(qcodes[0].N)]], axis=1
        )
    else:
        pauli_probs = probs
        # print(
        #     "P\n{}\nsum(P) = {}\nP(I) = {}".format(
        #         pauli_probs, np.sum(pauli_probs), pauli_probs[0]
        #     )
        # )
    if leading_fraction > 0 and leading_fraction < 1:
        pauli_probs = CompleteDecoderKnowledge(leading_fraction, pauli_probs, qcodes[0])
    coset_probs = np.zeros(4, dtype=np.double)
    # Here we can use the details of the level-2 code if they're not the same as the level-1 code.
    correctable_probabilities = np.zeros(nlevels, dtype=np.double)
    # print("Uncorr: nlevels = %d" % (nlevels))
    for l in range(nlevels):
        # print("pauli_probs = {}".format(pauli_probs[pauli_probs <= 0]))
        if l == 0:
            # Get level-0 contribution -- this is just the probability of I.
            correctable_probabilities[l] = pauli_probs[0]
        elif l == 1:
            # Get level-1 contribution
            # These are all errors that are corected by the level-1 code.
            correctable_probabilities[l] = np.sum(
                pauli_probs[qcodes[0].PauliCorrectableIndices]
            )
            # correctable_probabilities[1] = 1 - sum(
            #     [pauli_probs[p] for p in qcode.PauliCorrectableIndices]
            # )
        elif l == 2:
            # This is computed in two steps.
            # First we need to account for errors that are completely removed by the level-1 code.
            # print("at l = 2\nqcodes[l - 2].N = {}".format(qcodes[l - 2].N))
            correctable_probabilities[l] = np.power(
                correctable_probabilities[l - 1], qcodes[l-1].N
            )
            # Then we need to account for errors that are mapped to a correctable pattern of logical errors, for the level-2 code to correct.
            for log in range(4):
                coset_probs[log] = ComputeResiduals(log, pauli_probs, qcodes[l-2])
            # print("Coset probs = {}".format(coset_probs))
            correctable_probabilities[l] += np.sum(
                np.prod(coset_probs[qcodes[l-1].Paulis_correctable[1:]], axis=1)
            )
            # print("Correctable probabilities\n{}".format(correctable_probabilities[l]))
        elif l == 3:
            # This is also computed in two steps.
            # First we need to account for errors are completely removed by the level-2 code.
            correctable_probabilities[l] = np.power(
                correctable_probabilities[l - 1], qcodes[l-1].N
            )
            # Then we need to account for errors that are mapped to a correctable pattern of logical errors, for the level-3 code to correct.
            # Update coset probs recursively
            pauli_probs = np.prod(
                np.tile(coset_probs, [qcodes[l-2].N, 1])[range(qcodes[l-2].N), qcodes[l-2].PauliOperatorsLST[:, range(qcodes[l-2].N)]], axis=1
            )
            for log in range(4):
                coset_probs[log] = ComputeResiduals(log, pauli_probs, qcodes[l-2])

            correctable_probabilities[l] += np.sum(
                np.prod(coset_probs[qcodes[l-1].Paulis_correctable[1:]], axis=1)
            )

        else:
            pass
    print("uncorrectable_probabilities\n{}".format(1 - correctable_probabilities))
    return 1 - correctable_probabilities
