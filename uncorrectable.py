import numpy as np
from define.QECCLfid import minwt as mw
from define.QECCLfid import clique as cq


def ComputeUnCorrProb(pauli_probs, qcode, nlevels, leading_fraction=0, method=None):
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
        return mw.ComputeUncorrProbs(pauli_probs, qcode, nlevels, leading_fraction)
    elif method == "maxclique":
        return cq.ComputeUncorrProbs(pauli_probs, qcode)
    else:
        print("Invalid method %s. Use 'maxclique' or 'minwt'." % (method))
    return None
