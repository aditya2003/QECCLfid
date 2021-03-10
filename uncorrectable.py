import numpy as np
from define.QECCLfid import minwt as mw
from define.QECCLfid import clique as cq
from define.qcode import PrepareSyndromeLookUp


def ComputeUnCorrProb(pauli_probs, qcodes, nlevels, leading_fraction=0, method=None, misc=None):
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
        for qcode in qcodes:
            if qcode.lookup is None:
                PrepareSyndromeLookUp(qcode)
        return mw.ComputeUncorrProbs(pauli_probs, qcodes, nlevels, leading_fraction, misc)
    elif method == "maxclique":
        # Consider getting rid of this feature
        # Not updated for different codes across levels
        return cq.ComputeUncorrProbs(pauli_probs, qcodes[0])
    else:
        print("Invalid method %s. Use 'maxclique' or 'minwt'." % (method))
    return None
