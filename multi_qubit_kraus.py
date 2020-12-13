import numpy as np

# from define.QECCLfid.multi_qubit_tests import test_get_PTMelem_ij # only for debugging
from define.QECCLfid.utils import Dot, Kron, Dagger, circular_shift
from define.QECCLfid.ptm import get_PTMelem_ij, get_Pauli_tensor, fix_index_after_tensor
from define import qcode as qc


def get_Chielem_ii(krausdict, Pilist, n_qubits):
    r"""
    Calculates the diagonal entry in chi matrix corresponding to each Pauli in Pilist
    Assumes each Pauli in list of Paulis Pilist to be a tensor on n_qubits
    Calculates chi_ii = sum_k |<Pi, A_k>|^2
    where A_k is thr Kraus operator and Pi is the Pauli operator
    Kraus dict has the format ("support": list of kraus ops on the support)
    Assumes qubits
    Assumes kraus ops to be square with dim 2**(number of qubits in support)
    """
    #     Pres stores the addition of all kraus applications
    #     Pi_term stores result of individual kraus applications to Pi
    chi = np.zeros(len(Pilist), dtype=np.double)
    for i in range(len(Pilist)):
        Pi = get_Pauli_tensor(Pilist[i])
        for key, (support, krausList) in krausdict.items():
            indices = support + tuple(map(lambda x: x + n_qubits, support))
            indices_Pi = indices[len(indices) // 2 :]
            for kraus in krausList:
                if len(indices) > 0:
                    kraus_reshape_dims = [2] * (2 * int(np.log2(kraus.shape[0])))
                    indices_kraus = range(len(kraus_reshape_dims) // 2)
                    Pi_term = np.tensordot(
                        Pi,
                        kraus.reshape(kraus_reshape_dims),
                        (indices_Pi, indices_kraus),
                    )
                    Pi_term = fix_index_after_tensor(Pi_term, indices_Pi)
                    # Take trace and absolute value
                    einsum_inds = list(range(len(Pi_term.shape) // 2)) + list(
                        range(len(Pi_term.shape) // 2)
                    )
                    chi[i] += (
                        np.power(np.abs(np.einsum(Pi_term, einsum_inds)), 2)
                        / 4 ** n_qubits
                    )
                else:
                    # Take trace and absolute value
                    if i == 0:
                        chi[i] += np.abs(kraus) ** 2
                    else:
                        chi[i] += 0
    return chi

def get_Chielem_broadcast(krausdict, Pilist, n_qubits):
    r"""
    Calculates the diagonal entry in chi matrix corresponding to each Pauli in Pilist
    Assumes each Pauli in list of Paulis Pilist to be a tensor on n_qubits
    Calculates chi_ii = sum_k |<Pi, A_k>|^2
    where A_k is thr Kraus operator and Pi is the Pauli operator
    Kraus dict has the format ("support": list of kraus ops on the support)
    Assumes qubits
    Assumes kraus ops to be square with dim 2**(number of qubits in support)
    """
    #     Pres stores the addition of all kraus applications
    #     Pi_term stores result of individual kraus applications to Pi
    chi = np.zeros(len(Pilist), dtype=np.double)
    einsum_inds = np.arange(2*n_qubits,dtype=np.int)%n_qubits
    for key, (support, krausList) in krausdict.items():
            indices = support + tuple(map(lambda x: x + n_qubits, support))
            indices_Pi = indices[len(indices) // 2 :]
            for kraus in krausList:
                if len(indices) > 0:
                    kraus_reshape_dims = [2] * (2 * int(np.log2(kraus.shape[0])))
                    indices_kraus = range(len(kraus_reshape_dims) // 2)
                    Pi_terms = map(lambda Pi : np.tensordot(
                        get_Pauli_tensor(Pi),
                        kraus.reshape(kraus_reshape_dims),
                        (indices_Pi, indices_kraus),
                    ),Pilist)
                    Pi_terms = map(lambda Pi_term:fix_index_after_tensor(Pi_term, indices_Pi), Pi_terms)
                    # Take trace and absolute value

                    chi += np.fromiter(map (lambda Pi_term :(
                        np.power(np.abs(np.einsum(Pi_term, einsum_inds)), 2)
                        / 4 ** n_qubits
                    ),Pi_terms),dtype=np.double)
                else:
                    # Take trace and absolute value
                    chi[0] += np.abs(kraus) ** 2
    return chi


def get_process_correlated(qcode, kraus_dict):
    r"""
    Generates LS part of the process matrix for Eps = sum of unitary errors
    p_error^k is the probability associated to a k-qubit unitary (except weight <= w_thresh)
    rotation_angle is the angle used for each U = exp(i*rotation_agnle*H)
    return linearized Pauli transfer matrix matrix in LS ordering for T=0
    """
    nstabs = 2 ** (qcode.N - qcode.K)
    nlogs = 4 ** qcode.K
    (ops, phases) = qc.GetOperatorsForTLSIndex(qcode, range(nstabs * nlogs))
    ops_tensor = list(map(get_Pauli_tensor, ops))
    process = np.zeros(nstabs * nstabs * nlogs * nlogs, dtype=np.double)
    for i in range(len(ops_tensor)):
        process[i * nstabs * nlogs : (i + 1) * nstabs * nlogs] = get_PTMelem_ij(
            kraus_dict, ops_tensor[i], ops_tensor, qcode.N, phases[i], phases
        )
        # print("Test for {}\n{}".format(i, test_get_PTMelem_ij(kraus_dict, ops_tensor[i], ops_tensor, qcode.N)))
    return process


def get_process_diagLST(qcode, kraus_dict):
    r"""
    Generates diagonal of the process matrix in LST ordering for Eps = sum of unitary errors
    p_error^k is the probability associated to a k-qubit unitary (except weight<= w_thresh)
    rotation_angle is the angle used for each U = exp(i*rotation_agnle*H)
    return linearized diagonal of the process matrix in LST ordering
    """
    nstabs = 2 ** (qcode.N - qcode.K)
    nlogs = 4 ** qcode.K
    diag_process = np.zeros(nstabs * nstabs * nlogs, dtype=np.double)
    (ops, phases) = qc.GetOperatorsForLSTIndex(qcode, range(nstabs * nstabs * nlogs))
    for i in range(len(diag_process)):
        op_tensor = get_Pauli_tensor(ops[i]) * phases[i]
        diag_process[i] = get_PTMelem_ij(kraus_dict, op_tensor, [op_tensor], qcode.N)[0]
    return diag_process


def get_chi_diagLST(qcode, kraus_dict):
    r"""
    Generates diagonal of the chi matrix in LST ordering for Eps = sum of unitary errors
    p_error^k is the probability associated to a k-qubit unitary (except weight <= w_thresh)
    rotation_angle is the angle used for each U = exp(i*rotation_agnle*H)
    return linearized diagonal of the chi matrix in LST ordering
    """
    nstabs = 2 ** (qcode.N - qcode.K)
    nlogs = 4 ** qcode.K
    (ops, __) = qc.GetOperatorsForLSTIndex(qcode, range(nstabs * nstabs * nlogs))
    chi = get_Chielem_ii(kraus_dict, ops, qcode.N)
    # print("Sum of chi = {}, infid = {}\nElements of chi\n{}".format(np.sum(chi), 1 - chi[0], np.sort(chi)[::-1]))
    return chi
