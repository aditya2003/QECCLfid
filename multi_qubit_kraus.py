import numpy as np
from collections import deque
import random
import scipy as sc
import sys
from define import randchans as rc
from define import qcode as qc
from define import globalvars as gv


def extend_gate(support, mat, extended_support):
    r"""
    Extend a gate supported on some qubits in support, to a set of qubit labels, by padding identities.

    Let the given gate :math:`G` have support on

    .. math::
        :nowrap:

        \begin{gather*}
        \vec{s} = s_{1} s_{2} \ldots s_{m}
        \end{gather*}

    and the extended support be

    .. math::
        :nowrap:

        \begin{gather*}
        \vec{e} = e_{1} e_{2} \ldots e_{N}
        \end{gather*}

    - Let :math:`\Gamma \gets \mathbb{I}`

    - For each :math:`i` in 1 to :math:`m`

            - Let :math:`j~\gets` index of :math:`s_{i}` of :math:`\vec{e}`

            - For :math:`k = j-1` to :math:`i`, do

            - Let :math:`d \gets i - (j-1)`

            - Let :math:`\Gamma \gets \Gamma \cdot SWAP_{k, k+d}.`

    - return

    .. math::
        :nowrap:

        \begin{gather*}
        \Gamma \cdot (G \otimes \mathbb{I}^{\otimes (N - m)}) \cdot \Gamma^{-1}
        \end{gather*}
    """
    SWAP = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
    )
    #     print("Extend {} on {} to {}".format(mat, support, extended_support))
    support_range = np.array(
        [np.asscalar(np.argwhere(extended_support == s)) for s in support], dtype=np.int
    )
    # print("support_range = {}".format(support_range))
    if np.asscalar(np.ptp(support_range)) < 2:
        # print(
        #     "M = I_(%d) o G o I_(%d)"
        #     % (np.min(support_range), extended_support.size - 1 - np.max(support_range))
        # )
        return Kron(
            np.identity(2 ** np.min(support_range)),
            dot(SWAP, mat, SWAP.T) if (support_range[0] > support_range[-1]) else mat,
            np.identity(2 ** (extended_support.size - 1 - np.max(support_range))),
        )
    swap = np.identity(2 ** extended_support.size, dtype=np.double)
    for i in range(len(support)):
        j = np.asscalar(np.argwhere(extended_support == support[i]))
        d = np.sign(i - j)
        # print("bring {} to {} in direction {}".format(j, i, d))
        if not (d == 0):
            for k in range(j, i, d):
                swap = dot(
                    swap,
                    extend_gate(
                        [extended_support[k], extended_support[k + d]],
                        SWAP,
                        extended_support,
                    ),
                )
    # print("M = G o I_(%d)" % (extended_support.size - len(support)))
    return dot(
        swap,
        Kron(mat, np.identity(2 ** (extended_support.size - len(support)))),
        swap.T,
    )


def Dagger(x):
    return np.conj(np.transpose(x))


def Kron(*mats):
    """
    Extend the standard Kronecker product, to a list of matrices, where the Kronecker product is recursively taken from left to right.
    """
    if len(mats) < 2:
        return mats[0]
    return np.kron(mats[0], Kron(*(mats[1:])))


def Dot(*mats):
    """
    Extend the standard Dot product, to a list of matrices, where the Kronecker product is recursively taken from left to right.
    """
    if len(mats) < 2:
        return mats[0]
    return np.dot(mats[0], dot(*(mats[1:])))


def circular_shift(li, start, end, direction="right"):
    r"""
    Circular shifts a part of the list `li` between start and end indices
    """
    d = deque(li[start : end + 1])
    if direction == "right":
        d.rotate(1)
    else:
        d.rotate(-1)
    return li[:start] + list(d) + li[end + 1 :]


def get_Pauli_tensor(Pauli):
    r"""
    Takes an n-qubit pauli as a list and converts it to the tensor form
    """
    listOfPaulis = [gv.Pauli[i] for i in Pauli]
    if listOfPaulis:
        Pj = listOfPaulis[0]
        n_qubits = len(listOfPaulis)
        for i in range(1, n_qubits):
            Pj = np.tensordot(Pj, listOfPaulis[i], axes=0)
        indices_Pj = [i for i in range(0, len(Pj.shape), 2)] + [
            i for i in range(1, len(Pj.shape), 2)
        ]
    else:
        raise ValueError(f"Invalid Pauli {Pauli} ")
    return np.transpose(Pj, indices_Pj)


def fix_index_after_tensor(tensor, indices_changed):
    r"""
    Tensor product alters the order of indices. This function helps reorder to fix them back.
    """
    n = len(tensor.shape) - 1
    perm_list = list(range(len(tensor.shape)))
    n_changed = len(indices_changed)
    for i in range(len(indices_changed)):
        index = indices_changed[i]
        perm_list = circular_shift(perm_list, index, n - n_changed + i + 1, "right")
    return np.transpose(tensor, perm_list)


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
            for kraus in krausList:
                if len(indices) > 0:
                    kraus_reshape_dims = [2] * (2 * int(np.log2(kraus.shape[0])))
                    indices_Pi = indices[len(indices) // 2 :]
                    indices_kraus = range(len(kraus_reshape_dims) // 2)
                    #                 Pi_term = np.tensordot(Pi,Dagger(kraus).reshape(kraus_reshape_dims),(indices_Pi,indices_kraus))
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


def get_PTMelem_ij(krausdict, Pi, Pjlist, n_qubits):
    r"""
    Assumes Paulis Pi,Pj to be a tensor on n_qubits
    Calculates Tr(Pj Eps(Pi)) for each Pj in Pjlist
    Kraus dict has the format ("support": list of kraus ops on the support)
    Assumes qubits
    Assumes kraus ops to be square with dim 2**(number of qubits in support)
    """
    #     Pres stores the addition of all kraus applications
    #     Pi_term stores result of individual kraus applications to Pi
    Pres = np.zeros_like(Pi)
    for key, (support, krausList) in krausdict.items():
        indices = support + tuple(map(lambda x: x + n_qubits, support))
        for kraus in krausList:
            if len(indices) > 0:
                kraus_reshape_dims = [2] * (2 * int(np.log2(kraus.shape[0])))
                indices_Pi = indices[len(indices) // 2 :]
                indices_kraus = range(len(kraus_reshape_dims) // 2)
                Pi_term = np.tensordot(
                    Pi,
                    Dagger(kraus).reshape(kraus_reshape_dims),
                    (indices_Pi, indices_kraus),
                )
                Pi_term = fix_index_after_tensor(Pi_term, indices_Pi)
                indices_Pi = indices[: len(indices) // 2]
                indices_kraus = range(len(kraus_reshape_dims))[
                    len(kraus_reshape_dims) // 2 :
                ]
                Pi_term = np.tensordot(
                    Pi_term,
                    kraus.reshape(kraus_reshape_dims),
                    (indices_Pi, indices_kraus),
                )
                Pi_term = fix_index_after_tensor(Pi_term, indices_Pi)
                Pres += Pi_term
            else:
                Pres = Pi * (np.abs(kraus) ** 2)
    # take dot product with Pj and trace
    trace_vals = np.zeros(len(Pjlist), dtype=np.double)
    indices_Pi = list(range(len(Pi.shape) // 2))
    indices_Pj = list(range(len(Pi.shape) // 2, len(Pi.shape)))
    for i in range(len(Pjlist)):
        Pj = Pjlist[i]
        Pres_times_Pj = np.tensordot(Pres, Pj, (indices_Pi, indices_Pj))
        # Take trace
        einsum_inds = list(range(len(Pres_times_Pj.shape) // 2)) + list(
            range(len(Pres_times_Pj.shape) // 2)
        )
        trace_vals[i] = np.real(np.einsum(Pres_times_Pj, einsum_inds)) / 2 ** n_qubits
    return trace_vals


def get_kraus_ising(J, mu, qcode):
    r"""
    Sub-routine to prepare the dictionary for errors arising due to Ising type interaction
    H = - J \sum_{i} Z_i Z_{i+1} - mu \sum_{i} Z_i
    Returns :
    dict[key] = (support,krauslist)
    where key = number associated to the operation applied (not significant)
    support = tuple describing which qubits the kraus ops act on
    krauslist = krauss ops acting on support
    """
    kraus_dict = {}
    ZZ = np.kron(gv.Paulis[3],gv.Paulis[3])
    if qcode.interaction_graph is None:
        # Asssume nearest neighbour in numerically sorted order
        qcode.interaction_graph = np.array([(i,i+1) for i in range(qcode.N-1)],dtype=np.int8)
    Ham = np.zeros(2**qcode.N, dtype = np.double)
    for (i,j) in qcode.interaction_graph :
        Ham = Ham + J * extend_gate([i,j], ZZ, range(qcode.N))
    for i in range(qcode.N):
        Ham = Ham + mu*extend_gate([i], gv.Paulis[3], range(qcode.N))
    kraus = linalg.expm(-1j  * Ham)
    kraus_dict.append(0:(list(range(qcode.N)),[kraus]))
    return kraus_dict


def get_kraus_random(p_error, rotation_angle, qcode, w_thresh=3):
    r"""
    Sub-routine to prepare the dictionary for error eps = sum of unitary errors
    Generates kraus as random multi-qubit unitary operators rotated by rotation_angle
    Probability associated with each weight-k error scales exponentially p_error^k (except weights <= w_thresh)
    Returns :
    dict[key] = (support,krauslist)
    where key = number associated to the operation applied (not significant)
    support = tuple describing which qubits the kraus ops act on
    krauslist = krauss ops acting on support
    """
    kraus_dict = {}

    kraus_count = 0
    norm_coeff = np.zeros(qcode.N + 1, dtype=np.double)
    # for n_q in range(qcode.N + 1):
    for n_q in range(1, qcode.N + 1): # Removed identity kraus
        if n_q == 0:
            p_q = 1 - p_error
        elif n_q <= w_thresh:
            p_q = np.power(0.1, n_q - 1) * p_error
        else:
            p_q = np.power(p_error, n_q)
        # Number of unitaries of weight n_q is max(1,qcode.N-n_q-1)
        if n_q == 0:
            nops_q = 1
        else:
            nops_q = max(1, (qcode.N + n_q) // 2) # Aditya's version
            # nops_q = sc.special.comb(qcode.N, n_q, exact=True) * n_q
        norm_coeff[n_q] += p_q
        for __ in range(nops_q):
            support = tuple(sorted((random.sample(range(qcode.N), n_q))))
            if n_q == 0:
                rand_unitary = 1.0
            else:
                rand_unitary = rc.RandomUnitary(
                    rotation_angle/(2**n_q), 2 ** n_q, method="exp"
                ) # Aditya version
                # rand_unitary = rc.RandomUnitary(
                #     rotation_angle/(n_q), 2 ** n_q, method="exp"
                # )
            kraus_dict[kraus_count] = (support, [rand_unitary * 1 / np.sqrt(nops_q)])
            kraus_count += 1

    norm_coeff /= np.sum(norm_coeff)
    # Renormalize kraus ops
    for key, (support, krauslist) in kraus_dict.items():
        for k in range(len(krauslist)):
            kraus_dict[key][1][k] *= np.sqrt(norm_coeff[len(support)])
    return kraus_dict


def get_process_correlated(qcode, kraus_dict):
    r"""
    Generates LS part of the process matrix for Eps = sum of unitary errors
    p_error^k is the probability associated to a k-qubit unitary (except weight <= w_thresh)
    rotation_angle is the angle used for each U = exp(i*rotation_agnle*H)
    return linearized Pauli transfer matrix matrix in LS ordering for T=0
    """
    nstabs = 2 ** (qcode.N - qcode.K)
    nlogs = 4 ** qcode.K
    ops = qc.GetOperatorsForTLSIndex(qcode, range(nstabs * nlogs))
    ops_tensor = list(map(get_Pauli_tensor, ops))
    process = np.zeros(nstabs * nstabs * nlogs * nlogs, dtype=np.double)
    for i in range(len(ops_tensor)):
        process[i * nstabs * nlogs : (i + 1) * nstabs * nlogs] = get_PTMelem_ij(
            kraus_dict, ops_tensor[i], ops_tensor, qcode.N
        )
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
    ops = qc.GetOperatorsForLSTIndex(qcode, range(nstabs * nstabs * nlogs))
    for i in range(len(diag_process)):
        op_tensor = get_Pauli_tensor(ops[i])
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
    ops = qc.GetOperatorsForLSTIndex(qcode, range(nstabs * nstabs * nlogs))
    chi = get_Chielem_ii(kraus_dict, ops, qcode.N)
    print("Sum of chi = {},infid = {}".format(np.sum(chi), 1 - chi[0]))
    return chi


def get_process_chi(qcode, method = "random", *params):
    nstabs = 2 ** (qcode.N - qcode.K)
    nlogs = 4 ** qcode.K
    if method == "random":
        p_error,rotation_angle,w_thresh = params[:3]
        kraus_dict = get_kraus_random(p_error, rotation_angle, qcode, w_thresh)
    elif method == "ising":
        J, mu = params[:2]
        kraus_dict = get_kraus_ising(J, mu, qcode)
    chi = get_chi_diagLST(qcode, kraus_dict)
    process = get_process_correlated(qcode, kraus_dict)
    return (process, chi)
