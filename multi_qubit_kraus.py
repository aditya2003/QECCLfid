import numpy as np
from scipy.stats import unitary_group
from collections import deque
import random
import scipy as sc


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


def dot(*mats):
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


def get_Pauli_tensor(listOfPaulis):
    if listOfPaulis:
        Pj = listOfPaulis[0]
        n_qubits = len(listOfPaulis)
        for i in range(1, n_qubits):
            Pj = np.tensordot(Pj, listOfPaulis[i], axes=0)
        indices_Pj = [i for i in range(0, len(Pj.shape), 2)] + [
            i for i in range(1, len(Pj.shape), 2)
        ]
    else:
        raise ValueError(f"Invalid list of Paulis {listOfPaulis} ")
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


def get_PTMelem_ij(krausdict, Pi, Pjlist, n_qubits):
    r"""
    Assumes Paulis Pi,Pj to be a tensor on n_qubits
    Calculates Tr(Pj Eps(Pi))
    Kraus dict has the format ("support": list of kraus ops on the support)
    Assumes qubits
    Assumes kraus ops to be square with dim 2**(n_qubits in support)
    """
    for key, (support, krausList) in krausdict.items():
        indices = support + tuple(map(lambda x: x + n_qubits, support))
        for kraus in krausList:
            kraus_reshape_dims = [2] * (2 * int(np.log2(kraus.shape[0])))
            indices_Pi = indices[len(indices) // 2 :]
            indices_kraus = range(len(kraus_reshape_dims) // 2)
            Pi = np.tensordot(
                Pi,
                Dagger(kraus).reshape(kraus_reshape_dims),
                (indices_Pi, indices_kraus),
            )
            Pi = fix_index_after_tensor(Pi, indices_Pi)
            indices_Pi = indices[: len(indices) // 2]
            indices_kraus = range(len(kraus_reshape_dims))[
                len(kraus_reshape_dims) // 2 :
            ]
            Pi = np.tensordot(
                Pi, kraus.reshape(kraus_reshape_dims), (indices_Pi, indices_kraus)
            )
            Pi = fix_index_after_tensor(Pi, indices_Pi)
        # take dot product with Pj and trace
        trace_vals = np.zeros(len(Pjlist), dtype=np.complex128)
        for i in range(len(Pjlist)):
            Pj = Pjlist[i]
            indices_Pi = list(range(len(Pi.shape) // 2))
            indices_Pj = list(range(len(Pi.shape) // 2, len(Pi.shape)))
            Pi_times_Pj = np.tensordot(Pi, Pj, (indices_Pi, indices_Pj))
            # Take trace
            einsum_inds = list(range(len(Pi_times_Pj.shape) // 2)) + list(
                range(len(Pi_times_Pj.shape) // 2)
            )
            trace_vals[i] = np.real(np.einsum(Pi_times_Pj, einsum_inds)) / 2 ** n_qubits
    return trace_vals
