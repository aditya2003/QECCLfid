def random_input_PTelem_ij(dkraus, nkraus, jlistlength, n_qubits=5):
    pauli_list = [
        np.eye(2),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]]),
    ]
    Pi = get_Pauli_tensor([random.choice(pauli_list)] * n_qubits)
    Pjlist = []
    for i in range(jlistlength):
        Pjlist.append(get_Pauli_tensor([random.choice(pauli_list)] * n_qubits))
    krausdict = {}
    for i in range(nkraus):
        k = dkraus
        support = tuple(sorted((random.sample(range(n_qubits), k))))
        #         print(support)
        r1, r2, r3 = (
            np.random.rand(2 ** k, 2 ** k),
            np.random.rand(2 ** k, 2 ** k),
            np.random.rand(2 ** k, 2 ** k),
        )
        u1, u2, u3 = [
            (1 / np.sqrt(3)) * sc.linalg.expm(-1j * (r1 + Dagger(r1)) / 2),
            (1 / np.sqrt(3)) * sc.linalg.expm(-1j * (r2 + Dagger(r2)) / 2),
            (1 / np.sqrt(3)) * sc.linalg.expm(-1j * (r3 + Dagger(r3)) / 2),
        ]
        #         print(np.linalg.norm(Dot(3*u1,Dagger(u1))-np.eye(2**k)))
        krauslist = [u1, u2, u3]
        krausdict[i] = (support, krauslist)
    return (krausdict, Pi, Pjlist)


def test_get_PTMelem_ij(krausdict, Pi, Pjlist, n_qubits):
    state = Pi.reshape(2 ** n_qubits, 2 ** n_qubits)
    for key, (support, krausList) in krausdict.items():
        for kraus in krausList:
            extended_kraus = extend_gate(
                np.array(support), np.array(kraus), np.array(list(range(n_qubits)))
            )
            state = dot(extended_kraus, state, Dagger(extended_kraus))
        # take dot product with Pj and trace
    trace_vals = np.zeros(len(Pjlist), dtype=np.complex128)
    for i in range(len(Pjlist)):
        Pj = Pjlist[i].reshape(2 ** n_qubits, 2 ** n_qubits)
        trace_vals[i] = np.real(np.trace(dot(Pj, state))) / 2 ** n_qubits
    trace_vals2 = get_PTMelem_ij(krausdict, Pi, Pjlist, n_qubits)
    return np.allclose(trace_vals, trace_vals2)


def random_test_PTelem_ij(dkraus, nkraus, jlistlength, n_qubits=5):
    (krausdict, Pi, Pjlist) = random_input_PTelem_ij(
        dkraus, nkraus, jlistlength, n_qubits
    )
    return test_get_PTMelem_ij(krausdict, Pi, Pjlist, n_qubits)
