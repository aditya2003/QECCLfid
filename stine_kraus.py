def StineToKraus(U):
    # Compute the Krauss operators for the input quantum channel, which is represented in the Stinespring dialation
    # The Krauss operator T_k is given by: <a|T_k|b> = <a e_k|U|b e_0> , where {|e_i>} is a basis for the environment and |a>, |b> are basis vectors of the system
    # Note that a k-qubit channel needs to be generated from a unitary matrix on 3*k qubits where 2*k qubits represent the environment.
    nq = int(np.log2(U.shape[0]))//3
    environment = np.eye(4**nq)[:,:,np.newaxis]
    system = np.eye(2**nq)[:,:,np.newaxis]
    krauss = np.zeros((4**nq, 2**nq, 2**nq), dtype=np.complex128)
    for ki in range(4**nq):
        ## The Krauss operator T_k is given by: <a|T_k|b> = <a e_k|U|b e_0>.
        for ri in range(2**nq):
            for ci in range(2**nq):
                leftProduct = HermitianConjugate(
                    np.dot(
                        U, np.kron(system[ri, :, :], environment[ki, :, :])
                    )
                )
                krauss[ki, ri, ci] = np.dot(
                    leftProduct, np.kron(system[ci, :, :], environment[0, :, :])
                )[0, 0]
    return krauss