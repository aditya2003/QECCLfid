import numpy as np
from define.QECCLfid.ptm import get_Pauli_tensor, fix_index_after_tensor

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
					# print("einsum_inds_old = {}".format(einsum_inds))
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




def get_PTMelem_ij(krausdict, Pi, Pjlist, n_qubits,phasei=None,phasej=None):
	r"""
	Assumes Paulis Pi,Pj to be a tensor on n_qubits
	Calculates Tr(Pj Eps(Pi)) for each Pj in Pjlist
	Kraus dict has the format ("support": list of kraus ops on the support)
	Assumes qubits
	Assumes kraus ops to be square with dim 2**(number of qubits in support)
	"""
	#     Pres stores the addition of all kraus applications
	#     Pi_term stores result of individual kraus applications to Pi
	if phasei is None:
		phasei = 1
	if phasej is None:
		phasej = np.ones(len(Pjlist))
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
		raw_trace = np.einsum(Pres_times_Pj, einsum_inds)*phasei*phasej[i]
		# if np.abs(np.imag(raw_trace)) > 1E-15:
		# 	print("raw_trace {}: {}".format(i, raw_trace))
		trace_vals[i] = np.real(raw_trace) / 2 ** n_qubits
	return trace_vals


def get_process_correlated_list(qcode, kraus_dict):
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
	process = np.array([get_PTMelem_list(kraus_dict, ops_tensor[i], ops_tensor, qcode.N, phases[i], phases) for i in range(nstabs * nlogs)], dtype = np.double).flatten()
	# print("Test for {}\n{}".format(i, test_get_PTMelem_ij(kraus_dict, ops_tensor[i], ops_tensor, qcode.N)))
	return process



def get_Chielem_list(krausdict, Pilist, n_qubits):
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
	einsum_inds = [i for i in range(n_qubits)] + [i for i in range(n_qubits)]
	chi = np.zeros(len(Pilist), dtype=np.double)
	for key, (support, krausList) in krausdict.items():
		indices = support + tuple(map(lambda x: x + n_qubits, support))
		indices_Pi = indices[len(indices) // 2 :]
		Pilist_tensorflow = tf.convert_to_tensor(np.array([get_Pauli_tensor(Pi) for Pi in Pilist], dtype = np.complex128), np.complex128)
		for kraus in krausList:
			if len(indices) > 0:
				kraus_reshape_dims = [2] * (2 * int(np.log2(kraus.shape[0])))
				indices_kraus = range(len(kraus_reshape_dims) // 2)
				kraus_tensorflow = tf.convert_to_tensor(kraus.reshape(kraus_reshape_dims))
				Pi_terms = tf.vectorized_map(lambda x: tf.tensordot(x, kraus_tensorflow, (indices_Pi, indices_kraus)), Pilist_tensorflow)
				# Building the chi matrix
				chi += np.array([
					np.power(np.abs(np.einsum(Pi_terms[i], einsum_inds)), 2)
					/ 4 ** n_qubits
				for i in range(Pi_terms.shape[0])], dtype = np.double)
			else:
				# Take trace and absolute value
				chi[0] += np.abs(kraus) ** 2
	return chi


def get_Chielem_map(krausdict, Pilist, n_qubits):
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
	einsum_inds = [i for i in range(n_qubits)] + [i for i in range(n_qubits)]
	print("einsums = {}".format(einsum_inds))
	for key, (support, krausList) in krausdict.items():
		indices = support + tuple(map(lambda x: x + n_qubits, support))
		indices_Pi = indices[len(indices) // 2 :]
		for kraus in krausList:
			if len(indices) > 0:
				kraus_reshape_dims = [2] * (2 * int(np.log2(kraus.shape[0])))
				indices_kraus = range(len(kraus_reshape_dims) // 2)
				# Pi_terms = map(lambda Pi : np.tensordot(
				# 	get_Pauli_tensor(Pi),
				# 	kraus.reshape(kraus_reshape_dims),
				# 	(indices_Pi, indices_kraus),
				# ),Pilist)
				# Pi_terms = map(lambda Pi :fix_index_after_tensor(np.tensordot(get_Pauli_tensor(Pi), kraus.reshape(kraus_reshape_dims), (indices_Pi, indices_kraus)), indices_Pi), Pilist)
				# Pi_terms = map(lambda Pi_term: fix_index_after_tensor(Pi_term, indices_Pi), Pi_terms)
				# print("First element of Pi_terms = {}".format(next(Pi_terms).shape))
				# Take trace and absolute value
				chi += np.fromiter(
					map(
						lambda Pi: (
							np.power(
								np.abs(
									np.einsum(
										fix_index_after_tensor(
											np.tensordot(
												get_Pauli_tensor(Pi),
												kraus.reshape(kraus_reshape_dims),
												(indices_Pi, indices_kraus),
											),
											indices_Pi,
										),
										einsum_inds,
									)
								),
								2,
							)
							/ 4 ** n_qubits
						),
						Pilist,
					),
					dtype=np.double,
				)
			else:
				# Take trace and absolute value
				chi[0] += np.abs(kraus) ** 2
	return chi