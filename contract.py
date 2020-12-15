import string
import numpy as np

def TupleToString(tup):
	# Convert a tuple to a string where each number is assigned to its string form.
	return "".join(list(map(lambda i: string.printable[10 + i], tup)))

def YieldScalar(tndict):
	# Check if contracting all the tensors in the dictionary yields a scalar.
	supports = np.array([np.array(list(support), dtype = np.int) for (support, __) in tndict], dtype = np.int).flatten()
	#print("supports = {}".format(supports))
	visited = np.zeros(supports.shape[0], dtype = np.int)
	for i in range(supports.size):
		not_scalar = 1
		for j in range(supports.size):
			if (i != j):
				if (supports[i] == supports[j]):
					not_scalar = 0
					#print("Comparing support[%d] = %d and support[%d] = %d." % (i, supports[i], j, supports[j])) 
	scalar = 1 - not_scalar
	return scalar

def TensorTrace(tndict, opt = "greedy"):
	# Contract a set of tensors provided as a dictionary.
	# The result of the traction must be a number, in other words, there should be no free indices.
	if (YieldScalar(tndict) == 0):
		print("The Tensor contraction does not result in a trace.")
		return None
	# We want to use np.einsum(...) to perform the Tensor contraction.
	# Every element of the input dictionary is a tuple, containing the support of a tensor and its operator form.
	# Numpy's einsum function simply needs the Tensors and the corresponding labels associated to its indices.
	# We will convert the support of each Tensor to a string, that serves as its label.
	scheme = ",".join([TupleToString(support) for (support, __) in tndict])
	print("Contraction scheme: {}".format(scheme))
	ops = [op for (__, op) in tndict]
	ops_args = ", ".join([("ops[%d]" % d) for d in range(len(ops))])
	print("np.einsum_path(\'%s->\', %s, optimize=\'%s\')" % (scheme, ops_args, opt))
	path = eval("np.einsum_path(\'%s->\', %s, optimize=\'%s\')" % (scheme, ops_args, opt))
	print("Contraction process\n{}: {}\n{}".format(path[0][0], path[0][1:], path[1]))
	trace = np.einsum(scheme, *ops, optimize=path[0])
	#print("Trace = {}.".format(trace))
	return trace


if __name__ == '__main__':
	N = 24
	dims = 4
	bond = 4
	# create some numpy tensors
	tensors = [np.random.rand(*[bond]*dims) for __ in range(0, N, 2)]
	# labels should ensure that there is no free index.
	labels = [tuple([(i + j) % N for j in range(4)]) for i in range(0, N, 2)]
	print(labels)
	# Prepare the dictionary.
	tndict = [(labels[i], tensors[i]) for i in range(len(labels))]
	#print([tnop for (lab, tnop) in tndict])
	trace = TensorTrace(tndict)
	print("Trace = {}.".format(trace))