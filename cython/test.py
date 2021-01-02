import numpy as np
from krauss_theta import KrausToTheta

if __name__ == '__main__':
	# Test KrausToTheta function.
	nq = 4
	kraus = np.random.rand(4**nq, 2**nq, 2**nq) + 1j * np.random.rand(4**nq, 2**nq, 2**nq)
	print("Kraus with shape {}\n{}".format(kraus.shape, kraus))
	theta = KrausToTheta(kraus, nq)
	print("Theta\n{}".format(theta))