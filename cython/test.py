import numpy as np
from krauss_ptm import KrausToPTM
from krauss_theta import KrausToTheta
from timeit import default_timer as timer

if __name__ == '__main__':
	# Test KrausToTheta function.
	nq = 4
	kraus = np.random.rand(4**nq, 2**nq, 2**nq) + 1j * np.random.rand(4**nq, 2**nq, 2**nq)
	print("Kraus with shape {}\n{}".format(kraus.shape, kraus))
	click = timer()
	theta = KrausToTheta(kraus, nq)
	print("Theta matrix was constructed in %.3f seconds." % (timer() - click))
	print("Theta\n{}".format(theta))
	click = timer()
	theta = KrausToTheta(kraus, nq)
	print("PTM\n{}".format(theta))
	print("PTM was constructed in %.3f seconds." % (timer() - click))