# global imports
import unittest
import numpy as np
import numpy.testing as nptest

# local imports
from .. import helper as bhlp
from .. import network as bnet

np.random.seed(123456)


class HelperRegressionTestCase(unittest.TestCase):
    
    def test_missing_states(self):
        M = 2
        N = 2
        steps = 20
        a_s = np.ones(M*N*steps/2).reshape(steps/2, M*N)
        a_s = np.vstack([a_s, np.zeros(M*N*steps/2).reshape(steps/2, M*N)])
        a_s[0:int(0.2*steps), 0:2] = 0.
        # p(0,0) = 0.5 + 0.2 = 0.7
        expected_joints = np.array([[0.7, 0., 0., 0.3],
                                    [0.5, 0., 0., 0.5]])
        joints = bhlp.get_joints(a_s, 0., M)
        for i in range(M):
            nptest.assert_array_almost_equal(expected_joints[i], joints[i])
        expected_marginals = np.array([[0.3, 0.3],
                                       [0.5, 0.5]])
        marginals = bhlp.get_marginals_multi_bm(a_s, 0., M)
        for i in range(M):
            nptest.assert_array_almost_equal(expected_marginals[i], marginals[i])

    def test_normalized_positive_density_DKL(self):
        p = np.array([0.4, 0.1, 0.3, 0.2])
        q = np.array([0.5, 0.1, 0.2, 0.2])
        expected_dkl = np.sum([p[i] * np.log(p[i] / q[i]) for i in xrange(4)])
        dkl = bhlp.get_DKL(p, q)
        self.assertEqual(expected_dkl, dkl)
        q = np.array([0.7, 0.1, 0.3, 0.2])
        self.assertRaises(AssertionError, bhlp.get_DKL, p, q)
        q = np.array([-0.6, 0.8, 0.4, 0.4])
        self.assertRaises(AssertionError, bhlp.get_DKL, p, q)
        q = np.array([0., 0.5, 0.1, 0.4])
        self.assertRaises(AssertionError, bhlp.get_DKL, p, q)
        M = 2
        p = np.array([[0.4, 0.1, 0.3, 0.2],
                      [0.05, 0.5, 0.4, 0.05]])
        q = np.array([[0.5, 0.2, 0.2, 0.1],
                      [0.1, 0.5, 0.3, 0.1]])
        expected_dkl = [np.sum([p[j, i] * np.log(p[j, i] / q[j, i]) for i in xrange(4)]) for j in xrange(2)]
        dkl = bhlp.get_DKL_multi_bm(p, q, M)
        nptest.assert_array_almost_equal(expected_dkl, dkl)
        bhlp.get_DKL_multi_bm(p, q, M=2)
        q = np.array([[0.5, 0.2, 0.2, 0.1],
                      [0.8, 0.5, 0.3, 0.1]])
        self.assertRaises(AssertionError, bhlp.get_DKL_multi_bm, p, q, M)
        q = np.array([[0.5, 0.2, 0.2, 0.1],
                      [-0.1, 0.6, 0.4, 0.1]])
        self.assertRaises(AssertionError, bhlp.get_DKL_multi_bm, p, q, M)
        q = np.array([[0., 0.2, 0.2, 0.6],
                      [0.1, 0.5, 0.3, 0.1]])
        self.assertRaises(AssertionError, bhlp.get_DKL_multi_bm, p, q, M)

    def test_initial_state(self):
        N = 5
        W = bhlp.create_BM_weight_matrix(N, np.random.uniform, low=-1., high=1.)
        b = bhlp.create_BM_biases(N, np.random.uniform, low=-1., high=1.)
        beta = 0.8
        sinit = bhlp.random_initial_condition(N)
        rNrec = [1, 3]
        Tmax = 5e4
        tau = 10.
        sinit_sim, a_times, a_s = bnet.simulate_eve_sparse(
            W, b, tau, sinit, Tmax, rNrec, [N], [bhlp.Fsigma], beta=beta)
        nptest.assert_array_equal(np.arange(rNrec[0], rNrec[1]), np.unique(a_s[:, 0]))
        nptest.assert_array_equal(sinit[rNrec[0]:rNrec[1]], sinit_sim)
        self.assertEqual(len(sinit), N)
        self.assertEqual(len(sinit_sim), rNrec[1] - rNrec[0])
