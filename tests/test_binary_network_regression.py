import unittest
import numpy as np
import numpy.testing as nptest

import helper as bhlp
import network as bnet

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
        marginals = bhlp.get_marginals(a_s, 0., M)
        for i in range(M):
            nptest.assert_array_almost_equal(expected_marginals[i], marginals[i])

    def test_normalized_density_DKL(self):
        p = np.array([0.4, 0.1, 0.3, 0.2])
        q = np.array([0.5, 0.1, 0.2, 0.2])
        bhlp.get_DKL(p, q)
        q = np.array([0.7, 0.1, 0.3, 0.2])
        self.assertRaises(ValueError, bhlp.get_DKL, p, q)
        q = np.array([-0.6, 0.8, 0.4, 0.4])
        self.assertTrue(bhlp.get_DKL(p, q) is np.nan)
        q = np.array([0., 0.5, 0.1, 0.4])
        self.assertTrue(bhlp.get_DKL(p, q) is np.nan)
        M = 2
        p = np.array([[0.4, 0.1, 0.3, 0.2],
                      [0.05, 0.5, 0.4, 0.05]])
        q = np.array([[0.5, 0.2, 0.2, 0.1],
                      [0.1, 0.5, 0.3, 0.1]])
        bhlp.get_DKL(p, q, M=2)
        q = np.array([[0.5, 0.2, 0.2, 0.1],
                      [0.8, 0.5, 0.3, 0.1]])
        self.assertRaises(ValueError, bhlp.get_DKL, p, q, M)
        q = np.array([[0.5, 0.2, 0.2, 0.1],
                      [-0.1, 0.6, 0.4, 0.1]])
        self.assertFalse(bhlp.get_DKL(p, q, M)[0] is np.nan)
        self.assertTrue(bhlp.get_DKL(p, q, M)[1] is np.nan)
        q = np.array([[0., 0.2, 0.2, 0.6],
                      [0.1, 0.5, 0.3, 0.1]])
        self.assertTrue(bhlp.get_DKL(p, q, M)[0] is np.nan)
        self.assertFalse(bhlp.get_DKL(p, q, M)[1] is np.nan)

    def test_initial_state(self):
        N = 4
        W = bhlp.create_BM_weight_matrix(N)
        b = bhlp.create_BM_biases(N)
        beta = 0.8
        sinit = np.random.randint(0, 2, N)
        Nrec = 2
        expected_s0 = sinit[:Nrec]
        Tmax = 5e4
        tau = 10.
        s0, a_times, a_s = bnet.simulate_eve_sparse(
            W, b, tau, sinit, Tmax, Nrec, [N], [bhlp.Fsigma], beta=beta)
        joints = bhlp.get_joints_sparse(s0, a_s, 0)
        self.assertEqual(len(sinit), N)
        self.assertEqual(len(s0), Nrec)
        self.assertEqual(len(joints), 2**Nrec)
        nptest.assert_array_almost_equal(expected_s0, s0)
