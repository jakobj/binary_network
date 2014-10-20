import unittest
import numpy as np
import numpy.testing as nptest

import helper as bhlp

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
        p = np.array([0.4, 0.1, 0.3, 0.2])
        q = np.array([0.7, 0.1, 0.3, 0.2])
        self.assertRaises(ValueError, bhlp.get_DKL, p, q)
        p = np.array([0.4, 0.1, 0.3, 0.2])
        q = np.array([-0.6, 0.8, 0.4, 0.4])
        self.assertRaises(ValueError, bhlp.get_DKL, p, q)
