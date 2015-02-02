import unittest
import numpy as np

import meanfield as bmf

np.random.seed(123456)


class MeanfieldRegressionTestCase(unittest.TestCase):

    def setUp(self):
        epsilon = 0.1
        N = 100
        gamma = 0.2
        self.g = 8.
        self.w = 0.35
        self.b = np.array([0.7, 0.9])
        self.NE = int(gamma * N)
        self.NI = N - self.NE
        self.KE = int(epsilon * self.NE)
        self.KI = int(epsilon * self.NI)
        self.mu = np.array([0.6, 0.5])
        self.sigma = np.array([0.35, 0.73])
        self.mfi = bmf.BinaryMeanfield(
            epsilon, N, gamma, self.g, self.w, self.b)

    def test_get_m(self):
        self.assertRaises(ValueError, self.mfi.get_m_c_iter, 0.2)

    def test_get_m_c_iter(self):
        self.assertRaises(ValueError, self.mfi.get_m, 0.2)

    def test_get_mu_input(self):
        self.assertRaises(ValueError, self.mfi.get_mu_input, 0.2)

    def test_get_sigma_input(self):
        self.assertRaises(ValueError, self.mfi.get_sigma_input, 0.2)
        self.mfi.get_sigma_input([0.2, 0.2], [[0.2, 0.2], [0.2, 0.2]])
        self.assertRaises(ValueError, self.mfi.get_sigma_input, 0.2, [[0.2, 0.2], [0.2, 0.2]])
        self.assertRaises(ValueError, self.mfi.get_sigma_input, [0.2, 0.2], [0.2, 0.2])

