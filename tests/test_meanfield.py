# global imports
import unittest
import numpy as np
import numpy.testing as nptest
import scipy.integrate as scint

# local imports
from .. import helper as bhlp
from .. import network as bnet
from .. import meanfield as bmf
from .. import unitginzburgmeanfield as ugbmf

np.random.seed(123456)


class MeanfieldTestCase(unittest.TestCase):

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

    def test_get_mu_input(self):
        expected_mu_input = self.KE * self.w * \
            self.mu[0] + self.KI * (-self.g * self.w) * self.mu[1]
        mu_input = self.mfi.get_mu_input(self.mu)
        self.assertAlmostEqual(expected_mu_input, mu_input[0])
        self.assertAlmostEqual(expected_mu_input, mu_input[1])

    def test_get_sigma_input(self):
        CEE = 0.003
        CIE = CEI = 0.1
        CII = -0.003
        sigma_input = self.mfi.get_sigma_input(self.mu)
        expected_sigma_input = np.sqrt(
            self.KE * self.w ** 2 * self.mu[0] * (1. - self.mu[0]) + self.KI * (-self.g * self.w) ** 2 * self.mu[1] * (1. - self.mu[1]))
        self.assertAlmostEqual(expected_sigma_input, sigma_input[0])
        self.assertAlmostEqual(expected_sigma_input, sigma_input[1])
        C = np.array([[CEE, CIE],
                      [CEI, CII]])
        sigma_input = self.mfi.get_sigma_input(self.mu, C)
        expected_sigma_input = np.sqrt(
            self.KE * self.w ** 2 *
            self.mu[0] * (1. - self.mu[0]) + self.KI *
            (-self.g * self.w) ** 2 * self.mu[1] * (1. - self.mu[1])
            + (self.KE * self.w) ** 2 * CEE + 2. * self.KE * self.KI * (-self.g * self.w ** 2) * CEI + (self.KI * (-self.g * self.w)) ** 2 * CII)
        self.assertAlmostEqual(expected_sigma_input, sigma_input[0])
        self.assertAlmostEqual(expected_sigma_input, sigma_input[1])

    def test_get_suszeptibility(self):
        mu_input = self.mfi.get_mu_input(self.mu)
        sigma_input = self.mfi.get_sigma_input(self.mu)
        expected_S0 = 1. / \
            (np.sqrt(2. * np.pi) * sigma_input[0]) * \
            np.exp(-(mu_input[0] + self.b[0])
                   ** 2 / (2. * sigma_input[0] ** 2))
        expected_S1 = 1. / \
            (np.sqrt(2. * np.pi) * sigma_input[1]) * \
            np.exp(-(mu_input[1] + self.b[1])
                   ** 2 / (2. * sigma_input[1] ** 2))
        S = self.mfi.get_suszeptibility(mu_input, sigma_input)
        self.assertAlmostEqual(expected_S0, S[0])
        self.assertAlmostEqual(expected_S1, S[1])

    def test_get_w_meanfield(self):
        mu_input = self.mfi.get_mu_input(self.mu)
        sigma_input = self.mfi.get_sigma_input(self.mu)
        S = self.mfi.get_suszeptibility(mu_input, sigma_input)
        expected_w00 = self.KE * self.w * S[0]
        expected_w01 = self.KI * (-self.g * self.w) * S[0]
        expected_w10 = self.KE * self.w * S[1]
        expected_w11 = self.KI * (-self.g * self.w) * S[1]
        W = self.mfi.get_w_meanfield(self.mu)
        self.assertAlmostEqual(expected_w00, W[0, 0])
        self.assertAlmostEqual(expected_w01, W[0, 1])
        self.assertAlmostEqual(expected_w10, W[1, 0])
        self.assertAlmostEqual(expected_w11, W[1, 1])

    def test_c_meanfield(self):
        epsilon = 0.1
        N = 100.
        gamma = 0.
        g = 8.
        w = 0.35
        b = np.array([0., 0.9])
        mfi = bmf.BinaryMeanfield(epsilon, N, gamma, g, w, b)
        mu = mfi.get_mu_meanfield(np.array([0.5, 0.5]))
        wII = mfi.get_w_meanfield(mu)[1, 1]
        AI = bhlp.get_sigma2(mu)[1] / N
        expected_CII = wII / (1. - wII) * AI
        C = mfi.get_c_meanfield(mu)
        self.assertAlmostEqual(expected_CII, C[1, 1])

    def test_comp_network_meanfield(self):
        N = 10
        Nnoise = 500
        T = 1.5e4
        w = 0.1
        g = 8.
        epsilon = 0.3
        gamma = 0.3
        mu_target = 0.15
        tau = 10.
        Nrec = 60

        W = np.zeros((N + Nnoise, N + Nnoise))
        W[:N, N:] = bhlp.create_noise_weight_matrix(
            N, Nnoise, gamma, g, w, epsilon)
        W[N:, N:] = bhlp.create_BRN_weight_matrix(
            Nnoise, w, g, epsilon, gamma)
        b = np.zeros(N + Nnoise)
        b[:N] = -w / 2.
        b[N:] = -1. * \
            bhlp.get_mu_input(epsilon, Nnoise, gamma, g, w, mu_target) - w / 2.
        sinit = np.array(np.random.randint(0, 2, N + Nnoise), dtype=np.int)

        times, a_s, a_times_ui, a_ui = bnet.simulate_eve(
            W, b, tau, sinit, T, [0, N + Nrec], [N + Nnoise], [bhlp.Ftheta], Nrec_ui=N)
        a_ui = a_ui[200:]
        a_s = a_s[200:]

        # empirical
        mu_noise_activity = np.mean(a_s[:, N:])
        std_noise_activity = np.mean(np.std(a_s[:, N:], axis=0))
        mu_noise = np.mean(a_ui[:, :N])
        std_noise = np.mean(np.std(a_ui[:, :N], axis=0))

        # meanfield
        mfcl = bmf.BinaryMeanfield(
            epsilon, Nnoise, gamma, g, w, np.array([b[N + 1], b[N + 1]]))
        # naive
        mu_naive = mfcl.get_m(np.array([0.2, 0.2]).T)
        std_naive = bhlp.get_sigma(mu_naive)[1]
        mu_naive_input = mfcl.get_mu_input(mu_naive)[1]
        std_naive_input = mfcl.get_sigma_input(mu_naive)[1]
        mu_naive = mu_naive[1]

        # improved (i.e., with correlations)
        mu_iter, c_iter = mfcl.get_m_c_iter(np.array([0.2, 0.2]).T)
        std_iter = bhlp.get_sigma(mu_iter)[1]
        mu_iter_input = mfcl.get_mu_input(mu_iter)[1]
        std_iter_input = mfcl.get_sigma_input(mu_iter, c_iter)[1]
        mu_iter = mu_iter[1]

        self.assertAlmostEqual(
            mu_noise_activity, mu_naive, delta=0.1 * mu_naive)
        self.assertAlmostEqual(
            std_noise_activity, std_naive, delta=0.1 * std_naive)
        self.assertAlmostEqual(mu_noise, mu_naive_input,
                               delta=abs(0.2 * mu_naive_input))
        self.assertAlmostEqual(
            std_noise, std_naive_input, delta=abs(0.2 * std_naive_input))

        self.assertAlmostEqual(
            mu_noise_activity, mu_iter, delta=0.05 * mu_iter)
        self.assertAlmostEqual(
            std_noise_activity, std_iter, delta=0.04 * std_iter)
        self.assertAlmostEqual(
            mu_noise, mu_iter_input, delta=abs(0.04 * mu_iter_input))
        self.assertAlmostEqual(std_noise, std_iter_input,
                               delta=abs(0.04 * std_iter_input))


class GinzburgUnitMeanfieldTestCase(unittest.TestCase):

    def setUp(self):
        self.N = 17
        muJ = -0.4
        sigmaJ = 0.1
        self.mu_target = 0.48
        self.beta = .4
        self.J = bhlp.create_BM_weight_matrix(self.N, np.random.normal, loc=muJ, scale=sigmaJ)
        self.b = bhlp.create_BM_biases_threshold_condition(self.N, muJ, self.mu_target)
        self.mf_net = ugbmf.BinaryMeanfield(self.J, self.b, self.beta)
        # example mean activity and correlation
        self.mu = np.random.uniform(0.2, 0.6, self.N)
        self.C = np.random.normal(0., 0.02, (self.N, self.N))
        for i in xrange(self.N):
            self.C[i, i] = self.mu[i] * (1. - self.mu[i])

    def test_get_mu_input(self):
        mu = np.random.uniform(0.2, 0.6, self.N)
        expected_mu_input = np.dot(self.J, mu)
        mu_input = self.mf_net.get_mu_input(mu)
        nptest.assert_array_almost_equal(expected_mu_input, mu_input)

    def test_get_sigma2_input(self):
        expected_sigma2_input = np.dot(self.J ** 2, self.C.diagonal())
        sigma2_input = self.mf_net.get_sigma2_input(np.diag(self.C.diagonal()))
        nptest.assert_array_almost_equal(expected_sigma2_input, sigma2_input)
        expected_sigma2_input = np.dot(
            self.J, np.dot(self.C, self.J.T)).diagonal()
        sigma2_input = self.mf_net.get_sigma2_input(self.C)
        nptest.assert_array_almost_equal(expected_sigma2_input, sigma2_input)

    def test_get_mu_meanfield(self):
        mu_input = self.mf_net.get_mu_input(self.mu)
        sigma2_input = self.mf_net.get_sigma2_input(self.C)
        expected_m = np.zeros(self.N)
        for i in xrange(self.N):
            def f(x):
                return 1. / (1. + np.exp(-self.beta * x)) \
                    * 1. / np.sqrt(2. * np.pi * sigma2_input[i]) \
                    * np.exp(-(x - mu_input[i] - self.b[i]) ** 2 / (2 * sigma2_input[i]))
            expected_m[i], error = scint.quad(f, -3e2, 3e2)
            self.assertLess(error, 1e-7)
        m = self.mf_net.get_mu_meanfield(self.mu, self.C)
        nptest.assert_array_almost_equal(expected_m, m, decimal=5)

    def test_get_suszeptibility(self):
        mu_input = self.mf_net.get_mu_input(self.mu)
        sigma2_input = self.mf_net.get_sigma2_input(self.C)
        expected_S = np.empty(self.N)
        for i in xrange(self.N):
            def f(x):
                return self.beta / (1. + np.exp(-self.beta * x)) ** 2 * np.exp(-self.beta * x) \
                    * 1. / np.sqrt(2. * np.pi * sigma2_input[i]) \
                    * np.exp(-(x - mu_input[i] - self.b[i]) ** 2 / (2 * sigma2_input[i]))
            expected_S[i], error = scint.quad(f, -2e2, 2e2)
            self.assertLess(error, 1e-7)
        S = self.mf_net.get_suszeptibility(self.mu, self.C)
        nptest.assert_array_almost_equal(expected_S, S, decimal=4)

    def test_get_w_meanfield(self):
        S = self.mf_net.get_suszeptibility(self.mu, self.C)
        expected_W = self.J.copy()
        for i in xrange(self.N):
            expected_W[i, :] = expected_W[i, :] * S[i]
        W = self.mf_net.get_w_meanfield(self.mu, self.C)
        nptest.assert_array_almost_equal(expected_W.flatten(), W.flatten())

    def test_m_corr_iter(self):
        lamb = 0.5
        # TODO rename function to get_theo_rates_covariances
        expected_rates, expected_cov = bhlp.get_theo_covariances(
            self.J, self.b, self.beta)
        rates, cov = self.mf_net.get_m_corr_iter(
            np.ones(self.N) * self.mu_target, lamb)
        nptest.assert_array_almost_equal(expected_rates, rates, decimal=5)
        nptest.assert_array_almost_equal(
            expected_cov.flatten(), cov.flatten(), decimal=4)


if __name__ == '__main__':
    unittest.main()
