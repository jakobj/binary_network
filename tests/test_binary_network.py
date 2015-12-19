import unittest
import numpy as np
import numpy.testing as nptest

from .. import helper as hlp
from .. import network as bnet

np.random.seed(13456)


class NetworkTestCase(unittest.TestCase):

    def test_unconnected_mean_variance(self):
        N = 100
        W = np.zeros((N, N))
        b = np.ones(N) * 0.2
        sinit = np.random.randint(0, 2, N)
        beta = 0.5
        tau = 10.
        Nrec = 20
        steps = 5e4
        expected_mean = 1. / (1. + np.exp(-b[0]))
        expected_variance = hlp.get_sigma2(expected_mean)
        for i, sim in enumerate([bnet.simulate, bnet.simulate_eve]):
            if i == 0:
                a_s = sim(W, b, sinit, steps, Nrec, [N], [hlp.Fsigma],
                          beta=beta)[1]
            else:
                a_s = sim(W, b, tau, sinit, steps * tau / N,
                          [0, Nrec], [N], [hlp.Fsigma], beta=beta)[1]
            mean = np.mean(a_s)
            variance = np.var(a_s)
            self.assertAlmostEqual(expected_mean, mean, places=1)
            self.assertAlmostEqual(expected_variance, variance, places=1)

    def test_multiple_activation_functions(self):
        N = 100
        W = np.zeros((N, N))
        N1 = 15
        b = np.ones(N) * 0.2
        b[N1:] = 0.9
        sinit = np.random.randint(0, 2, N)
        tau = 10.
        Nrec = 20
        steps = 8e4
        beta = 0.7

        def F2(x, beta):
            return 0 if 1. / (1 + np.exp(-beta * x + 0.7)) < np.random.rand() else 1

        for i, sim in enumerate([bnet.simulate, bnet.simulate_eve, bnet.simulate_eve_sparse]):
            if i == 0:
                a_s = sim(W, b, sinit, steps, Nrec, [N1, N],
                          [hlp.Fsigma, F2], beta)[1]
            elif i == 1:
                a_s = sim(W, b, tau, sinit, steps * tau / N,
                          [0, Nrec], [N1, N], [hlp.Fsigma, F2],
                          beta=beta)[1]
            elif i == 2:
                a_times, a_s = sim(W, b, tau, sinit, steps * tau / N,
                                   [0, Nrec], [N1, N], [hlp.Fsigma, F2],
                                   beta=beta)
                a_s = hlp.get_all_states_from_sparse(a_s, Nrec, 0)
            a_means = np.mean(a_s, axis=0)
            expected_means = np.ones(Nrec) * 1. / (1. + np.exp(-b[0]))
            nptest.assert_array_almost_equal(
                expected_means, a_means, decimal=1)

    def test_joint_distribution(self):
        N = 2
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        beta = 0.5
        sinit = np.random.randint(0, 2, N)
        steps = 5e4
        tau = 10.
        Nrec = 2
        for i, sim in enumerate([bnet.simulate, bnet.simulate_eve]):
            if i == 0:
                a_states, a_s = sim(W, b, sinit, steps,
                                    Nrec, [N], [hlp.Fsigma], beta=beta)
            else:
                a_states, a_s = sim(W, b, tau, sinit, steps *
                                    tau / N, [0, Nrec], [N], [hlp.Fsigma], beta=beta)
            joints = hlp.get_joints(a_s, 0)
            expected_joints = hlp.get_theo_joints(W, b, beta)
            nptest.assert_array_almost_equal(
                expected_joints, joints, decimal=1)

    def test_sparse_simulation(self):
        N = 2
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        beta = 0.8
        sinit = np.random.randint(0, 2, N)
        Tmax = 3e5
        tau = 10.
        rNrec = [0, 2]
        a_times, a_s = bnet.simulate_eve_sparse(
            W, b, tau, sinit, Tmax, rNrec, [N], [hlp.Fsigma], beta=beta)
        self.assertGreater(np.min(a_times), 0.)
        self.assertEqual(len(a_times), len(a_s))
        joints = hlp.get_joints_sparse(a_s, rNrec[1] - rNrec[0], 0)
        expected_joints = hlp.get_theo_joints(W, b, beta)
        nptest.assert_array_almost_equal(expected_joints, joints, decimal=2)

    def test_marginal_distribution(self):
        N = 2
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        beta = 0.7
        sinit = np.random.randint(0, 2, N)
        steps = 5e4
        tau = 10.
        Nrec = 2
        for i, sim in enumerate([bnet.simulate, bnet.simulate_eve]):
            if i == 0:
                a_s = sim(W, b, sinit, steps, Nrec,
                          [N], [hlp.Fsigma], beta=beta)[1]
            else:
                a_s = sim(W, b, tau, sinit, steps * tau / N, [0, Nrec],
                          [N], [hlp.Fsigma], beta=beta)[1]
            marginals = hlp.get_marginals(a_s, 0)
            rvs, expected_marginals = hlp.get_theo_marginals(W, b, beta)
            nptest.assert_array_almost_equal(
                expected_marginals, marginals, decimal=2)

    def test_bin_binary_data(self):
        N = 2
        tbin = 0.04
        time = 2.
        times = np.array([0., 0.1, 0.35, 0.8, 0.95, 1.68])
        a_s = np.array([[0, 0], [1, 0], [1, 1], [1, 0], [0, 0], [1, 0]])
        expected_times = np.arange(0., time + tbin, tbin)
        expected_bin = np.empty((N, len(expected_times)))
        for i, t in enumerate(expected_times):
            idl = np.where(times <= t)[0]
            expected_bin[0][i] = a_s[idl[-1], 0]
            expected_bin[1][i] = a_s[idl[-1], 1]
        times_bin, st = hlp.bin_binary_data(times, a_s, tbin, 0., time)
        nptest.assert_array_equal(expected_times, times_bin)
        nptest.assert_array_equal(expected_bin, st)

    def test_auto_corr(self):
        N = 58
        sinit = np.zeros(N)
        tau = 10.
        Nrec = N
        time = 3.5e3
        mu_target = 0.4
        tbin = .6
        tmax = 600.
        expected_var = mu_target * (1. - mu_target)
        expected_timelag = np.hstack(
            [-1. * np.arange(tbin, tmax + tbin, tbin)[::-1], 0, np.arange(tbin, tmax + tbin, tbin)])
        expected_autof = expected_var * \
            np.exp(-1. * abs(expected_timelag) / tau)

        # Network case (correlated sources)
        w = 0.2
        g = 8.
        gamma = 0.
        epsilon = 0.2
        W_brn = hlp.create_BRN_weight_matrix(N, w, g, epsilon, gamma)
        b_brn = -1. * \
            hlp.get_mu_input(epsilon, N, gamma, g, w, mu_target) * \
            np.ones(N) - 1. * w / 2
        a_times_brn, a_s_brn = bnet.simulate_eve(
            W_brn, b_brn, tau, sinit.copy(), time, [0, Nrec], [N], [hlp.Ftheta])
        self.assertAlmostEqual(
            mu_target, np.mean(a_s_brn), delta=0.1 * np.mean(a_s_brn))
        times_bin_brn, st_brn = hlp.bin_binary_data(
            a_times_brn, a_s_brn, tbin, 0., time)
        timelag_brn, autof_brn = hlp.autocorrf(
            times_bin_brn, st_brn[:30], tmax)
        nptest.assert_array_almost_equal(expected_timelag, timelag_brn)
        self.assertLess(np.sum(autof_brn - expected_autof),
                        0.5 * np.sum(autof_brn))

        # Poisson (independent)
        W = np.zeros((N, N))
        b = np.ones(N) * hlp.sigmainv(mu_target)
        a_times, a_s = bnet.simulate_eve(
            W, b, tau, sinit.copy(), time, [0, Nrec], [N], [hlp.Fsigma])
        self.assertAlmostEqual(
            mu_target, np.mean(a_s), delta=0.1 * np.mean(a_s))
        times_bin, st = hlp.bin_binary_data(a_times, a_s, tbin, 0., time)
        timelag, autof = hlp.autocorrf(times_bin, st[:30], tmax)
        nptest.assert_array_almost_equal(expected_timelag, timelag)
        nptest.assert_array_almost_equal(expected_autof, abs(autof), decimal=2)
        self.assertLess(np.sum(autof - expected_autof),
                        0.5 * np.sum(autof))

    def test_cross_corr(self):
        N = 60
        sinit = np.zeros(N)
        tau = 10.
        Nrec = N
        time = 3e3
        mu_target = 0.4
        tbin = .8
        tmax = 400.
        expected_var = mu_target * (1. - mu_target)
        expected_timelag = np.hstack(
            [-1. * np.arange(tbin, tmax + tbin, tbin)[::-1], 0, np.arange(tbin, tmax + tbin, tbin)])
        expected_autof = expected_var * \
            np.exp(-1. * abs(expected_timelag) / tau)
        expected_cross_brn = -0.0001
        expected_cross = 0.
        expected_crossf = np.zeros(len(expected_timelag))

        # Network case (correlated sources)
        w = 0.2
        g = 8.
        gamma = 0.1
        epsilon = 0.3
        W_brn = hlp.create_BRN_weight_matrix(N, w, g, epsilon, gamma)
        b_brn = -1. * \
            hlp.get_mu_input(epsilon, N, gamma, g, w, mu_target) * \
            np.ones(N) - 1. * w / 2
        a_times_brn, a_s_brn = bnet.simulate_eve(
            W_brn, b_brn, tau, sinit.copy(), time, [0, Nrec], [N], [hlp.Ftheta])
        self.assertTrue(abs(np.mean(a_s_brn) - mu_target) < 0.1 * mu_target)
        times_bin_brn, st_brn = hlp.bin_binary_data(
            a_times_brn, a_s_brn, tbin, 0., time)
        timelag_brn, autof_brn, crossf_brn = hlp.crosscorrf(
            times_bin_brn, st_brn[:30], tmax)
        nptest.assert_array_almost_equal(expected_timelag, timelag_brn)
        self.assertLess(abs(np.sum(autof_brn - expected_autof)),
                        0.5 * np.sum(abs(autof_brn)))
        self.assertLess(crossf_brn[abs(timelag_brn) < 1e-10][0],
                        expected_cross_brn)

        # Poisson case (independent sources)
        W = np.zeros((N, N))
        b = np.ones(N) * hlp.sigmainv(mu_target)
        a_times, a_s = bnet.simulate_eve(
            W, b, tau, sinit.copy(), time, [0, Nrec], [N], [hlp.Fsigma])
        self.assertTrue(abs(np.mean(a_s) - mu_target) < 0.1 * mu_target)
        times_bin, st = hlp.bin_binary_data(a_times, a_s, tbin, 0., time)
        timelag, autof, crossf = hlp.crosscorrf(times_bin, st[:30], tmax)
        nptest.assert_array_almost_equal(expected_timelag, timelag)
        nptest.assert_array_almost_equal(expected_autof, abs(autof), decimal=2)
        nptest.assert_array_almost_equal(expected_crossf, crossf, decimal=2)
        self.assertAlmostEqual(
            expected_cross, abs(crossf[abs(timelag) < 1e-10][0]), places=2)

    def test_input(self):
        N = 20
        Nnoise = 220
        sinit = np.zeros(N + Nnoise)
        tau = 10.
        Nrec = N
        time = 2e4
        mu_target = 0.46
        Nrec_ui = N

        beta = .8
        w = 0.2
        g = 8.
        gamma = 0.2
        epsilon = 0.3
        expected_mu_input = hlp.get_mu_input(
            epsilon, Nnoise, gamma, g, w, mu_target)
        expected_std_input = hlp.get_sigma_input(
            epsilon, Nnoise, gamma, g, w, mu_target)

        # Poisson case (independent sources)
        W = np.zeros((N + Nnoise, N + Nnoise))
        W[:N, N:] = hlp.create_noise_weight_matrix(
            N, Nnoise, gamma, g, w, epsilon)
        b = np.zeros(N + Nnoise)
        b[:N] = -w / 2.
        b[N:] = hlp.sigmainv(mu_target, beta)
        for i, sim in enumerate([bnet.simulate, bnet.simulate_eve]):
            if i == 0:
                a_times, a_s, a_times_ui, a_ui = sim(
                    W, b, sinit.copy(), time, Nrec, [N, N + Nnoise], [hlp.Ftheta, hlp.Fsigma], Nrec_ui, beta)
                steps_warmup = 0.1 * time / (N + Nnoise) * N
            elif i == 1:
                a_times, a_s, a_times_ui, a_ui = sim(
                    W, b, tau, sinit.copy(), time, [0, Nrec], [N, N + Nnoise], [hlp.Ftheta, hlp.Fsigma], Nrec_ui, beta)
                steps_warmup = 0.1 * Nrec_ui * time / tau
            a_ui = a_ui[steps_warmup:]
            self.assertLess(abs(np.mean(a_ui) + w / 2. - expected_mu_input),
                            0.05 * abs(expected_mu_input))
            self.assertLess(abs(np.mean(np.std(a_ui, axis=0)) - expected_std_input),
                            0.05 * expected_std_input)


if __name__ == '__main__':
    unittest.main()
