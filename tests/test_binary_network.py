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
        rNrec = [0, 20]
        steps = int(5e4)
        expected_mean = 1. / (1. + np.exp(-b[0]))
        expected_variance = hlp.get_sigma2(expected_mean)
        for i, sim in enumerate([bnet.simulate, bnet.simulate_eve_sparse]):
            if i == 0:
                a_s = sim(W, b, sinit, steps, rNrec, [N], [hlp.Fsigma],
                          beta=beta)[1]
            else:
                sinit, a_times, a_s = sim(W, b, tau, sinit, steps * tau / N,
                                          rNrec, [N], [hlp.Fsigma], beta=beta)
                a_s = hlp.get_all_states_from_sparse(sinit, a_s)
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
        rNrec = [0, 20]
        steps = int(8e4)
        steps_warmup = 1e4
        beta = 0.7

        def F2(x, beta):
            return 0 if 1. / (1 + np.exp(-beta * x + 0.7)) < np.random.rand() else 1

        for i, sim in enumerate([bnet.simulate, bnet.simulate_eve_sparse]):
            if i == 0:
                a_s = sim(W, b, sinit, steps, rNrec, [N1, N],
                          [hlp.Fsigma, F2], beta)[1]
            elif i == 1:
                sinit, a_times, a_s = sim(W, b, tau, sinit, steps * tau / N,
                                          rNrec, [N1, N], [hlp.Fsigma, F2],
                                          beta=beta)
                a_s = hlp.get_all_states_from_sparse(sinit, a_s)
            a_means = hlp.get_marginals(a_s, steps_warmup)
            Nrec = rNrec[1] - rNrec[0]
            expected_means = np.ones(Nrec) * 1. / (1. + np.exp(-b[0]))
            nptest.assert_array_almost_equal(
                expected_means, a_means, decimal=1)

    def test_joint_distribution(self):
        N = 2
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        beta = 0.5
        sinit = hlp.random_initial_condition(N)
        steps = int(1e5)
        steps_warmup = 2e4
        tau = 10.
        rNrec = [0, 2]
        expected_joints = hlp.get_theo_joints(W, b, beta)

        # step-driven simulation
        a_states, a_s = bnet.simulate(W, b, sinit, steps,
                                      rNrec, [N], [hlp.Fsigma], beta=beta)
        joints = hlp.get_joints(a_s, steps_warmup)
        nptest.assert_array_almost_equal(
            expected_joints, joints, decimal=2)

        # event-driven simulation
        sinit, a_states, a_s = bnet.simulate_eve_sparse(W, b, tau, sinit, steps * tau / N, rNrec, [N], [hlp.Fsigma], beta=beta)
        joints = hlp.get_joints_sparse(sinit, a_s, steps_warmup)
        nptest.assert_array_almost_equal(
            expected_joints, joints, decimal=2)

    def test_sparse_simulation(self):
        N = 2
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        beta = 0.8
        sinit = np.random.randint(0, 2, N)
        Tmax = 3e5
        tau = 10.
        rNrec = [0, 2]
        sinit, a_times, a_s = bnet.simulate_eve_sparse(
            W, b, tau, sinit, Tmax, rNrec, [N], [hlp.Fsigma], beta=beta)
        self.assertGreater(np.min(a_times), 0.)
        self.assertEqual(len(a_times), len(a_s))
        joints = hlp.get_joints_sparse(sinit, a_s, 0)
        expected_joints = hlp.get_theo_joints(W, b, beta)
        nptest.assert_array_almost_equal(expected_joints, joints, decimal=2)

    def test_marginal_distribution(self):
        N = 2
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        beta = 0.7
        sinit = np.random.randint(0, 2, N)
        steps = int(8e4)
        steps_warmup = 1e4
        tau = 10.
        rNrec = [0, 2]
        rvs, expected_marginals = hlp.get_theo_marginals(W, b, beta)

        a_s = bnet.simulate(W, b, sinit, steps, rNrec,
                            [N], [hlp.Fsigma], beta=beta)[1]
        marginals = hlp.get_marginals(a_s, steps_warmup)
        nptest.assert_array_almost_equal(
            expected_marginals, marginals, decimal=2)

        sinit, a_times, a_s = bnet.simulate_eve_sparse(W, b, tau, sinit, steps * tau / N, rNrec,
                                                       [N], [hlp.Fsigma], beta=beta)
        a_s_full = hlp.get_all_states_from_sparse(sinit, a_s)
        marginals = hlp.get_marginals(a_s_full, steps_warmup)
        nptest.assert_array_almost_equal(
            expected_marginals, marginals, decimal=2)

    # def test_auto_corr(self):
    #     N = 58
    #     sinit = hlp.random_initial_condition(N)
    #     tau = 10.
    #     rNrec = [0, N]
    #     time = 5e3
    #     Twarmup = 1e2
    #     steps_warmup = hlp.get_steps_warmup(rNrec, Twarmup, tau)
    #     mu_target = 0.4
    #     tbin = .7
    #     tmax_lag = 600.
    #     expected_var = mu_target * (1. - mu_target)
    #     expected_timelag = np.hstack([-1. * np.arange(tbin, tmax_lag, tbin)[::-1], 0, np.arange(tbin, tmax_lag, tbin)])
    #     expected_autof = expected_var * \
    #         np.exp(-1. * abs(expected_timelag) / tau)

    #     # Network case (correlated sources)time_slices
    #     w = 0.2
    #     g = 8.
    #     gamma = 0.
    #     epsilon = 0.1
    #     W_brn = hlp.create_BRN_weight_matrix(N, w, g, epsilon, gamma)
    #     b_brn = -1. * \
    #         hlp.get_mu_input(epsilon, N, gamma, g, w, mu_target) * \
    #         np.ones(N) - 1. * w / 2
    #     sinit, a_times_brn, a_s_brn = bnet.simulate_eve_sparse(
    #         W_brn, b_brn, tau, sinit, time, rNrec, [N], [hlp.Ftheta])
    #     a_s_brn_full = hlp.get_all_states_from_sparse(sinit, a_s_brn)
    #     a_s_brn_full = hlp.adjust_recorded_states(a_s_brn_full, steps_warmup)
    #     a_times_brn = hlp.adjust_time_slices(a_times_brn, steps_warmup)
    #     self.assertAlmostEqual(
    #         mu_target, np.mean(a_s_brn_full), delta=0.1 * np.mean(a_s_brn_full))
    #     times_bin_brn, st_brn = hlp.bin_binary_data(
    #         a_times_brn, a_s_brn_full, tbin, Twarmup, time)
    #     timelag_brn, autof_brn = hlp.autocorrf(
    #         st_brn, tbin, tmax_lag)
    #     nptest.assert_array_almost_equal(expected_timelag, timelag_brn)
    #     self.assertLess(np.sum(autof_brn - expected_autof),
    #                     0.5 * np.sum(autof_brn))

    #     # Poisson (independent)
    #     W = np.zeros((N, N))
    #     b = np.ones(N) * hlp.sigmainv(mu_target)
    #     sinit, a_times, a_s = bnet.simulate_eve_sparse(
    #         W, b, tau, sinit.copy(), time, rNrec, [N], [hlp.Fsigma])
    #     a_s_full = hlp.get_all_states_from_sparse(sinit, a_s)
    #     a_s_full = hlp.adjust_recorded_states(a_s_full, steps_warmup)
    #     a_times = hlp.adjust_time_slices(a_times, steps_warmup)
    #     self.assertAlmostEqual(
    #         mu_target, np.mean(a_s_full), delta=0.1 * np.mean(a_s_full))
    #     times_bin, st = hlp.bin_binary_data(a_times, a_s_full, tbin, 0., time)
    #     timelag, autof = hlp.autocorrf(st, tbin, tmax_lag)
    #     nptest.assert_array_almost_equal(expected_timelag, timelag)
    #     nptest.assert_array_almost_equal(expected_autof, abs(autof), decimal=2)
    #     self.assertLess(np.sum(autof - expected_autof),
    #                     0.1 * np.sum(autof))

    # def test_cross_corr(self):
    #     N = 120
    #     sinit = hlp.random_initial_condition(N)
    #     tau = 10.
    #     rNrec = [0, N]
    #     time = 6e3
    #     mu_target = 0.4
    #     Twarmup = 1.e3
    #     tbin = .8
    #     tmax_lag = 550.
    #     expected_var = mu_target * (1. - mu_target)
    #     expected_timelag = np.hstack([-1. * np.arange(tbin, tmax_lag, tbin)[::-1], 0, np.arange(tbin, tmax_lag, tbin)])
    #     expected_autof = expected_var * \
    #         np.exp(-1. * abs(expected_timelag) / tau)
    #     expected_cross_brn = -0.0001
    #     expected_cross = 0.
    #     expected_crossf = np.zeros(len(expected_timelag))
    #     steps_warmup = hlp.get_steps_warmup(rNrec, Twarmup, tau)

    #     # Network case (correlated sources)
    #     w = 0.05
    #     g = 8.
    #     gamma = 0.
    #     epsilon = 0.08
    #     W_brn = hlp.create_BRN_weight_matrix(N, w, g, epsilon, gamma)
    #     b_brn = hlp.create_BRN_biases_threshold_condition(N, w, g, epsilon, gamma, mu_target)
    #     sinit, a_times_brn, a_s_brn = bnet.simulate_eve_sparse(
    #         W_brn, b_brn, tau, sinit.copy(), time, rNrec, [N], [hlp.Ftheta])
    #     a_s_brn_full = hlp.get_all_states_from_sparse(sinit, a_s_brn)
    #     a_s_brn_full = hlp.adjust_recorded_states(a_s_brn_full, steps_warmup)
    #     a_times_brn = hlp.adjust_time_slices(a_times_brn, steps_warmup)
    #     self.assertTrue(abs(np.mean(a_s_brn_full) - mu_target) < 0.1 * mu_target)
    #     times_bin_brn, st_brn = hlp.bin_binary_data(
    #         a_times_brn, a_s_brn_full, tbin, Twarmup, time)
    #     timelag_brn, autof_brn, crossf_brn = hlp.crosscorrf(st_brn, tbin, tmax_lag)

    #     nptest.assert_array_almost_equal(expected_timelag, timelag_brn)
    #     self.assertLess(abs(np.sum(autof_brn - expected_autof)),
    #                     0.7 * np.sum(abs(autof_brn)))
    #     self.assertLess(crossf_brn[abs(timelag_brn) < tbin][0],
    #                     expected_cross_brn)

    #     # Poisson case (independent sources)
    #     W = np.zeros((N, N))
    #     b = np.ones(N) * hlp.sigmainv(mu_target)
    #     sinit, a_times, a_s = bnet.simulate_eve_sparse(
    #         W, b, tau, sinit.copy(), time, rNrec, [N], [hlp.Fsigma])
    #     a_s_full = hlp.get_all_states_from_sparse(sinit, a_s)
    #     a_s_full = hlp.adjust_recorded_states(a_s_full, steps_warmup)
    #     a_times = hlp.adjust_time_slices(a_times, steps_warmup)
    #     self.assertTrue(abs(np.mean(a_s_full) - mu_target) < 0.1 * mu_target)
    #     times_bin, st = hlp.bin_binary_data(a_times, a_s_full, tbin, Twarmup, time)
    #     timelag, autof, crossf = hlp.crosscorrf(st, tbin, tmax_lag)

    #     nptest.assert_array_almost_equal(expected_timelag, timelag)
    #     nptest.assert_array_almost_equal(expected_autof, autof, decimal=2)
    #     nptest.assert_array_almost_equal(expected_crossf, crossf, decimal=2)
    #     self.assertAlmostEqual(
    #         expected_cross, abs(crossf[abs(timelag) < tbin][0]), places=2)

    def test_input(self):
        N = 20
        Nnoise = 220
        sinit = np.zeros(N + Nnoise)
        tau = 10.
        rNrec = [0, N]
        steps = int(2e4)
        steps_warmup = 1e3
        mu_target = 0.46
        rNrec_u = [0, N]

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
        a_steps, a_s, a_steps_ui, a_ui = bnet.simulate(
            W, b, sinit.copy(), steps, rNrec, [N, N + Nnoise], [hlp.Ftheta, hlp.Fsigma], beta, rNrec_u=rNrec_u)
        a_ui = a_ui[steps_warmup:]
        self.assertLess(abs(np.mean(a_ui) + w / 2. - expected_mu_input),
                        0.05 * abs(expected_mu_input))
        self.assertLess(abs(np.mean(np.std(a_ui, axis=0)) - expected_std_input),
                        0.05 * expected_std_input)

        time = steps / (N + Nnoise) * tau
        sinit, a_times, a_s, a_times_ui, a_ui = bnet.simulate_eve_sparse(
            W, b, tau, sinit.copy(), time, rNrec, [N, N + Nnoise], [hlp.Ftheta, hlp.Fsigma], beta, rNrec_u=rNrec_u)
        a_ui = a_ui[steps_warmup:]
        self.assertLess(abs(np.mean(a_ui) + w / 2. - expected_mu_input),
                        0.05 * abs(expected_mu_input))
        self.assertLess(abs(np.mean(np.std(a_ui, axis=0)) - expected_std_input),
                        0.05 * expected_std_input)

if __name__ == '__main__':
    unittest.main()
