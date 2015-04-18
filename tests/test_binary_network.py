import unittest
import numpy as np
import numpy.testing as nptest

import helper as hlp
import network as bnet

np.random.seed(123456)


class HelperTestCase(unittest.TestCase):

    def test_BM_weight_matrix(self):
        M = 3
        N = 3
        expected_diag = np.zeros(M * N)
        expected_offdiag = np.zeros((N, N))
        W = hlp.create_BM_weight_matrix(N, M)
        self.assertGreaterEqual(1., np.max(W))
        self.assertLessEqual(-1., np.min(W))
        self.assertEqual((M * N, M * N), np.shape(W))
        nptest.assert_array_equal(expected_diag, W.diagonal())
        self.assertEqual(0., np.sum(W - W.T))
        nptest.assert_array_equal(expected_offdiag, W[N:2 * N, :N])
        nptest.assert_array_equal(expected_offdiag, W[:N, N:2 * N])

    def test_BM_biases(self):
        M = 3
        N = 3
        b = hlp.create_BM_biases(N, M)
        self.assertEqual(M * N, len(b))
        expected_max = np.ones(M * N)
        expected_min = -1. * np.ones(M * N)
        nptest.assert_array_less(expected_min, b)
        nptest.assert_array_less(b, expected_max)

    def test_random_weight_matrix(self):
        N = 100
        w = 0.2
        g = 6
        epsilon = 0.1
        gamma = 0.8
        W = hlp.create_connectivity_matrix(N, w, g, epsilon, gamma)
        expected_diag = np.zeros(N)
        nptest.assert_array_equal(expected_diag, W.diagonal())
        NE = int(gamma * N)
        NI = N - NE
        for l in W:
            self.assertEqual(len(l[l > 0]), epsilon * NE)
            self.assertAlmostEqual(np.sum(l[l > 0]), epsilon * NE * w)
            self.assertEqual(len(l[l < 0]), epsilon * NI)
            self.assertAlmostEqual(
                np.sum(l[l < 0]), -1. * epsilon * NI * w * g)
            self.assertAlmostEqual(
                1. * len(l[l > 0]) / len(l[l < 0]), gamma / (1. - gamma))

    def test_noise_weight_matrix(self):
        Knoise = 100
        N = 3
        w = 0.2
        g = 6
        epsilon = 0.9
        Nnoise = int(Knoise/epsilon)
        gamma = 0.3
        W = hlp.create_noise_connectivity_matrix(
            N, Nnoise, gamma, g, w, epsilon)
        Knoise = int(epsilon * Nnoise)
        KEnoise = int(gamma * Knoise)
        KInoise = int(Knoise - KEnoise)
        for l in W:
            self.assertEqual(len(l[l > 0]), KEnoise)
            self.assertAlmostEqual(np.sum(l[l > 0]), KEnoise * w)
            self.assertEqual(len(l[l < 0]), KInoise)
            self.assertAlmostEqual(
                np.sum(l[l < 0]), -1. * KInoise * w * g)
            self.assertAlmostEqual(
                1. * len(l[l > 0]) / len(l[l < 0]), 1. * KEnoise / KInoise)

    def test_hybridnoise_weight_matrix(self):
        Knoise = 100
        N = 3
        w = 0.2
        g = 6
        epsilon = 0.9
        Nnoise = int(Knoise/epsilon)
        gamma = 0.3
        W = hlp.create_hybridnoise_connectivity_matrix(
            N, Nnoise, gamma, g, w, epsilon)
        Knoise = int(epsilon * Nnoise)
        KEnoise = int(gamma * Knoise)
        KInoise = int(Knoise - KEnoise)
        for l in W:
            self.assertEqual(len(l[l > 0]), KEnoise)
            self.assertAlmostEqual(np.sum(l[l > 0]), KEnoise * w)
            self.assertEqual(len(l[l < 0]), KInoise)
            self.assertAlmostEqual(
                np.sum(l[l < 0]), -1. * KInoise * w * g)
            self.assertAlmostEqual(
                1. * len(l[l > 0]) / len(l[l < 0]), 1. * KEnoise / KInoise)

    def test_indep_noise_weight_matrix(self):
        Knoise = 100
        N = 3
        w = 0.2
        g = 6
        gamma = 0.3
        W = hlp.create_indep_noise_connectivity_matrix(
            N, Knoise, gamma, g, w)
        KEnoise = int(gamma * Knoise)
        KInoise = int(Knoise - KEnoise)
        for l in W:
            self.assertEqual(len(l[l > 0]), KEnoise)
            self.assertAlmostEqual(np.sum(l[l > 0]), KEnoise * w)
            self.assertEqual(len(l[l < 0]), KInoise)
            self.assertAlmostEqual(
                np.sum(l[l < 0]), -1. * KInoise * w * g)
            self.assertAlmostEqual(
                1. * len(l[l > 0]) / len(l[l < 0]), 1. * KEnoise / KInoise)

    def test_noise_recurrent_weight_matrix(self):
        Nnoise = 100
        N = 200
        epsilon = 0.2
        W = hlp.create_noise_recurrent_connectivity_matrix(
            N, Nnoise, epsilon)
        self.assertGreaterEqual(1., np.max(W))
        self.assertLessEqual(-1., np.min(W))
        self.assertLess(np.sum(W), 0.01*N*Nnoise*epsilon)
        for l in W:
            self.assertEqual(len(l[l > 0])+len(l[l < 0]), epsilon * N)

    def test_get_energy(self):
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0.2, 0.2])
        s = np.array([1, 0])
        beta = .5
        expected_energy = -1. * beta * \
            np.sum(0.5 * np.dot(s.T, np.dot(W, s)) + np.dot(b, s))
        energy = hlp.get_energy(W, b, s, beta)
        self.assertAlmostEqual(expected_energy, energy)

    def test_get_theo_joints(self):
        N = 3
        W = hlp.create_BM_weight_matrix(N)
        b = hlp.create_BM_biases(N)
        beta = 0.5
        expected_joints = []
        states = hlp.get_states(N)
        for s in states:
            expected_joints.append(
                np.exp(-1. * beta * hlp.get_energy(W, b, s)))
        expected_joints = 1. * \
            np.array(expected_joints) / np.sum(expected_joints)
        joints = hlp.get_theo_joints(W, b, beta)
        nptest.assert_array_almost_equal(expected_joints, joints)
        M = 2
        W = hlp.create_BM_weight_matrix(N, M)
        b = hlp.create_BM_biases(N, M)
        beta = 0.5
        joints = hlp.get_theo_joints(W, b, beta, M)
        for i in range(M):
            expected_joints = []
            states = hlp.get_states(N)
            for s in states:
                expected_joints.append(
                    np.exp(-1. * beta * hlp.get_energy(W[i * N:(i + 1) * N, i * N:(i + 1) * N], b[i * N:(i + 1) * N], s)))
            expected_joints = 1. * \
                np.array(expected_joints) / np.sum(expected_joints)
            nptest.assert_array_almost_equal(expected_joints, joints[i])

    def test_get_theo_marginals(self):
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        beta = 0.5
        N = len(b)
        expected_marginals = []
        states = hlp.get_states(N)
        Z = 0
        for s in states:
            Z += np.exp(-1. * beta * hlp.get_energy(W, b, s))
        for i in range(2):
            statesi = states[states[:, i] == 1]
            p = 0
            for s in statesi:
                p += np.exp(-1. * beta * hlp.get_energy(W, b, s))
            expected_marginals.append(1. / Z * p)
        marginals = hlp.get_theo_marginals(W, b, beta)
        nptest.assert_array_almost_equal(expected_marginals, marginals)

    def test_get_states(self):
        N = 2
        expected_states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        states = hlp.get_states(N)
        nptest.assert_array_equal(expected_states, states)

    def test_get_variance_get_std(self):
        mu = 0.2
        expected_variance = mu * (1. - mu)
        expected_std = np.sqrt(expected_variance)
        variance = hlp.get_sigma2(mu)
        self.assertAlmostEqual(expected_variance, variance)
        std = hlp.get_sigma(mu)
        self.assertAlmostEqual(expected_std, std)

    def test_get_joints(self):
        N = 3
        steps = int(1e5)
        steps_warmup = 1e4
        a_s = np.random.randint(0, 2, N * steps).reshape(steps, N)
        a_s[:steps_warmup,:] = 0
        expected_joints = np.array([1. / (2 ** N)] * 2 ** N)
        joints = hlp.get_joints(a_s, steps_warmup)
        nptest.assert_array_almost_equal(expected_joints, joints, decimal=2)
        M = 2
        N = 3
        a_s = np.random.randint(0, 2, M * N * steps).reshape(steps, M * N)
        a_s[:steps_warmup, :] = 0
        joints = hlp.get_joints(a_s, steps_warmup, M)
        for i in range(M):
            nptest.assert_array_almost_equal(
                expected_joints, joints[i], decimal=2)

    def test_get_joints_sparse(self):
        N = 5
        steps = 4e4
        steps_warmup = 1e3
        a_s = np.vstack([np.random.randint(0, N, steps), np.random.randint(0, 2, steps)]).T
        a_s[:steps_warmup, 1] = 0
        expected_joints = np.ones(2**N) * 1. / (2 ** N)
        joints = hlp.get_joints_sparse(np.zeros(N), a_s, steps_warmup)
        self.assertAlmostEqual(1., np.sum(joints))
        nptest.assert_array_almost_equal(expected_joints, joints, decimal=2)
        M = 3
        a_s = np.vstack([np.random.randint(0, M * N, steps), np.random.randint(0, 2, steps)]).T
        a_s[:steps_warmup, 1] = 0
        joints = hlp.get_joints_sparse(np.zeros(M * N), a_s, steps_warmup, M)
        expected_sum = np.ones(M)
        nptest.assert_array_almost_equal(expected_sum, np.sum(joints, axis=1))
        for i in range(M):
            nptest.assert_array_almost_equal(
                expected_joints, joints[i], decimal=2)
        a_s = np.vstack([np.random.randint(0, N, steps), np.random.randint(0, 2, steps)]).T
        a_s[:steps_warmup, 1] = 0
        a_s[np.where(a_s[:, 0] == 4), 1] = 0
        expected_joints = 2. *np.ones(2**N) * 1. / (2 ** N)
        expected_joints[1::2] = 0.
        joints = hlp.get_joints_sparse(np.zeros(N), a_s, steps_warmup)
        nptest.assert_array_equal(np.zeros(2**N/2), joints[1::2])
        self.assertAlmostEqual(1., np.sum(joints))
        nptest.assert_array_almost_equal(expected_joints, joints, decimal=2)
        joints = hlp.get_joints_sparse(np.zeros(N), a_s, steps_warmup, prior='uniform')
        nptest.assert_array_less(np.zeros(2**N), joints)

    def test_get_marginals(self):
        N = int(1e5)
        a_s = np.random.randint(0, 2, N).reshape(int(N / 2), 2)
        expected_marginals = [0.5, 0.5]
        marginals = hlp.get_marginals(a_s, 0)
        nptest.assert_array_almost_equal(
            expected_marginals, marginals, decimal=2)

    def test_DKL(self):
        p = np.array([0.1, 0.3, 0.2, 0.4])
        q = np.array([0.2, 0.3, 0.1, 0.4])
        expected_DKL = np.sum([p[i] * np.log(p[i] / q[i])
                               for i in range(len(p))])
        DKL = hlp.get_DKL(p, q)
        nptest.assert_array_almost_equal(expected_DKL, DKL)
        M = 2
        p = np.array([[0.1, 0.3, 0.2, 0.4],
                      [0.6, 0.2, 0.1, 0.1]])
        q = np.array([[0.2, 0.3, 0.1, 0.4],
                      [0.5, 0.2, 0.2, 0.1]])
        DKL = hlp.get_DKL(p, q, M)
        for j in range(M):
            expected_DKL = np.sum([p[j, i] * np.log(p[j, i] / q[j, i])
                                   for i in range(len(p[j, :]))])
            nptest.assert_array_almost_equal(expected_DKL, DKL[j])

    def test_theta(self):
        x = np.array([1., -.1, -1., .1])
        expected_y = np.array([1., 0., 0., 1.])
        for yi, xi in zip(expected_y, x):
            self.assertAlmostEqual(yi, hlp.theta(xi))
        self.assertRaises(ValueError, hlp.theta, 0.)

    def test_sigmoidal(self):
        x = np.random.rand(int(1e2))
        expected_y = 1. / (1. + np.exp(-x))
        y = hlp.sigma(x)
        nptest.assert_array_almost_equal(expected_y, y)

    def test_sigmainv(self):
        beta = 0.781
        expected_x = np.random.rand(int(1e2))
        y = hlp.sigma(expected_x, beta)
        x = hlp.sigmainv(y, beta)
        nptest.assert_array_almost_equal(expected_x, x)

    def test_mun_sigman(self):
        N = 300
        epsilon = 0.1
        K = epsilon * N
        gamma = 0.8
        g = 6.
        w = 0.2
        smu = 0.2
        steps = int(1e5)
        KE = int(gamma * K)
        KI = K - KE
        sigmas = hlp.get_sigma(smu)
        xE = w * np.random.normal(smu, sigmas, (steps, KE))
        xI = -g * w * np.random.normal(smu, sigmas, (steps, KI))
        x = np.sum([np.sum(xE, axis=1), np.sum(xI, axis=1)], axis=0)
        expected_mu = np.mean(x)
        expected_sigma = np.std(x)
        mu = hlp.get_mu_input(epsilon, N, gamma, g, w, smu)
        self.assertAlmostEqual(expected_mu, mu, delta=0.02 * abs(expected_mu))
        sigma = hlp.get_sigma_input(epsilon, N, gamma, g, w, smu)
        self.assertAlmostEqual(
            expected_sigma, sigma, delta=0.02 * abs(expected_sigma))

    def test_Fsigma(self):
        samples = int(5e4)
        x = np.random.rand(samples)
        expected_y = 1. / (1. + np.exp(-0.5))
        y = []
        for xi in x:
            y.append(hlp.Fsigma(xi))
        y = np.mean(y)
        nptest.assert_array_almost_equal(expected_y, y, decimal=2)

    def test_beta_sigma_noise(self):
        beta_expected = 1.781
        sigma = hlp.get_sigma_input_from_beta(beta_expected)
        beta = hlp.get_beta_from_sigma_input(sigma)
        self.assertAlmostEqual(beta_expected, beta)

    def test_euclidean_distance(self):
        N = 25
        x = np.random.rand(N)
        y = np.random.rand(N)
        expected_dist = np.sqrt(np.dot(x-y, x-y))
        dist = hlp.get_euclidean_distance(x, y)
        nptest.assert_array_almost_equal(expected_dist, dist)
        y = -x
        expected_dist = 2. * np.sqrt(np.sum(x**2))
        dist = hlp.get_euclidean_distance(x, y)
        nptest.assert_array_almost_equal(expected_dist, dist)
        y = x
        expected_dist = 0.
        dist = hlp.get_euclidean_distance(x, y)
        self.assertAlmostEqual(expected_dist, dist)

    def test_adjusted_weights_and_bias(self):
        N = 10
        beta = .5
        beta_eff = 0.5
        J = hlp.create_BM_weight_matrix(N)
        b = hlp.create_BM_biases(N)
        b_eff = b - 20.
        expected_J_eff = beta/beta_eff * J
        expected_b_eff = beta/beta_eff * b + b_eff
        J_eff, b_eff = hlp.get_adjusted_weights_and_bias(J, b, b_eff, beta_eff, beta)
        nptest.assert_array_almost_equal(expected_J_eff, J_eff)
        nptest.assert_array_almost_equal(expected_b_eff, b_eff)


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
            elif i ==2:
                s0, a_times, a_s = sim(W, b, tau, sinit, steps * tau / N,
                                       [0, Nrec], [N1, N], [hlp.Fsigma, F2],
                                       beta=beta)
                a_s = hlp.get_all_states_from_sparse(s0, a_s, 0)
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
        s0, a_times, a_s = bnet.simulate_eve_sparse(
            W, b, tau, sinit, Tmax, rNrec, [N], [hlp.Fsigma], beta=beta)
        self.assertGreater(np.min(a_times), 0.)
        self.assertLess(np.min(a_times), tau)
        self.assertLess(np.max(a_times), 1.01 * Tmax)
        self.assertEqual(len(a_times), len(a_s))
        nptest.assert_array_equal(sinit, s0)
        joints = hlp.get_joints_sparse(s0, a_s, 0)
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
            expected_marginals = hlp.get_theo_marginals(W, b, beta)
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
        W_brn = hlp.create_connectivity_matrix(N, w, g, epsilon, gamma)
        b_brn = -1. * \
            hlp.get_mu_input(epsilon, N, gamma, g, w, mu_target) * \
            np.ones(N) - 1. * w / 2
        a_times_brn, a_s_brn = bnet.simulate_eve(
            W_brn, b_brn, tau, sinit.copy(), time, [0, Nrec], [N], [hlp.theta])
        self.assertAlmostEqual(
            mu_target, np.mean(a_s_brn), delta=0.1 * np.mean(a_s_brn))
        times_bin_brn, st_brn = hlp.bin_binary_data(
            a_times_brn, a_s_brn, tbin, 0., time)
        timelag_brn, autof_brn = hlp.autocorrf(
            times_bin_brn, st_brn[:30], tmax)
        nptest.assert_array_almost_equal(expected_timelag, timelag_brn)
        self.assertTrue(abs(np.sum(autof_brn - expected_autof))
                        < 0.5 * np.sum(abs(autof_brn)))

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
        self.assertTrue(abs(np.sum(abs(autof - expected_autof)))
                        < 0.5 * np.sum(abs(autof)))


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
        W_brn = hlp.create_connectivity_matrix(N, w, g, epsilon, gamma)
        b_brn = -1. * \
            hlp.get_mu_input(epsilon, N, gamma, g, w, mu_target) * \
            np.ones(N) - 1. * w / 2
        a_times_brn, a_s_brn = bnet.simulate_eve(
            W_brn, b_brn, tau, sinit.copy(), time, [0, Nrec], [N], [hlp.theta])
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
        N = 12
        Nnoise = 120
        sinit = np.zeros(N + Nnoise)
        tau = 10.
        Nrec = N
        time = 1.5e4
        mu_target = 0.42
        Nrec_ui = N
        beta = 0.5

        w = 0.2
        g = 8.
        gamma = 0.2
        epsilon = 0.3
        expected_mu_input = hlp.get_mu_input(
            epsilon, Nnoise, gamma, g, w, mu_target)
        expected_std_input = hlp.get_sigma_input(
            epsilon, Nnoise, gamma, g, w, mu_target)

        # Network case (correlated sources)
        W_brn = np.zeros((N + Nnoise, N + Nnoise))
        W_brn[:N, N:] = hlp.create_noise_connectivity_matrix(
            N, Nnoise, gamma, g, w, epsilon)
        W_brn[N:, N:] = hlp.create_connectivity_matrix(
            Nnoise, w, g, epsilon, gamma)
        b_brn = np.zeros(N + Nnoise)
        b_brn[:N] = -w / 2.
        b_brn[N:] = -1. * \
            hlp.get_mu_input(epsilon, Nnoise, gamma, g, w, mu_target) - \
            1. * w / 2
        for i, sim in enumerate([bnet.simulate, bnet.simulate_eve]):
            if i == 0:
                a_times_brn, a_s_brn, a_times_ui_brn, a_ui_brn = sim(
                    W_brn, b_brn, sinit.copy(), time, Nrec,
                    [N + Nnoise], [hlp.theta], Nrec_ui=Nrec_ui, beta=beta)
                steps_warmup = 0.1 * time/(N+Nnoise) * N
            elif i == 1:
                a_times_brn, a_s_brn, a_times_ui_brn, a_ui_brn = sim(
                    W_brn, b_brn, tau, sinit.copy(), time, [0, Nrec],
                    [N + Nnoise], [hlp.theta], Nrec_ui=Nrec_ui, beta=beta)
                steps_warmup = 0.1 * Nrec_ui * time/tau
            a_ui_brn = a_ui_brn[steps_warmup:]
            self.assertLess(abs(np.mean(a_ui_brn) + w / 2. - expected_mu_input),
                            0.04 * abs(expected_mu_input))
            self.assertLess(
                (np.mean(np.std(a_ui_brn, axis=0)) - expected_std_input), 0)

        # Poisson case (independent sources)
        W = np.zeros((N + Nnoise, N + Nnoise))
        W[:N, N:] = hlp.create_noise_connectivity_matrix(
            N, Nnoise, gamma, g, w, epsilon)
        b = np.zeros(N + Nnoise)
        b[:N] = -w / 2.
        b[N:] = hlp.sigmainv(mu_target)
        for i,sim in enumerate([bnet.simulate, bnet.simulate_eve]):
            if i == 0:
                a_times, a_s, a_times_ui, a_ui = sim(
                    W, b, sinit.copy(), time, Nrec, [N, N + Nnoise], [hlp.theta, hlp.Fsigma], Nrec_ui=Nrec_ui)
                steps_warmup = 0.1 * time/(N+Nnoise) * N
            elif i == 1:
                a_times, a_s, a_times_ui, a_ui = sim(
                    W, b, tau, sinit.copy(), time, [0, Nrec], [N, N + Nnoise], [hlp.theta, hlp.Fsigma], Nrec_ui=Nrec_ui)
                steps_warmup = 0.1 * Nrec_ui * time/tau
            a_ui = a_ui[steps_warmup:]
            self.assertLess(abs(np.mean(a_ui) + w / 2. - expected_mu_input),
                            0.05 * abs(expected_mu_input))
            self.assertLess(abs(np.mean(np.std(a_ui, axis=0)) - expected_std_input),
                            0.05 * expected_std_input)


if __name__ == '__main__':
    unittest.main()
