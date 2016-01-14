# global imports
import unittest
import numpy as np
import numpy.testing as nptest

# local imports
from .. import helper as bhlp

np.random.seed(13456)


class HelperTestCase(unittest.TestCase):

    def test_get_states(self):
        N = 2
        expected_states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        states = bhlp.get_states(N)
        nptest.assert_array_equal(expected_states, states)

    def test_get_states_as_strings(self):
        N = 2
        expected_states = np.array(['00', '01', '10', '11'])
        states = bhlp.get_states_as_strings(N)
        nptest.assert_array_equal(expected_states, states)

    def test_state_array_to_string(self):
        s = np.array([0, 0, 1])
        expected_s_string = '001'
        s_string = bhlp.state_array_to_string(s)
        self.assertEqual(expected_s_string, s_string)

    def test_state_array_to_int(self):
        s = np.array([1, 0, 1])
        expected_s_int = s[0] * 2 ** 2 + s[1] * 2 ** 1 + s[2] * 2 ** 0
        s_int = bhlp.state_array_to_int(s)
        self.assertEqual(expected_s_int, s_int)

    def test_state_string_from_int(self):
        s_int = 4
        N = 5
        expected_s_string = '00100'
        s_string = bhlp.state_string_from_int(s_int, N)
        self.assertEqual(expected_s_string, s_string)

    def test_state_array_from_int(self):
        s_int = 5
        N = 4
        expected_s = np.array([0, 1, 0, 1])
        s = bhlp.state_array_from_int(s_int, N)
        nptest.assert_array_equal(expected_s, s)

    def test_outdegree_distribution(self):
        M = 200
        N = 2000
        gamma = 0.8
        g = 1.
        w = 1.
        epsilon = 0.20
        NE = int(gamma * N)
        NI = int(N - NE)
        KE = int(epsilon * NE)
        KI = int(epsilon * NI)
        K = KE + KI

        W = bhlp.create_noise_weight_matrix(M, N, gamma, g, w, epsilon)
        outdegree = []
        for i in xrange(N):
            outdegree.append(len(np.where(abs(W[:, i]) > 0.)[0]))
        bins_out = np.arange(0, K)
        hist_out, bins_out = np.histogram(outdegree, bins_out, density=True)

        theo_hist_out = bhlp.outdegree_distribution(M, K, N, bins_out[:-1])
        nptest.assert_array_almost_equal(theo_hist_out, hist_out, decimal=1)

    def test_shared_input_distribution(self):
        M = 200
        N = 2000
        gamma = 0.8
        g = 1.
        w = 1.
        epsilon = 0.20
        NE = int(gamma * N)
        NI = int(N - NE)
        KE = int(epsilon * NE)
        KI = int(epsilon * NI)
        K = KE + KI

        W = bhlp.create_noise_weight_matrix(M, N, gamma, g, w, epsilon)
        shared_inputs = []
        for i in xrange(M):
            for j in xrange(M):
                if i != j:
                    shared_inputs.append(
                        len(np.where(abs(W[i, :] * W[j, :]) > 1e-12)[0]))

        bins_shared = np.arange(0, K)
        hist_shared, bins_shared = np.histogram(
            shared_inputs, bins_shared, density=True)

        theo_hist_shared = bhlp.shared_input_distribution(
            K, N, bins_shared[:-1])

        nptest.assert_array_almost_equal(
            theo_hist_shared, hist_shared, decimal=2)

    def test_create_BM_weight_matrix(self):
        N = 200
        expected_diag = np.zeros(N)

        # test with uniform distribution
        W = bhlp.create_BM_weight_matrix(
            N, np.random.uniform, low=-1., high=1.)
        self.assertGreaterEqual(1., np.max(W))
        self.assertLessEqual(-1., np.min(W))
        self.assertEqual((N, N), np.shape(W))
        nptest.assert_array_equal(expected_diag, W.diagonal())
        self.assertEqual(0., np.sum(W - W.T))

        # test with normal distribution
        mu = -1.
        sigma = 1.5
        W = bhlp.create_BM_weight_matrix(
            N, np.random.normal, loc=mu, scale=sigma)
        weights = [W[i, j] for i in xrange(N) for j in xrange(N) if i != j]
        self.assertAlmostEqual(mu, np.mean(weights), delta=0.02)
        self.assertAlmostEqual(sigma, np.std(weights), delta=0.02)
        self.assertEqual((N, N), np.shape(W))
        nptest.assert_array_equal(expected_diag, W.diagonal())
        self.assertEqual(0., np.sum(W - W.T))

        # test with beta distribution and target mean
        mean_weight = -1.45
        W = bhlp.create_BM_weight_matrix(
            N, np.random.beta, mean_weight=mean_weight, a=2., b=2.)
        weights = [W[i, j] for i in xrange(N) for j in xrange(N) if i != j]
        self.assertAlmostEqual(mean_weight, np.mean(weights), delta=0.01)
        self.assertEqual((N, N), np.shape(W))
        nptest.assert_array_equal(expected_diag, W.diagonal())
        self.assertEqual(0., np.sum(W - W.T))

    def test_multi_BM_weight_matrix(self):
        M = 3
        N = 3
        expected_diag = np.zeros(M * N)
        expected_offdiag = np.zeros((N, N))

        # test with uniform distribution
        W = bhlp.create_multi_BM_weight_matrix(
            N, M, np.random.uniform, low=-1., high=1.)
        self.assertGreaterEqual(1., np.max(W))
        self.assertLessEqual(-1., np.min(W))
        self.assertEqual((M * N, M * N), np.shape(W))
        nptest.assert_array_equal(expected_diag, W.diagonal())
        self.assertEqual(0., np.sum(W - W.T))
        nptest.assert_array_equal(expected_offdiag, W[N:2 * N, :N])
        nptest.assert_array_equal(expected_offdiag, W[:N, N:2 * N])

    def test_BRN_weight_matrix(self):
        N = 100
        w = 0.2
        g = 6
        epsilon = 0.1
        gamma = 0.8
        W = bhlp.create_BRN_weight_matrix(N, w, g, epsilon, gamma)
        expected_diag = np.zeros(N)
        nptest.assert_array_equal(expected_diag, W.diagonal())
        NE = int(gamma * N)
        NI = N - NE
        self.assertEqual(np.shape(W), (N, N))
        self.assertTrue(np.all(W[:, :NE] >= 0.))
        self.assertTrue(np.all(W[:, NE:] <= 0.))
        self.assertEqual(len(np.where(W[:, :NE] > 0.)[0]), epsilon * NE * N)
        self.assertEqual(len(np.where(W[:, NE:] < 0.)[0]), epsilon * NI * N)
        self.assertEqual(np.unique(W[W > 0]), [w])
        self.assertEqual(np.unique(W[W < 0]), [-g * w])
        self.assertAlmostEqual(
            1. * len(W[W > 0]) / len(W[W < 0]), gamma / (1. - gamma))

    def test_BRN_weight_matrix_fixed_indegree(self):
        N = 100
        K = 20
        w = 0.2
        g = 6
        gamma = 0.8
        W = bhlp.create_BRN_weight_matrix_fixed_indegree(N, w, g, K, gamma)
        expected_diag = np.zeros(N)
        nptest.assert_array_equal(expected_diag, W.diagonal())
        NE = int(gamma * N)
        self.assertEqual(np.shape(W), (N, N))
        self.assertTrue(np.all(W[:, :NE] >= 0.))
        self.assertTrue(np.all(W[:, NE:] <= 0.))
        self.assertEqual(len(W[np.abs(W) > 0]), K * N)
        self.assertAlmostEqual(len(np.where(W[:, :NE] > 0.)[0]), gamma * K * N)
        self.assertAlmostEqual(len(np.where(W[:, NE:] < 0.)[0]), (1. - gamma) * K * N)
        self.assertEqual(np.unique(W[W > 0]), [w])
        self.assertEqual(np.unique(W[W < 0]), [-g * w])
        self.assertAlmostEqual(
            1. * len(W[W > 0]) / len(W[W < 0]), gamma / (1. - gamma))

    def test_noise_weight_matrix(self):
        Knoise = 100
        M = 3
        w = 0.2
        g = 6
        epsilon = 0.8
        Nnoise = int(Knoise / epsilon)
        gamma = 0.3
        W = bhlp.create_noise_weight_matrix(M, Nnoise, gamma, g, w,
                                            epsilon)
        NEnoise = int(gamma * Nnoise)

        self.assertEqual(np.shape(W), (M, Nnoise))
        self.assertTrue(np.all(W[:, :NEnoise] >= 0.))
        self.assertTrue(np.all(W[:, NEnoise:] <= 0.))
        self.assertEqual(len(W[np.abs(W) > 0]), epsilon * Nnoise * M)
        self.assertEqual(np.unique(W[W > 0]), [w])
        self.assertEqual(np.unique(W[W < 0]), [-g * w])
        self.assertAlmostEqual(
            1. * len(W[W > 0]) / len(W[W < 0]), gamma / (1. - gamma), delta=0.1)

    def test_noise_weight_matrix_fixed_indegree(self):
        M = 100
        Knoise = 200
        w = 0.3
        g = 8.
        gamma = 0.3
        a_epsilon = np.arange(0.05, 1., 0.04)
        for epsilon in a_epsilon:
            Nnoise = int(Knoise / epsilon)
            W = bhlp.create_noise_weight_matrix_fixed_indegree(
                M, Nnoise, gamma, g, w, Knoise)
            self.assertEqual(np.shape(W), (M, Nnoise))
            self.assertTrue(np.all(W[:, :gamma * Nnoise] >= 0.))
            self.assertTrue(np.all(W[:, gamma * Nnoise:] <= 0.))
            self.assertAlmostEqual(len(np.where(W > 0.)[0]), gamma * Knoise * M)
            self.assertAlmostEqual(len(np.where(W < 0.)[0]), (1. - gamma) * Knoise * M)
            self.assertEqual(len(np.where(np.abs(W) > 0.)[0]), Knoise * M)
            self.assertAlmostEqual(np.sum(W[W > 0]), gamma * Knoise * M * w)
            self.assertAlmostEqual(np.sum(W[W < 0]), -g * (1. - gamma) * Knoise * M * w)

    def test_noise_weight_matrix_2dshuffle(self):
        Knoise = 100
        N = 3
        w = 0.2
        g = 6
        epsilon = 0.9
        Nnoise = int(Knoise / epsilon)
        gamma = 0.3
        W = bhlp.create_noise_weight_matrix_2dshuffle(
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
        W = bhlp.create_indep_noise_weight_matrix(
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

    def test_get_energy(self):
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0.2, 0.2])
        s = np.array([1, 0])
        beta = .5
        expected_energy = -1. * beta * \
            np.sum(0.5 * np.dot(s.T, np.dot(W, s)) + np.dot(b, s))
        energy = bhlp.get_energy(W, b, s, beta)
        self.assertAlmostEqual(expected_energy, energy)

    def test_get_theo_joints(self):
        N = 3
        W = bhlp.create_BM_weight_matrix(
            N, np.random.uniform, low=-1., high=1.)
        b = bhlp.create_BM_biases(N, np.random.uniform, low=-1., high=1.)
        beta = 0.5
        expected_joints = []
        states = bhlp.get_states(N)
        for s in states:
            expected_joints.append(
                np.exp(-1. * beta * bhlp.get_energy(W, b, s)))
        expected_joints = 1. * \
            np.array(expected_joints) / np.sum(expected_joints)
        joints = bhlp.get_theo_joints(W, b, beta)
        nptest.assert_array_almost_equal(expected_joints, joints)
        M = 2
        W = bhlp.create_multi_BM_weight_matrix(
            N, M, np.random.uniform, low=-1., high=1.)
        b = bhlp.create_multi_BM_biases(
            N, M, np.random.uniform, low=-1., high=1.)
        beta = 0.5
        joints = bhlp.get_theo_joints_multi_bm(W, b, beta, M)
        for i in range(M):
            expected_joints = []
            states = bhlp.get_states(N)
            for s in states:
                expected_joints.append(
                    np.exp(-1. * beta * bhlp.get_energy(W[i * N:(i + 1) * N, i * N:(i + 1) * N], b[i * N:(i + 1) * N], s)))
            expected_joints = 1. * \
                np.array(expected_joints) / np.sum(expected_joints)
            nptest.assert_array_almost_equal(expected_joints, joints[i])

    def test_get_conditionals_from_joints(self):
        N = 3
        joints = np.ones(2 ** N)
        joints /= np.sum(joints)
        random_vars = [1, 2]
        values = [0, 1]
        expected_states = np.array([[0, 0, 1], [1, 0, 1]])
        expected_cond = np.array([0.5, 0.5])
        cond_states, cond = bhlp.get_conditionals_from_joints(N, joints, random_vars, values)
        nptest.assert_array_equal(expected_states, cond_states)
        nptest.assert_array_almost_equal(expected_cond, cond)

    def test_get_theo_marginals(self):
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        beta = 0.5
        N = len(b)
        expected_marginals = []
        states = bhlp.get_states(N)
        Z = 0
        for s in states:
            Z += np.exp(-1. * beta * bhlp.get_energy(W, b, s))
        for i in range(2):
            statesi = states[states[:, i] == 1]
            p = 0
            for s in statesi:
                p += np.exp(-1. * beta * bhlp.get_energy(W, b, s))
            expected_marginals.append(1. / Z * p)
        rvs, marginals = bhlp.get_theo_marginals(W, b, beta)
        nptest.assert_array_almost_equal(expected_marginals, marginals)
        N = 3
        joints = np.ones(2 ** N)
        joints /= np.sum(joints)
        expected_marginals = np.array([0.5, 0.5])
        random_vars = [0, 2]
        rvs, marginals = bhlp.get_marginals_from_joints(N, joints, random_vars)
        nptest.assert_array_almost_equal(expected_marginals, marginals)

    def test_get_variance_get_std(self):
        mu = 0.2
        expected_variance = mu * (1. - mu)
        expected_std = np.sqrt(expected_variance)
        variance = bhlp.get_sigma2(mu)
        self.assertAlmostEqual(expected_variance, variance)
        std = bhlp.get_sigma(mu)
        self.assertAlmostEqual(expected_std, std)

    def test_get_joints(self):
        N = 3
        steps = int(1e5)
        steps_warmup = 1e4
        a_s = np.random.randint(0, 2, N * steps).reshape(steps, N)
        a_s[:steps_warmup, :] = 0
        expected_joints = np.array([1. / (2 ** N)] * 2 ** N)
        joints = bhlp.get_joints(a_s, steps_warmup)
        nptest.assert_array_almost_equal(expected_joints, joints, decimal=2)
        M = 2
        N = 3
        a_s = np.random.randint(0, 2, M * N * steps).reshape(steps, M * N)
        a_s[:steps_warmup, :] = 0
        joints = bhlp.get_joints_multi_bm(a_s, steps_warmup, M)
        for i in range(M):
            nptest.assert_array_almost_equal(
                expected_joints, joints[i], decimal=2)

    def test_get_joints_sparse(self):
        N = 5
        steps = 5e4
        steps_warmup = 1e3
        sinit = bhlp.random_initial_condition(N)
        a_s = np.empty((steps, 2))
        a_s[:, 0] = np.random.randint(0, N, steps)
        a_s[:, 1] = np.random.randint(0, 2, steps)
        a_s[:steps_warmup] = (0, 0)
        expected_joints = np.ones(2 ** N) * 1. / (2 ** N)
        joints = bhlp.get_joints_sparse(sinit, a_s, steps_warmup)
        self.assertAlmostEqual(1., np.sum(joints))
        nptest.assert_array_almost_equal(expected_joints, joints, decimal=2)
        M = 2
        sinit = bhlp.random_initial_condition(N * M)
        a_s = np.empty((steps, 2))
        a_s[:, 0] = np.random.randint(0, N * M, steps)
        a_s[:, 1] = np.random.randint(0, 2, steps)
        a_s[:steps_warmup] = (0, 0)
        joints = bhlp.get_joints_sparse_multi_bm(sinit, a_s, steps_warmup, M)
        expected_sum = np.ones(M)
        nptest.assert_array_almost_equal(expected_sum, np.sum(joints, axis=1))
        for i in range(M):
            nptest.assert_array_almost_equal(
                expected_joints, joints[i], decimal=2)
        sinit = bhlp.random_initial_condition(N)
        a_s = np.empty((steps, 2))
        a_s[:, 0] = np.random.randint(0, N, steps)
        a_s[:, 1] = np.random.randint(0, 2, steps)
        # allow only those states where the last neuron is off.  this
        # make the joint distribution non-trivial, by setting the
        # probability of half of the states to zero.
        a_s[a_s[:, 0] == 4, 1] = 0
        expected_joints = 2. * np.ones(2 ** N) * 1. / (2 ** N)
        expected_joints[1::2] = 0.
        joints = bhlp.get_joints_sparse(sinit, a_s, steps_warmup)
        nptest.assert_array_equal(np.zeros(2 ** N / 2), joints[1::2])
        self.assertAlmostEqual(1., np.sum(joints))
        nptest.assert_array_almost_equal(expected_joints, joints, decimal=2)

    def test_get_marginals(self):
        N = int(1e5)
        steps_warmup = 1e4
        a_s = np.random.randint(0, 2, N).reshape(int(N / 2), 2)
        a_s[:steps_warmup, :] = [0., 0.]
        expected_marginals = [0.5, 0.5]
        marginals = bhlp.get_marginals(a_s, steps_warmup)
        nptest.assert_array_almost_equal(
            expected_marginals, marginals, decimal=2)

    def test_DKL(self):
        p = np.array([0.1, 0.3, 0.2, 0.4])
        q = np.array([0.2, 0.3, 0.1, 0.4])
        expected_DKL = np.sum([p[i] * np.log(p[i] / q[i])
                               for i in range(len(p))])
        DKL = bhlp.get_DKL(p, q)
        nptest.assert_array_almost_equal(expected_DKL, DKL)
        M = 2
        p = np.array([[0.1, 0.3, 0.2, 0.4],
                      [0.6, 0.2, 0.1, 0.1]])
        q = np.array([[0.2, 0.3, 0.1, 0.4],
                      [0.5, 0.2, 0.2, 0.1]])
        DKL = bhlp.get_DKL_multi_bm(p, q, M)
        for j in range(M):
            expected_DKL = np.sum([p[j, i] * np.log(p[j, i] / q[j, i])
                                   for i in range(len(p[j, :]))])
            nptest.assert_array_almost_equal(expected_DKL, DKL[j])

    def test_theta(self):
        x = np.array([1., -.1, -1., .1, 0.])
        expected_y = np.array([1., 0., 0., 1., 1.])
        for yi, xi in zip(expected_y, x):
            self.assertAlmostEqual(yi, bhlp.theta(xi))

    def test_sigmoidal(self):
        x = np.random.rand(int(1e2))
        expected_y = 1. / (1. + np.exp(-x))
        y = bhlp.sigma(x)
        nptest.assert_array_almost_equal(expected_y, y)

    def test_sigmainv(self):
        beta = 0.781
        expected_x = np.random.rand(int(1e2))
        y = bhlp.sigma(expected_x, beta)
        x = bhlp.sigmainv(y, beta)
        nptest.assert_array_almost_equal(expected_x, x)

    def test_mun_sigman(self):
        N = 300
        epsilon = 0.1
        K = epsilon * N
        gamma = 0.8
        g = 6.
        w = 0.2
        smu = 0.2
        steps = int(1e6)
        KE = int(gamma * K)
        KI = K - KE
        sigmas = bhlp.get_sigma(smu)

        # generate E and I input, and calculate statistics of combined
        # input
        xE = w * np.random.normal(smu, sigmas, (steps, KE))
        xI = -g * w * np.random.normal(smu, sigmas, (steps, KI))
        x = np.sum([np.sum(xE, axis=1), np.sum(xI, axis=1)], axis=0)
        expected_mu = np.mean(x)
        expected_sigma = np.std(x)

        # compare to theoretical value
        mu = bhlp.get_mu_input(epsilon, N, gamma, g, w, smu)
        self.assertAlmostEqual(expected_mu, mu, delta=0.02 * abs(expected_mu))
        sigma = bhlp.get_sigma_input(epsilon, N, gamma, g, w, smu)
        self.assertAlmostEqual(
            expected_sigma, sigma, delta=0.02 * abs(expected_sigma))

    def test_Fsigma(self):
        samples = int(5e4)
        x = np.random.rand(samples)
        expected_y = 1. / (1. + np.exp(-0.5))
        y = []
        for xi in x:
            y.append(bhlp.Fsigma(xi))
        y = np.mean(y)
        nptest.assert_array_almost_equal(expected_y, y, decimal=2)

    def test_beta_sigma_noise(self):
        beta_expected = 1.781
        sigma = bhlp.get_sigma_input_from_beta(beta_expected)
        beta = bhlp.get_beta_from_sigma_input(sigma)
        self.assertAlmostEqual(beta_expected, beta)

    def test_euclidean_distance(self):
        N = 25
        x = np.random.rand(N)
        y = np.random.rand(N)
        expected_dist = np.sqrt(np.dot(x - y, x - y))
        dist = bhlp.get_euclidean_distance(x, y)
        nptest.assert_array_almost_equal(expected_dist, dist)
        y = -x
        expected_dist = 2. * np.sqrt(np.sum(x ** 2))
        dist = bhlp.get_euclidean_distance(x, y)
        nptest.assert_array_almost_equal(expected_dist, dist)
        y = x
        expected_dist = 0.
        dist = bhlp.get_euclidean_distance(x, y)
        self.assertAlmostEqual(expected_dist, dist)

    def test_adjusted_weights_and_bias(self):
        N = 10
        beta = .5
        beta_eff = 0.5
        J = bhlp.create_BM_weight_matrix(N, np.random.uniform, low=-1, high=1.)
        b = bhlp.create_BM_biases(N, np.random.uniform, low=-1, high=1.)
        b_eff = b - 20.
        expected_J_eff = beta / beta_eff * J
        expected_b_eff = beta / beta_eff * b + b_eff
        J_eff, b_eff = bhlp.get_adjusted_weights_and_bias(
            J, b, b_eff, beta_eff, beta)
        nptest.assert_array_almost_equal(expected_J_eff, J_eff)
        nptest.assert_array_almost_equal(expected_b_eff, b_eff)

    def test_get_isi(self):
        N = 3
        a_times = np.array([0.1, 0.5, 0.7, 1.5, 1.7])
        a_s = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]])
        expected_isi = np.array([[0.5, 1.2], [0.7], [1.5]])
        isi = bhlp.get_isi(a_times, a_s)
        for i in xrange(N):
            nptest.assert_array_almost_equal(expected_isi[i], isi[i])

    def test_get_transition_count(self):
        N = 3
        a_s = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]])
        expected_counts = np.array([2, 1, 1])
        counts = bhlp.get_transition_count(a_s)
        nptest.assert_array_equal(expected_counts, counts)
        expected_average_counts = 1. * np.sum(expected_counts) / N
        average_counts = bhlp.get_transition_count(a_s, average=True)
        self.assertAlmostEqual(expected_average_counts, average_counts)

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
        times_bin, st = bhlp.bin_binary_data(times, a_s, tbin, 0., time)
        nptest.assert_array_equal(expected_times, times_bin)
        nptest.assert_array_equal(expected_bin, st)
