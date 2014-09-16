import unittest
import numpy as np
import numpy.testing as nptest

import helper as hlp
import network as bnet

class HelperTestCase(unittest.TestCase):

    # def setUp(self):
    # def tearDown(self):
    
    def test_BM_weight_matrix(self):
        N = 10
        expected_diag = np.zeros(N)
        W = hlp.create_BM_weight_matrix(N)
        self.assertEqual((N,N), np.shape(W))
        nptest.assert_array_equal(expected_diag, W.diagonal())
        self.assertEqual(0., np.sum(W-W.T))

    def test_BM_biases(self):
        N = 10
        b = hlp.create_BM_biases(N)
        self.assertEqual(N, len(b))
        expected_max = np.ones(N)
        expected_min = np.ones(N)*(-1.)
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
        NE = int(gamma*N)
        NI = N-NE
        for l in W:
            self.assertEqual(len(l[l > 0]), epsilon*NE)
            self.assertAlmostEqual(np.sum(l[l > 0]), epsilon*NE*w)
            self.assertEqual(len(l[l < 0]), epsilon*NI)
            self.assertAlmostEqual(np.sum(l[l < 0]), -1.*epsilon*NI*w*g)
            self.assertAlmostEqual(1.*len(l[l > 0])/len(l[l < 0]), gamma/(1.-gamma))

    def test_get_E(self):
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0.2, 0.2])
        s = np.array([1,0])
        expected_E = np.sum(0.5*np.dot(s.T, np.dot(W, s)) + np.dot(b,s))
        E = hlp.get_E(W, b, s)
        self.assertAlmostEqual(expected_E, E)

    def test_get_theo_joints(self):
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        N = len(b)
        expected_joints = []
        states = hlp.get_states(N)
        for s in states:
            expected_joints.append(np.exp(hlp.get_E(W, b, s)))
        expected_joints = 1.*np.array(expected_joints)/np.sum(expected_joints)
        joints = hlp.get_theo_joints(W,b)
        nptest.assert_array_almost_equal(expected_joints, joints)

    def test_get_theo_marginals(self):
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        N = len(b)
        expected_marginals = []
        states = hlp.get_states(N)
        Z = 0
        for s in states:
            Z += np.exp(hlp.get_E(W, b, s))
        for i in range(2):
            statesi = states[states[:,i] == 1]
            p = 0
            for s in statesi:
                p += np.exp(hlp.get_E(W, b, s))
            expected_marginals.append(1./Z*p)
        marginals = hlp.get_theo_marginals(W,b)
        nptest.assert_array_almost_equal(expected_marginals, marginals)

    def test_get_states(self):
        N = 2
        expected_states = np.array([[0,0], [0,1], [1,0], [1,1]])
        states = hlp.get_states(N)
        nptest.assert_array_equal(expected_states, states)

    def test_get_variance_get_std(self):
        mu = 0.2
        expected_variance = mu*(1.-mu)
        expected_std = np.sqrt(expected_variance)
        variance = hlp.get_variance(mu)
        self.assertAlmostEqual(expected_variance, variance)
        std = hlp.get_std(mu)
        self.assertAlmostEqual(expected_std, std)

    def test_get_joints(self):
        N = int(1e5)
        a_s = np.random.randint(0, 2, N).reshape(int(N/2), 2)
        expected_joints = [0.25, 0.25, 0.25, 0.25]
        joints = hlp.get_joints(a_s, 0)
        nptest.assert_array_almost_equal(expected_joints, joints, decimal=2)

    def test_get_marginals(self):
        N = int(1e5)
        a_s = np.random.randint(0, 2, N).reshape(int(N/2), 2)
        expected_marginals = [0.5, 0.5]
        marginals = hlp.get_marginals(a_s, 0)
        nptest.assert_array_almost_equal(expected_marginals, marginals, decimal=2)

    def test_DKL(self):
        p = np.array([0.1, 0.3, 0.2, 0.4])
        q = np.array([0.2, 0.3, 0.1, 0.4])
        expected_DKL = np.sum([p[i]*np.log(p[i]/q[i]) for i in range(len(p))])
        DKL = hlp.get_DKL(p, q)
        nptest.assert_array_almost_equal(expected_DKL, DKL)

    def test_theta(self):
        x = np.array([1., -.1, -1., .1])
        expected_y = np.array([1., 0., 0., 1.])
        y = hlp.theta(x)
        nptest.assert_array_equal(expected_y, y)
        x = np.array([1., 0., -1., .1])
        self.assertRaises(ValueError, hlp.theta, x)

    def test_sigmoidal(self):
        x = np.random.rand(int(1e2))
        expected_y = 1./(1. + np.exp(-x))
        y = hlp.sigma(x)
        nptest.assert_array_almost_equal(expected_y, y)

    def test_sigmainv(self):
        expected_x = np.random.rand(int(1e2))
        y = hlp.sigma(expected_x)
        x = hlp.sigmainv(y)
        nptest.assert_array_almost_equal(expected_x, x)

    def test_mun_sigman(self):
        K = 20
        gamma = 0.8
        g = 4.
        w = 0.2
        smu = 0.2
        steps = int(1e5)
        KE = int(gamma*K)
        KI = K-KE
        sigmas = hlp.get_std(smu)
        xE = w*np.random.normal(0, sigmas, (steps, KE))
        xI = -g*w*np.random.normal(0, sigmas, (steps, KI))
        x = np.sum([np.sum(xE, axis=1), np.sum(xI, axis=1)], axis=0)
        expected_mu = np.mean(x)
        expected_std = np.std(x)
        mu = hlp.get_mun(K, gamma, g, w, smu)
        self.assertAlmostEqual(expected_mu, mu, places=2)
        std = hlp.get_sigman(K, gamma, g, w, sigmas)
        self.assertAlmostEqual(expected_std, std, places=2)

    def test_weight_noise(self):
        K = 50
        gamma = 0.8
        g = 6
        smu = 0.4
        beta = .7
        steps = int(2e5)
        sigmas = hlp.get_std(smu)
        expected_std = np.sqrt(8./(np.pi*beta**2))
        w = hlp.get_weight_noise(beta, sigmas, K, gamma, g)
        KE = int(gamma*K)
        KI = K-KE
        xE = w*np.random.normal(0, sigmas, (steps, KE))
        xI = -g*w*np.random.normal(0, sigmas, (steps, KI))
        x = np.sum([np.sum(xE, axis=1), np.sum(xI, axis=1)], axis=0)
        std = np.std(x)
        self.assertAlmostEqual(expected_std, std, places=2)

    def test_Fsigma(self):
        samples = int(5e4)
        x = np.random.rand(samples)
        expected_y = 1./(1.+ np.exp(-0.5))
        y = []
        for xi in x:
            y.append(hlp.Fsigma(xi))
        y = np.mean(y)
        nptest.assert_array_almost_equal(expected_y, y, decimal=2)


class NetworkTestCase(unittest.TestCase):

    def test_unconnected_mean_variance(self):
        N = 100
        W = np.zeros((N,N))
        b = np.ones(N)*0.2
        sinit = np.random.randint(0, 2, N)
        Nrec = 20
        steps = 2e5
        expected_mean = 1./(1.+np.exp(-b[0]))
        expected_variance = hlp.get_variance(expected_mean)
        def F(x):
            return 0 if 1./(1+np.exp(-x)) < np.random.rand() else 1
        a_states, a_s = bnet.simulate(W, b, sinit, steps, Nrec, [N], [F])
        mean = np.mean(a_s)
        variance = np.var(a_s)
        self.assertAlmostEqual(expected_mean, mean, places=1)
        self.assertAlmostEqual(expected_variance, variance, places=1)

    def test_multiple_activation_functions(self):
        N = 100
        W = np.zeros((N,N))
        N1 = 15
        b = np.ones(N)*0.2
        b[N1:] = 0.9
        sinit = np.random.randint(0, 2, N)
        Nrec = 20
        steps = 2e5
        def F1(x):
            return 0 if 1./(1+np.exp(-x)) < np.random.rand() else 1
        def F2(x):
            return 0 if 1./(1+np.exp(-x+0.7)) < np.random.rand() else 1
        a_states, a_s = bnet.simulate(W, b, sinit, steps, Nrec, [N1,N], [F1,F2])
        a_means = np.mean(a_s, axis=0)
        expected_means = np.ones(Nrec)*1./(1.+np.exp(-b[0]))
        nptest.assert_array_almost_equal(expected_means, a_means, decimal=1)

    def test_joint_distribution(self):
        N = 2
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        sinit = np.random.randint(0, 2, N)
        Nrec = 2
        steps = 1e5
        def F(x):
            return 0 if 1./(1+np.exp(-x)) < np.random.rand() else 1
        a_states, a_s = bnet.simulate(W, b, sinit, steps, Nrec, [N], [F])
        joints = hlp.get_joints(a_s, 0)
        expected_joints = hlp.get_theo_joints(W,b)
        nptest.assert_array_almost_equal(expected_joints, joints, decimal=1)

    def test_marginal_distribution(self):
        N = 2
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        sinit = np.random.randint(0, 2, N)
        Nrec = 2
        steps = 2e5
        def F(x):
            return 0 if 1./(1+np.exp(-x)) < np.random.rand() else 1
        a_states, a_s = bnet.simulate(W, b, sinit, steps, Nrec, [N], [F])
        marginals = hlp.get_marginals(a_s, 0)
        expected_marginals = hlp.get_theo_marginals(W,b)
        nptest.assert_array_almost_equal(expected_marginals, marginals, decimal=2)


if __name__ == '__main__':
    unittest.main()
