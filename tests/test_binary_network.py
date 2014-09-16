import unittest
import numpy as np
import numpy.testing as nptest

import helper as hlp
import binary_network as bnetwork

class HelperTestCase(unittest.TestCase):

    # def setUp(self):
    # def tearDown(self):
    
    def test_BM_weight_matrix(self):
        N = 10
        expected_diag = np.zeros(N)
        W = hlp.create_BM_weight_matrix(N)
        nptest.assert_array_equal(expected_diag, W.diagonal())
        self.assertEqual(0., np.sum(W-W.T))

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

    def test_get_variance(self):
        mu = 0.2
        expected_variance = mu*(1.-mu)
        variance = hlp.get_variance(mu)
        self.assertAlmostEqual(expected_variance, variance)

    def test_get_joints(self):
        N = 1e5
        a_s = np.random.randint(0, 2, N).reshape(int(N/2), 2)
        expected_joints = [0.25, 0.25, 0.25, 0.25]
        joints = hlp.get_joints(a_s, 0)
        nptest.assert_array_almost_equal(expected_joints, joints, decimal=2)

    def test_get_marginals(self):
        N = 1e5
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
        a_states, a_s = bnetwork.simulate(W, b, sinit, steps, Nrec, [N], [F])
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
        a_states, a_s = bnetwork.simulate(W, b, sinit, steps, Nrec, [N1,N], [F1,F2])
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
        a_states, a_s = bnetwork.simulate(W, b, sinit, steps, Nrec, [N], [F])
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
        a_states, a_s = bnetwork.simulate(W, b, sinit, steps, Nrec, [N], [F])
        marginals = hlp.get_marginals(a_s, 0)
        expected_marginals = hlp.get_theo_marginals(W,b)
        nptest.assert_array_almost_equal(expected_marginals, marginals, decimal=2)


if __name__ == '__main__':
    unittest.main()
