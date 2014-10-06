import unittest
import numpy as np
import numpy.testing as nptest

import helper as hlp
import network as bnet

np.random.seed(123456)

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

    def test_noise_weight_matrix(self):
        Nnoise = 100
        N = 3
        w = 0.2
        g = 6
        epsilon = 0.2
        gamma = 0.8
        W = hlp.create_noise_connectivity_matrix(N, Nnoise, gamma, g, w, epsilon)
        NEnoise = int(gamma*Nnoise)
        NInoise = Nnoise-NEnoise
        for l in W:
            self.assertEqual(len(l[l > 0]), epsilon*NEnoise)
            self.assertAlmostEqual(np.sum(l[l > 0]), epsilon*NEnoise*w)
            self.assertEqual(len(l[l < 0]), epsilon*NInoise)
            self.assertAlmostEqual(np.sum(l[l < 0]), -1.*epsilon*NInoise*w*g)
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
        for yi, xi in zip(expected_y, x):
            self.assertAlmostEqual(yi, hlp.theta(xi))
        self.assertRaises(ValueError, hlp.theta, 0.)

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
        K = 30
        gamma = 0.8
        g = 6.
        w = 0.2
        smu = 0.2
        steps = int(1e5)
        KE = int(gamma*K)
        KI = K-KE
        sigmas = hlp.get_std(smu)
        xE = w*np.random.normal(smu, sigmas, (steps, KE))
        xI = -g*w*np.random.normal(smu, sigmas, (steps, KI))
        x = np.sum([np.sum(xE, axis=1), np.sum(xI, axis=1)], axis=0)
        expected_mu = np.mean(x)
        expected_std = np.std(x)
        mu = hlp.get_mun(K, gamma, g, w, smu)
        self.assertTrue( abs(expected_mu - mu) < 0.05*abs(expected_mu))
        std = hlp.get_sigman(K, gamma, g, w, sigmas)
        self.assertTrue( abs(expected_std - std) < 0.05*expected_std)

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
        xE = w*np.random.normal(smu, sigmas, (steps, KE))
        xI = -g*w*np.random.normal(smu, sigmas, (steps, KI))
        self.assertAlmostEqual(1./w*np.mean(xE), smu, places=3)
        self.assertAlmostEqual(1./(-g*w)*np.mean(xI), smu, places=3)
        x = np.sum([np.sum(xE, axis=1), np.sum(xI, axis=1)], axis=0)
        std = np.std(x)
        self.assertAlmostEqual(expected_std, std, places=1)

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
        tau = 10.
        Nrec = 20
        steps = 2e5
        expected_mean = 1./(1.+np.exp(-b[0]))
        expected_variance = hlp.get_variance(expected_mean)
        for i,sim in enumerate([bnet.simulate, bnet.simulate_eve]):
            if i == 0:
                a_states, a_s = sim(W, b, sinit, steps, Nrec, [N], [hlp.Fsigma])
            else:
                a_states, a_s = sim(W, b, tau, sinit, steps*tau/N, Nrec, [N], [hlp.Fsigma])
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
        tau = 10.
        Nrec = 20
        steps = 2e5
        def F1(x):
            return 0 if 1./(1+np.exp(-x)) < np.random.rand() else 1
        def F2(x):
            return 0 if 1./(1+np.exp(-x+0.7)) < np.random.rand() else 1
        for i,sim in enumerate([bnet.simulate, bnet.simulate_eve]):
            if i == 0:
                a_states, a_s = sim(W, b, sinit, steps, Nrec, [N1,N], [F1,F2])
            else:
                a_states, a_s = sim(W, b, tau, sinit, steps*tau/N, Nrec, [N1,N], [F1,F2])
            a_means = np.mean(a_s, axis=0)
            expected_means = np.ones(Nrec)*1./(1.+np.exp(-b[0]))
            nptest.assert_array_almost_equal(expected_means, a_means, decimal=1)

    def test_joint_distribution(self):
        N = 2
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        sinit = np.random.randint(0, 2, N)
        tau = 10.
        Nrec = 2
        steps = 1e5
        for i,sim in enumerate([bnet.simulate, bnet.simulate_eve]):
            if i == 0:
                a_states, a_s = sim(W, b, sinit, steps, Nrec, [N], [hlp.Fsigma])
            else:
                a_states, a_s = sim(W, b, tau, sinit, steps*tau/N, Nrec, [N], [hlp.Fsigma])
            joints = hlp.get_joints(a_s, 0)
            expected_joints = hlp.get_theo_joints(W,b)
            nptest.assert_array_almost_equal(expected_joints, joints, decimal=1)

    def test_marginal_distribution(self):
        N = 2
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        sinit = np.random.randint(0, 2, N)
        tau = 10.
        Nrec = 2
        steps = 2e5
        for i,sim in enumerate([bnet.simulate, bnet.simulate_eve]):
            if i == 0:
                a_states, a_s = sim(W, b, sinit, steps, Nrec, [N], [hlp.Fsigma])
            else:
                a_states, a_s = sim(W, b, tau, sinit, steps*tau/N, Nrec, [N], [hlp.Fsigma])
            marginals = hlp.get_marginals(a_s, 0)
            expected_marginals = hlp.get_theo_marginals(W,b)
            nptest.assert_array_almost_equal(expected_marginals, marginals, decimal=2)

    def test_bin_binary_data(self):
        N = 2
        tbin = 0.04
        time = 2.
        times = np.array([0., 0.1, 0.35, 0.8, 0.95, 1.68])
        a_s = np.array([[0,0], [1,0], [1,1], [1,0], [0,0], [1,0]])
        expected_times = np.arange(0., time+tbin, tbin)
        expected_bin = np.empty((N,len(expected_times)))
        for i,t in enumerate(expected_times):
            idl = np.where(times <= t)[0]
            expected_bin[0][i] = a_s[idl[-1],0]
            expected_bin[1][i] = a_s[idl[-1],1]
        times_bin, st = hlp.bin_binary_data(times, a_s, tbin, time)
        nptest.assert_array_equal(expected_times, times_bin)
        nptest.assert_array_equal(expected_bin, st)

    def test_auto_corr(self):
        N = 48
        sinit = np.zeros(N)
        tau = 10.
        Nrec = N
        time = 3.5e3
        mu_target = 0.4
        tbin = .6
        tmax = 600.
        expected_mu = np.ones(N)*mu_target
        expected_var = mu_target*(1.-mu_target)
        expected_timelag = np.hstack([-1.*np.arange(tbin,tmax+tbin,tbin)[::-1],0,np.arange(tbin,tmax+tbin,tbin)])
        expected_autof = expected_var*np.exp(-1.*abs(expected_timelag)/tau)

        # Network case (correlated sources)
        w = 0.2
        g = 8.
        gamma = 0.
        epsilon = 0.3
        W_brn = hlp.create_connectivity_matrix(N, w, g, epsilon, gamma)
        b_brn = -1.*hlp.get_mun(epsilon*N, gamma, g, w, mu_target)*np.ones(N)-1.*w/2
        a_times_brn, a_s_brn = bnet.simulate_eve(W_brn, b_brn, tau, sinit.copy(), time, Nrec, [N], [hlp.theta])
        nptest.assert_array_almost_equal(expected_mu, np.mean(a_s_brn, axis=0), decimal=1)
        times_bin_brn, st_brn = hlp.bin_binary_data(a_times_brn, a_s_brn, tbin, time)
        timelag_brn, autof_brn = hlp.autocorrf(times_bin_brn, st_brn[:30], tmax)
        nptest.assert_array_almost_equal(expected_timelag, timelag_brn)
        self.assertTrue(abs(np.sum(autof_brn-expected_autof)) < 0.5*np.sum(abs(autof_brn)))

        # Poisson (independent)
        W = np.zeros((N, N))
        b = np.ones(N)*hlp.sigmainv(mu_target)
        a_times, a_s = bnet.simulate_eve(W, b, tau, sinit.copy(), time, Nrec, [N], [hlp.Fsigma])
        nptest.assert_array_almost_equal(expected_mu, np.mean(a_s, axis=0), decimal=1)
        times_bin, st = hlp.bin_binary_data(a_times, a_s, tbin, time)
        timelag, autof = hlp.autocorrf(times_bin, st[:30], tmax)
        nptest.assert_array_almost_equal(expected_timelag, timelag)
        nptest.assert_array_almost_equal(expected_autof, abs(autof), decimal=2)
        self.assertTrue(abs(np.sum(abs(autof-expected_autof))) < 0.5*np.sum(abs(autof)))

    def test_cross_corr(self):
        N = 50
        sinit = np.zeros(N)
        tau = 10.
        Nrec = N
        time = 3e3
        mu_target = 0.4
        tbin = .8
        tmax = 400.
        expected_var = mu_target*(1.-mu_target)
        expected_timelag = np.hstack([-1.*np.arange(tbin,tmax+tbin,tbin)[::-1],0,np.arange(tbin,tmax+tbin,tbin)])
        expected_autof = expected_var*np.exp(-1.*abs(expected_timelag)/tau)
        expected_cross_brn = -0.003
        expected_cross = 0.
        expected_crossf = np.zeros(len(expected_timelag))

        # Network case (correlated sources)
        w = 0.2
        g = 8.
        gamma = 0.
        epsilon = 0.3
        W_brn = hlp.create_connectivity_matrix(N, w, g, epsilon, gamma)
        b_brn = -1.*hlp.get_mun(epsilon*N, gamma, g, w, mu_target)*np.ones(N)-1.*w/2
        a_times_brn, a_s_brn = bnet.simulate_eve(W_brn, b_brn, tau, sinit.copy(), time, Nrec, [N], [hlp.theta])
        self.assertTrue( abs(np.mean(a_s_brn) - mu_target) < 0.1*mu_target)
        times_bin_brn, st_brn = hlp.bin_binary_data(a_times_brn, a_s_brn, tbin, time)
        timelag_brn, autof_brn, crossf_brn = hlp.crosscorrf(times_bin_brn, st_brn[:30], tmax)
        nptest.assert_array_almost_equal(expected_timelag, timelag_brn)
        self.assertTrue(abs(np.sum(autof_brn-expected_autof)) < 0.5*np.sum(abs(autof_brn)))
        self.assertTrue(expected_cross_brn > crossf_brn[abs(timelag_brn) < 1e-10][0])

        # Poisson case (independent sources)
        W = np.zeros((N, N))
        b = np.ones(N)*hlp.sigmainv(mu_target)
        a_times, a_s = bnet.simulate_eve(W, b, tau, sinit.copy(), time, Nrec, [N], [hlp.Fsigma])
        self.assertTrue( abs(np.mean(a_s) - mu_target) < 0.1*mu_target)
        times_bin, st = hlp.bin_binary_data(a_times, a_s, tbin, time)
        timelag, autof, crossf = hlp.crosscorrf(times_bin, st[:30], tmax)
        nptest.assert_array_almost_equal(expected_timelag, timelag)
        nptest.assert_array_almost_equal(expected_autof, abs(autof), decimal=2)
        nptest.assert_array_almost_equal(expected_crossf, crossf, decimal=2)
        self.assertAlmostEqual(expected_cross, abs(crossf[abs(timelag) < 1e-10][0]), places=2)

    def test_input(self):
        N = 10
        Nnoise = 100
        sinit = np.zeros(N+Nnoise)
        tau = 10.
        Nrec = N
        time = 5e3
        mu_target = 0.42
        std_target = hlp.get_std(mu_target)

        w = 0.2
        g = 8.
        gamma = 0.
        epsilon = 0.3
        expected_mu_input = hlp.get_mun(epsilon*Nnoise, gamma, g, w, mu_target)
        expected_std_input = hlp.get_sigman(epsilon*Nnoise, gamma, g, w, std_target)

        # Network case (correlated sources)
        W_brn = np.zeros((N+Nnoise, N+Nnoise))
        W_brn[:N,N:] = hlp.create_noise_connectivity_matrix(N, Nnoise, gamma, g, w, epsilon)
        W_brn[N:,N:] = hlp.create_connectivity_matrix(Nnoise, w, g, epsilon, gamma)
        b_brn = np.zeros(N+Nnoise)
        b_brn[:N] = -w/2.
        b_brn[N:] = -1.*hlp.get_mun(epsilon*Nnoise, gamma, g, w, mu_target)-1.*w/2
        a_times_brn, a_s_brn, a_ui_brn = bnet.simulate_eve(W_brn, b_brn, tau, sinit.copy(), time, Nrec, [N+Nnoise], [hlp.theta], record_ui=True, Nrec_ui=10)
        self.assertTrue( abs(np.mean(a_ui_brn)+w/2. - expected_mu_input) < 0.05*abs(expected_mu_input))
        self.assertTrue( (np.mean(np.std(a_ui_brn, axis=0)) - expected_std_input) < 0)

        # Poisson case (independent sources)
        W = np.zeros((N+Nnoise, N+Nnoise))
        W[:N,N:] = hlp.create_noise_connectivity_matrix(N, Nnoise, gamma, g, w, epsilon)
        b = np.zeros(N+Nnoise)
        b[:N] = -w/2.
        b[N:] = hlp.sigmainv(mu_target)
        a_times, a_s, a_ui = bnet.simulate_eve(W, b, tau, sinit.copy(), time, Nrec, [N,N+Nnoise], [hlp.theta, hlp.Fsigma], record_ui=True, Nrec_ui=10)
        self.assertTrue( abs(np.mean(a_ui)+w/2. - expected_mu_input) < 0.04*abs(expected_mu_input))
        self.assertTrue( abs(np.mean(np.std(a_ui, axis=0)) - expected_std_input)< 0.04*expected_std_input)

    def test_calibrate(self):
        N = 20
        Nnoise = 120
        tau = 10.
        time = 10e3
        beta = 1.
        mu_target = 0.41
        mu_noise_target = -1.8
        std_noise_target = 8./(np.pi*beta**2)
        tbin = 0.8
        tmax = 500.

        Nact = int(mu_target*(N+Nnoise))
        sinit = np.random.permutation(np.hstack([np.ones(Nact), np.zeros(N+Nnoise-Nact)]))
        w = 0.2
        g = 8.
        gamma = 0.
        epsilon = 0.3

        # Network case (correlated sources)
        w_adj, b_adj = hlp.calibrate_noise(N, Nnoise, epsilon, gamma, g, w, tau, time, mu_target, mu_noise_target, std_noise_target)

        W_brn = np.empty((N+Nnoise, N+Nnoise))
        W_brn[:N,N:] = hlp.create_noise_connectivity_matrix(N, Nnoise, gamma, g, w_adj, epsilon)
        W_brn[N:,N:] = hlp.create_connectivity_matrix(Nnoise, w, g, epsilon, gamma)
        b_brn = np.zeros(N+Nnoise)
        b_brn[:N] = -w_adj/2.+b_adj
        b_brn[N:] = -1.*hlp.get_mun(epsilon*Nnoise, gamma, g, w, mu_target)-1.*w/2.
        a_times_brn, a_s_brn, a_ui_brn = bnet.simulate_eve(W_brn, b_brn, tau, sinit.copy(), time, 0, [N+Nnoise], [hlp.theta], record_ui=True, Nrec_ui=N)
        self.assertTrue( abs(np.mean(a_ui_brn)+w/2. - mu_noise_target) < 0.1*abs(mu_noise_target) )
        self.assertTrue( abs(np.mean(np.std(a_ui_brn, axis=0)) - std_noise_target)< 0.1*std_noise_target )

        times_u_brn, u_brn = hlp.bin_binary_data(a_times_brn, a_ui_brn, tbin, time)
        timelag_brn, autof_brn, crossf_brn = hlp.crosscorrf(times_u_brn, u_brn, tmax)
        self.assertTrue( abs(np.max(autof_brn) - std_noise_target**2) < 0.1*std_noise_target**2)
        self.assertTrue( abs(np.max(autof_brn) - np.mean(np.std(a_ui_brn, axis=0))**2) < 0.01*np.mean(np.std(a_ui_brn, axis=0))**2)
        self.assertTrue( np.max(crossf_brn)/np.max(autof_brn) < 0.2*epsilon)

        # Poisson case (indendendent sources)
        w_adj, b_adj = hlp.calibrate_poisson_noise(N, Nnoise, epsilon, gamma, g, w, tau, time, mu_target, mu_noise_target, std_noise_target)

        W = np.empty((N+Nnoise, N+Nnoise))
        W[:N,N:] = hlp.create_noise_connectivity_matrix(N, Nnoise, gamma, g, w_adj, epsilon)
        b = np.zeros(N+Nnoise)
        b[:N] = -w_adj/2.+b_adj
        b[N:] = hlp.sigmainv(mu_target)
        a_times, a_s, a_ui = bnet.simulate_eve(W, b, tau, sinit.copy(), time, 0, [N, N+Nnoise], [hlp.theta, hlp.Fsigma], record_ui=True, Nrec_ui=N)
        self.assertTrue( abs(np.mean(a_ui)+w/2. - mu_noise_target) < 0.05*abs(mu_noise_target) )
        self.assertTrue( abs(np.mean(np.std(a_ui, axis=0)) - std_noise_target)< 0.05*std_noise_target )

        times_u, u = hlp.bin_binary_data(a_times, a_ui, tbin, time)
        timelag, autof, crossf = hlp.crosscorrf(times_u, u, tmax)
        self.assertTrue( abs(np.max(autof) - std_noise_target**2) < 0.05*std_noise_target**2)
        self.assertTrue( abs(np.max(crossf)/np.max(autof) - epsilon) < 0.05*epsilon)


if __name__ == '__main__':
    unittest.main()
