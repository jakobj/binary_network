import unittest
import numpy as np
import numpy.testing as nptest

import helper as hlp
import network as bnet
import meanfield as bmf

np.random.seed(123456)


class HelperTestCase(unittest.TestCase):

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

    def test_get_energy(self):
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0.2, 0.2])
        s = np.array([1,0])
        expected_energy = -1.*np.sum(0.5*np.dot(s.T, np.dot(W, s)) + np.dot(b,s))
        energy = hlp.get_energy(W, b, s)
        self.assertAlmostEqual(expected_energy, energy)

    def test_get_theo_joints(self):
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        beta = 0.5
        N = len(b)
        expected_joints = []
        states = hlp.get_states(N)
        for s in states:
            expected_joints.append(np.exp(-1.*beta*hlp.get_energy(W, b, s)))
        expected_joints = 1.*np.array(expected_joints)/np.sum(expected_joints)
        joints = hlp.get_theo_joints(W, b, beta)
        nptest.assert_array_almost_equal(expected_joints, joints)

    def test_get_theo_marginals(self):
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        beta = 0.8
        N = len(b)
        expected_marginals = []
        states = hlp.get_states(N)
        Z = 0
        for s in states:
            Z += np.exp(-1.*beta*hlp.get_energy(W, b, s))
        for i in range(2):
            statesi = states[states[:,i] == 1]
            p = 0
            for s in statesi:
                p += np.exp(-1.*beta*hlp.get_energy(W, b, s))
            expected_marginals.append(1./Z*p)
        marginals = hlp.get_theo_marginals(W, b, beta)
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
        variance = hlp.get_sigma2(mu)
        self.assertAlmostEqual(expected_variance, variance)
        std = hlp.get_sigma(mu)
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
        beta = 0.781
        expected_x = np.random.rand(int(1e2))
        y = hlp.sigma(expected_x, beta)
        x = hlp.sigmainv(y, beta)
        nptest.assert_array_almost_equal(expected_x, x)

    def test_mun_sigman(self):
        N = 300
        epsilon = 0.1
        K = epsilon*N
        gamma = 0.8
        g = 6.
        w = 0.2
        smu = 0.2
        steps = int(1e5)
        KE = int(gamma*K)
        KI = K-KE
        sigmas = hlp.get_sigma(smu)
        xE = w*np.random.normal(smu, sigmas, (steps, KE))
        xI = -g*w*np.random.normal(smu, sigmas, (steps, KI))
        x = np.sum([np.sum(xE, axis=1), np.sum(xI, axis=1)], axis=0)
        expected_mu = np.mean(x)
        expected_sigma = np.std(x)
        mu = hlp.get_mu_input(epsilon, N, gamma, g, w, smu)
        self.assertAlmostEqual(expected_mu, mu, delta=0.02*abs(expected_mu))
        sigma = hlp.get_sigma_input(epsilon, N, gamma, g, w, smu)
        self.assertAlmostEqual(expected_sigma, sigma, delta=0.02*abs(expected_sigma))

    def test_Fsigma(self):
        samples = int(5e4)
        x = np.random.rand(samples)
        expected_y = 1./(1.+ np.exp(-0.5))
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
        expected_variance = hlp.get_sigma2(expected_mean)
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
        def F1(x, beta):
            return 0 if 1./(1+np.exp(-beta*x)) < np.random.rand() else 1
        def F2(x, beta):
            return 0 if 1./(1+np.exp(-beta*x+0.7)) < np.random.rand() else 1
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
        beta = 0.8
        sinit = np.random.randint(0, 2, N)
        tau = 10.
        Nrec = 2
        steps = 1e5
        for i,sim in enumerate([bnet.simulate, bnet.simulate_eve]):
            if i == 0:
                a_states, a_s = sim(W, b, sinit, steps, Nrec, [N], [hlp.Fsigma], beta=beta)
            else:
                a_states, a_s = sim(W, b, tau, sinit, steps*tau/N, Nrec, [N], [hlp.Fsigma], beta=beta)
            joints = hlp.get_joints(a_s, 0)
            expected_joints = hlp.get_theo_joints(W, b, beta)
            nptest.assert_array_almost_equal(expected_joints, joints, decimal=1)

    def test_marginal_distribution(self):
        N = 2
        W = np.array([[0., 0.5], [0.5, 0.]])
        b = np.array([0., 0.6])
        beta = 0.7
        sinit = np.random.randint(0, 2, N)
        tau = 10.
        Nrec = 2
        steps = 2e5
        for i,sim in enumerate([bnet.simulate, bnet.simulate_eve]):
            if i == 0:
                a_states, a_s = sim(W, b, sinit, steps, Nrec, [N], [hlp.Fsigma], beta=beta)
            else:
                a_states, a_s = sim(W, b, tau, sinit, steps*tau/N, Nrec, [N], [hlp.Fsigma], beta=beta)
            marginals = hlp.get_marginals(a_s, 0)
            expected_marginals = hlp.get_theo_marginals(W, b, beta)
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
        N = 58
        sinit = np.zeros(N)
        tau = 10.
        Nrec = N
        time = 3.5e3
        mu_target = 0.4
        tbin = .6
        tmax = 600.
        expected_var = mu_target*(1.-mu_target)
        expected_timelag = np.hstack([-1.*np.arange(tbin,tmax+tbin,tbin)[::-1],0,np.arange(tbin,tmax+tbin,tbin)])
        expected_autof = expected_var*np.exp(-1.*abs(expected_timelag)/tau)

        # Network case (correlated sources)
        w = 0.2
        g = 8.
        gamma = 0.
        epsilon = 0.2
        W_brn = hlp.create_connectivity_matrix(N, w, g, epsilon, gamma)
        b_brn = -1.*hlp.get_mu_input(epsilon, N, gamma, g, w, mu_target)*np.ones(N)-1.*w/2
        a_times_brn, a_s_brn = bnet.simulate_eve(W_brn, b_brn, tau, sinit.copy(), time, Nrec, [N], [hlp.theta])
        self.assertAlmostEqual(mu_target, np.mean(a_s_brn), delta=0.1*np.mean(a_s_brn))
        times_bin_brn, st_brn = hlp.bin_binary_data(a_times_brn, a_s_brn, tbin, time)
        timelag_brn, autof_brn = hlp.autocorrf(times_bin_brn, st_brn[:30], tmax)
        nptest.assert_array_almost_equal(expected_timelag, timelag_brn)
        self.assertTrue(abs(np.sum(autof_brn-expected_autof)) < 0.5*np.sum(abs(autof_brn)))

        # Poisson (independent)
        W = np.zeros((N, N))
        b = np.ones(N)*hlp.sigmainv(mu_target)
        a_times, a_s = bnet.simulate_eve(W, b, tau, sinit.copy(), time, Nrec, [N], [hlp.Fsigma])
        self.assertAlmostEqual(mu_target, np.mean(a_s), delta=0.1*np.mean(a_s))
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
        b_brn = -1.*hlp.get_mu_input(epsilon, N, gamma, g, w, mu_target)*np.ones(N)-1.*w/2
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
        Nnoise = 120
        sinit = np.zeros(N+Nnoise)
        tau = 10.
        Nrec = N
        time = 5e3
        mu_target = 0.42

        w = 0.2
        g = 8.
        gamma = 0.
        epsilon = 0.3
        expected_mu_input = hlp.get_mu_input(epsilon, Nnoise, gamma, g, w, mu_target)
        expected_std_input = hlp.get_sigma_input(epsilon, Nnoise, gamma, g, w, mu_target)

        # Network case (correlated sources)
        W_brn = np.zeros((N+Nnoise, N+Nnoise))
        W_brn[:N,N:] = hlp.create_noise_connectivity_matrix(N, Nnoise, gamma, g, w, epsilon)
        W_brn[N:,N:] = hlp.create_connectivity_matrix(Nnoise, w, g, epsilon, gamma)
        b_brn = np.zeros(N+Nnoise)
        b_brn[:N] = -w/2.
        b_brn[N:] = -1.*hlp.get_mu_input(epsilon, Nnoise, gamma, g, w, mu_target)-1.*w/2
        a_times_brn, a_s_brn, a_times_ui_brn, a_ui_brn = bnet.simulate_eve(W_brn, b_brn, tau, sinit.copy(), time, Nrec, [N+Nnoise], [hlp.theta], Nrec_ui=10)
        self.assertTrue( abs(np.mean(a_ui_brn)+w/2. - expected_mu_input) < 0.05*abs(expected_mu_input))
        self.assertTrue( (np.mean(np.std(a_ui_brn, axis=0)) - expected_std_input) < 0)

        # Poisson case (independent sources)
        W = np.zeros((N+Nnoise, N+Nnoise))
        W[:N,N:] = hlp.create_noise_connectivity_matrix(N, Nnoise, gamma, g, w, epsilon)
        b = np.zeros(N+Nnoise)
        b[:N] = -w/2.
        b[N:] = hlp.sigmainv(mu_target)
        a_times, a_s, a_times_ui, a_ui = bnet.simulate_eve(W, b, tau, sinit.copy(), time, Nrec, [N,N+Nnoise], [hlp.theta, hlp.Fsigma], Nrec_ui=10)
        self.assertTrue( abs(np.mean(a_ui)+w/2. - expected_mu_input) < 0.04*abs(expected_mu_input))
        self.assertTrue( abs(np.mean(np.std(a_ui, axis=0)) - expected_std_input)< 0.04*expected_std_input)


class MeanfieldTestCase(unittest.TestCase):

    def setUp(self):
        epsilon = 0.1
        N = 100
        gamma = 0.2
        self.g = 8.
        self.w = 0.35
        self.b = np.array([0.7, 0.9])
        self.NE = int(gamma*N)
        self.NI = N-self.NE
        self.KE = int(epsilon*self.NE)
        self.KI = int(epsilon*self.NI)
        self.mu = np.array([0.6, 0.5])
        self.sigma = np.array([0.35, 0.73])
        self.mfi = bmf.BinaryMeanfield(epsilon, N, gamma, self.g, self.w, self.b)

    def test_get_mu_input(self):
        expected_mu_input = self.KE*self.w*self.mu[0]+self.KI*(-self.g*self.w)*self.mu[1]
        mu_input = self.mfi.get_mu_input(self.mu)
        self.assertAlmostEqual(expected_mu_input, mu_input[0])
        self.assertAlmostEqual(expected_mu_input, mu_input[1])

    def test_get_sigma_input(self):
        CEE = 0.003
        CIE = CEI = 0.1
        CII = -0.003
        sigma_input = self.mfi.get_sigma_input(self.mu)
        expected_sigma_input = np.sqrt(self.KE*self.w**2*self.mu[0]*(1.-self.mu[0])+self.KI*(-self.g*self.w)**2*self.mu[1]*(1.-self.mu[1]))
        self.assertAlmostEqual(expected_sigma_input, sigma_input[0])
        self.assertAlmostEqual(expected_sigma_input, sigma_input[1])
        C = np.array([[CEE, CIE],
                      [CEI, CII]])
        sigma_input = self.mfi.get_sigma_input(self.mu, C)
        expected_sigma_input = np.sqrt(
            self.KE*self.w**2*self.mu[0]*(1.-self.mu[0])+self.KI*(-self.g*self.w)**2*self.mu[1]*(1.-self.mu[1])
            +(self.KE*self.w)**2*CEE+2.*self.KE*self.KI*(-self.g*self.w**2)*CEI+(self.KI*(-self.g*self.w))**2*CII)
        self.assertAlmostEqual(expected_sigma_input, sigma_input[0])
        self.assertAlmostEqual(expected_sigma_input, sigma_input[1])

    def test_get_suszeptibility(self):
        mu_input = self.mfi.get_mu_input(self.mu)
        sigma_input = self.mfi.get_sigma_input(self.mu)
        expected_S0 = 1./(np.sqrt(2.*np.pi)*sigma_input[0])*np.exp(-(mu_input[0]+self.b[0])**2/(2.*sigma_input[0]**2))
        expected_S1 = 1./(np.sqrt(2.*np.pi)*sigma_input[1])*np.exp(-(mu_input[1]+self.b[1])**2/(2.*sigma_input[1]**2))
        S = self.mfi.get_suszeptibility(mu_input, sigma_input)
        self.assertAlmostEqual(expected_S0, S[0])
        self.assertAlmostEqual(expected_S1, S[1])

    def test_get_w_meanfield(self):
        mu_input = self.mfi.get_mu_input(self.mu)
        sigma_input = self.mfi.get_sigma_input(self.mu)
        S = self.mfi.get_suszeptibility(mu_input, sigma_input)
        expected_w00 = self.KE*self.w*S[0]
        expected_w01 = self.KI*(-self.g*self.w)*S[0]
        expected_w10 = self.KE*self.w*S[1]
        expected_w11 = self.KI*(-self.g*self.w)*S[1]
        W = self.mfi.get_w_meanfield(self.mu)
        self.assertAlmostEqual(expected_w00, W[0,0])
        self.assertAlmostEqual(expected_w01, W[0,1])
        self.assertAlmostEqual(expected_w10, W[1,0])
        self.assertAlmostEqual(expected_w11, W[1,1])

    def test_c_meanfield(self):
        epsilon = 0.1
        N = 100.
        gamma = 0.
        g = 8.
        w = 0.35
        b = np.array([0., 0.9])
        mfi = bmf.BinaryMeanfield(epsilon, N, gamma, g, w, b)
        mu = mfi.get_mu_meanfield(np.array([0.5, 0.5]))
        wII = mfi.get_w_meanfield(mu)[1,1]
        AI = hlp.get_sigma2(mu)[1]/N
        expected_CII = wII/(1.-wII)*AI
        C = mfi.get_c_meanfield(mu)
        self.assertAlmostEqual(expected_CII, C[1, 1])

    def test_comp_network_meanfield(self):
        N = 10
        Nnoise = 500
        T = 1e4
        w = 0.1
        g = 8.
        epsilon = 0.3
        gamma = 0.2
        mu_target = 0.15
        tau = 10.
        Nrec = 50

        W = np.zeros((N+Nnoise, N+Nnoise))
        W[:N,N:] = hlp.create_noise_connectivity_matrix(N, Nnoise, gamma, g, w, epsilon)
        W[N:,N:] = hlp.create_connectivity_matrix(Nnoise, w, g, epsilon, gamma)
        b = np.zeros(N+Nnoise)
        b[:N] = -w/2.
        b[N:] = -1.*hlp.get_mu_input(epsilon, Nnoise, gamma, g, w, mu_target)-w/2.
        sinit = np.array(np.random.randint(0, 2, N+Nnoise), dtype=np.int)

        times, a_s, a_times_ui, a_ui = bnet.simulate_eve(W, b, tau, sinit, T, N+Nrec, [N+Nnoise], [hlp.theta], Nrec_ui=N)
        a_ui = a_ui[200:]
        a_s = a_s[200:]

        # empirical
        mu_noise_activity = np.mean(a_s[:,N:])
        std_noise_activity = np.mean(np.std(a_s[:,N:], axis=0))
        mu_noise = np.mean(a_ui[:,:N])
        std_noise = np.mean(np.std(a_ui[:,:N], axis=0))

        # meanfield
        mfcl = bmf.BinaryMeanfield(epsilon, Nnoise, gamma, g, w, np.array([b[N+1], b[N+1]]))
        # naive
        mu_naive = mfcl.get_m(np.array([0.2,0.2]).T)
        std_naive = hlp.get_sigma(mu_naive)[1]
        mu_naive_input = mfcl.get_mu_input(mu_naive)[1]
        std_naive_input = mfcl.get_sigma_input(mu_naive)[1]
        mu_naive = mu_naive[1]

        # improved (i.e., with correlations)
        mu_iter, c_iter = mfcl.get_m_c_iter(np.array([0.2,0.2]).T)
        std_iter = hlp.get_sigma(mu_iter)[1]
        mu_iter_input = mfcl.get_mu_input(mu_iter)[1]
        std_iter_input = mfcl.get_sigma_input(mu_iter, c_iter)[1]
        mu_iter = mu_iter[1]

        self.assertAlmostEqual(mu_noise_activity, mu_naive, delta=0.1*mu_naive)
        self.assertAlmostEqual(std_noise_activity, std_naive, delta=0.1*std_naive)
        self.assertAlmostEqual(mu_noise, mu_naive_input, delta=abs(0.2*mu_naive_input))
        self.assertAlmostEqual(std_noise, std_naive_input, delta=abs(0.2*std_naive_input))

        self.assertAlmostEqual(mu_noise_activity, mu_iter, delta=0.04*mu_iter)
        self.assertAlmostEqual(std_noise_activity, std_iter, delta=0.04*std_iter)
        self.assertAlmostEqual(mu_noise, mu_iter_input, delta=abs(0.04*mu_iter_input))
        self.assertAlmostEqual(std_noise, std_iter_input, delta=abs(0.04*std_iter_input))


if __name__ == '__main__':
    unittest.main()
