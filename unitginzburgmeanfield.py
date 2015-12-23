# global imports
import numpy as np

"""
ref Helias14:
Helias, Tetzlaff, Diesmann (2014) The Correlation Structure of
Local Neuronal Networks Intrinsically Results from Recurrent Dynamics
PLoS Comput Biol 10(1): e1003428
DOI: 10.1371/journal.pcbi.1003428
"""


class BinaryMeanfield(object):
    """this module allows one to calculate the stationary firing rates and
    pairwise correlations in a network of binary neurons with
    sigmoidal activation function from the connectivity matrix and
    bias vector.

    """

    def __init__(self, J, b, beta):
        """
        J: connectivity matrix
        b: bias vector
        beta: inverse temperature (controls slope of gain function)
        """
        self.J = J
        self.b = b
        self.N = len(b)
        self.beta = beta
        self.mu = np.zeros(self.N)

    def get_mu_meanfield(self, mu, C):
        """
        rates from rates and covariances
        similar to Formula (7) in Helias14
        mu: rate vector
        C: covariance matrix
        """
        # calculate input statistics from rates and covariances
        h_mu = self.get_mu_input(mu)
        h_mu += self.b
        h_sigma2 = self.get_sigma2_input(C)
        
        # Taylor expansion
        # of activation function under the integral after substitution
        eb = np.exp(self.beta * h_mu)
        C0 = eb / (eb + 1.)
        C0 = 1. / (1. + np.exp(-1. * self.beta * h_mu))
        C2 = (eb ** 2 - eb) / (2. * (eb + 1.) ** 3) * self.beta ** 2
        C4 = (eb ** 4 - 11. * eb ** 3 + 11. * eb ** 2 - eb) \
            / (24. * (eb + 1.) ** 5) * self.beta ** 4
        mu = C0 - C2 * h_sigma2 - C4 * 3. * h_sigma2 ** 2

        # Numerical integration
        # mu = np.empty(self.N)
        # for i in xrange(self.N):
        #     def f(x):
        #         return 1./(1. + np.exp(-self.beta * x)) * 1./np.sqrt(2. * np.pi * h_sigma2[i]) * np.exp(-(x - h_mu[i])**2 / (2. * h_sigma2[i]))
        #     mu[i], error = scint.quad(f, -50., 50.)
        #     assert(error < 1e-7), 'Integration error while determining mean activity.'

        # Sommerfeld expansion
        # WARNING: seems to fail due to size of h_sigma2
        # mu[i] = 0.5 * (1. + scsp.erf(h_mu[i]/np.sqrt(h_sigma2[i]))) \
        #         - 1./self.beta**2 * 1./np.sqrt(np.pi * h_sigma2[i]) * np.pi**2 / 6. * 2.* h_mu[i]/h_sigma2[i] * np.exp(-h_mu[i]**2 / h_sigma2[i])
        return mu

    def get_mu_input(self, mu):
        """
        Mean input given presynaptic activity mu
        mu: rate vector
        """
        return np.dot(self.J, mu)

    def get_sigma2_input(self, C):
        """
        Standard deviation of input given presynaptic activity mu
        C: covariance matrix
        """
        assert(np.all(C.diagonal() >= 0.))
        sigma2_input = (np.dot(np.dot(self.J, C), self.J.T)).diagonal()
        assert(np.all(sigma2_input >= 0.))
        return sigma2_input

    def get_suszeptibility(self, mu, C):
        """
        Suszeptibility (i.e., derivative of Gain function) from rates and covariances
        see Formula (8) in Helias14
        mu: rate vector
        C: covariance matrix
        """
        h_mu = self.get_mu_input(mu)
        h_mu += self.b
        h_sigma2 = self.get_sigma2_input(C)

        # Taylor expansion
        # of derivative of activation function under the integral after substitution
        eb = np.exp(self.beta * h_mu)
        C0 = eb / (eb + 1) ** 2
        C2 = (eb ** 3 - 4. * eb ** 2 + eb) / \
            (2. * (eb + 1.) ** 4) * self.beta ** 2
        C4 = (eb ** 5 - 26 * eb ** 4 + 66 * eb ** 3 - 26 * eb ** 2 + eb) \
            / (24. * (eb + 1) ** 6) * self.beta ** 4
        S = self.beta * (C0 + C2 * h_sigma2 + C4 * 3. * h_sigma2 ** 2)

        # Numerical integration
        # S = np.empty(self.N)
        # for i in xrange(self.N):
        #     def f(x):
        #         return self.beta / (np.exp(self.beta * x) + np.exp(-self.beta * x) + 2) * 1./np.sqrt(2. * np.pi * h_sigma2[i]) * np.exp(-(x - h_mu[i])**2 / (2. * h_sigma2[i]))
        #     S[i], error = scint.quad(f, -50., 50.)
        #     assert(error < 1e-7), 'Integration error while determining suszeptibility.'
        return S

    def get_w_meanfield(self, mu, C):
        """
        Linearized weights
        see Formula (10) in Helias14
        mu: rate vector
        C: covariance matrix
        """
        S = np.diag(self.get_suszeptibility(mu, C))
        return np.dot(S, self.J)

    def get_m_corr_iter(self, mu0, lamb, C0=None):
        """Calculate correlations iteratively from rate vector
        mu0: initial guess for rate vector
        lamb: slowness parameter (controls convergence speed)
        C: initial guess for covariance matrix
        """
        Dmu = 1e10
        Dc = 1e10
        if C0 is None:
            C0 = np.zeros((self.N, self.N))
        mu = mu0.copy()
        C = C0.copy()
        for i, m_i in enumerate(mu):
            C[i, i] = m_i * (1. - m_i)
        while Dmu > 1e-9 or Dc > 1e-9:
            mu_new = self.get_mu_meanfield(mu, C)
            Dmu = np.max(abs(mu - mu_new))
            mu = (1. - lamb) * mu + lamb * mu_new

            W = self.get_w_meanfield(mu, C)
            WC = np.dot(W, C)
            C_new = 0.5 * WC + 0.5 * WC.T
            for i, m_i in enumerate(mu):
                C_new[i, i] = m_i * (1. - m_i)
            Dc = np.max(abs(C - C_new))
            C = (1. - lamb) * C + lamb * C_new
        return mu, C
