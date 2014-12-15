import numpy as np
import scipy.special as scsp
import scipy.optimize as scop
import scipy.integrate as scint

"""""""""""
DISCLAIMER: SEVERELY OUTDATED DOCSTRINGS
"""""""""""

"""
ref Helias14:
Helias, Tetzlaff, Diesmann (2014) The Correlation Structure of
Local Neuronal Networks Intrinsically Results from Recurrent Dynamics
PLoS Comput Biol 10(1): e1003428
DOI: 10.1371/journal.pcbi.1003428
"""

class BinaryMeanfield(object):
    """
    this module can calculate the stationary firing rate and mean
    correlations in a network of binary neurons with an excitatory
    and inhibitory population, from connectivity statistics
    b is the bias vector (2d, corresponding to -1*threshold)
    """

    def __init__(self, J, b, beta):
        self.J = J
        self.b = b
        self.N = len(b)
        self.beta = beta
        self.mu = np.zeros(self.N)


    def get_mu_meanfield(self, mu, C):
        """
        Self-consistent rate
        Formula (7) in Helias14
        """
        h_mu = self.get_mu_input(mu)
        h_mu += self.b
        h_sigma2 = self.get_sigma2_input(mu, C)
        mu = np.empty(self.N)
        for i in xrange(self.N):
            def f(x):
                return 1./(1. + np.exp(-self.beta * x)) * 1./np.sqrt(2. * np.pi * h_sigma2[i]) * np.exp(-(x - h_mu[i])**2 / (2. * h_sigma2[i]))
            mu[i], error = scint.quad(f, -50., 50.)
            assert(error < 1e-7), 'Integration error while determining mean activity.'
        return mu


    def get_mu_input(self, mu):
        """
        Mean input given presynaptic activity mu
        Formula (4) in Helias14
        """
        return np.dot(self.J, mu)


    def get_sigma2_input(self, mu, C):
        """
        Standard deviation of input given presynaptic activity mu
        (and correlations C)
        For C=None: formula (6) in Helias14
        For C: formula (13) in Helias14
        """
        assert(np.all(C.diagonal() >= 0.))
        sigma2_input = (np.dot(np.dot(self.J, C), self.J.T)).diagonal()
        return sigma2_input


    def get_suszeptibility(self, mu, C):
        """
        Suszeptibility (i.e., derivative of Gain function) for Gaussian
        input with mean mu and standard deviation sigma
        Formula (8) in Helias14
        """
        h_mu = self.get_mu_input(mu)
        h_sigma2 = self.get_sigma2_input(mu, C)
        S = np.empty(self.N)
        for i in xrange(self.N):
            def f(x):
                return 1. / (np.exp(self.beta * x) + np.exp(-self.beta * x) + 2) * 1./np.sqrt(2. * np.pi * h_sigma2[i]) * np.exp(-(x - h_mu[i] - self.b[i])**2 / (2. * h_sigma2[i]))
            S[i], error = scint.quad(f, -50., 50.)
            assert(error < 1e-7), 'Integration error while determining suszeptibility.'
        return S


    def get_w_meanfield(self, mu, C):
        """
        Linearized population averaged weights
        Formula (10) in Helias14
        """
        S = np.diag(self.get_suszeptibility(mu, C))
        return np.dot(S, self.J)


    def get_m_corr_iter(self, mu0, lamb, C=None):
        """Calculate correlations iteratively from mean rates
        """
        Dmu = 1e10
        Dc = 1e10
        if C is None:
            C = np.zeros((self.N, self.N))
        mu = mu0.copy()
        C = C.copy()
        for i, m_i in enumerate(mu):
            C[i, i] = m_i * (1. - m_i)
        while Dmu > 1e-10 or Dc > 1e-10:
            mu_new = self.get_mu_meanfield(mu, C)
            Dmu = np.max(abs(mu - mu_new))
            mu = (1. - lamb) * mu + lamb * mu_new

            W = self.get_w_meanfield(mu, C)
            WC = np.dot(W, C)
            C_new = 0.5*WC + 0.5*WC.T
            for i, m_i in enumerate(mu):
                C_new[i, i] = m_i * (1. - m_i)
            Dc = np.max(abs(C - C_new))
            C = (1. - lamb) * C + lamb * C_new
        return mu, C
