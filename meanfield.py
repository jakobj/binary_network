# global imports
import numpy as np
import scipy.special as scsp
import scipy.optimize as scop

# local imports
import helper as bhlp

"""
ref Helias14:
Helias, Tetzlaff, Diesmann (2014) The Correlation Structure of
Local Neuronal Networks Intrinsically Results from Recurrent Dynamics
PLoS Comput Biol 10(1): e1003428
DOI: 10.1371/journal.pcbi.1003428
"""


class BinaryMeanfield(object):
    """
    this module allows one to calculate the stationary firing rate and
    average correlations in a network of binary neurons with one
    excitatory and one inhibitory population from connectivity
    statistics
    """

    def __init__(self, epsilon, N, gamma, g, w, b, K=None):
        """
        epsilon: connectivity
        N: total number of neurons
        gamma: relative size of the excitatory population
        g: relative weight of inhibitory connections
        w: weight of excitatory connections
        b: biases (2d vector), corresponds to -1. * threshold
        K: indegree, can be provided as alternative to connectivity
        """
        if epsilon is not None:
            assert(K is None), 'Please provide connectivity OR indegree.'
        elif epsilon is None:
            assert(K is not None), 'Please provide connectivity OR indegree.'
        self.NE = int(gamma * N)
        self.NI = int(N - self.NE)
        if epsilon is not None:
            KE = int(epsilon * self.NE)
            KI = int(epsilon * self.NI)
        else:
            KE = int(gamma * K)
            KI = int(K - KE)
        self.K = np.array([[KE, KI],
                           [KE, KI]])
        self.J = np.array([[w, -g * w],
                           [w, -g * w]])
        self.b = np.array(b)
        self.C = np.array([[0., 0.],
                           [0., 0.]])
        self.mu = np.array([0., 0.])

    def get_mu_meanfield(self, mu0, C=None):
        """
        Self-consistent rate
        Formula (7) in Helias14
        mu0: average rates
        C: average correlations
        """
        if C is None:
            C = np.array([[0., 0.],
                          [0., 0.]])

        def f(mu):
            h_mu = self.get_mu_input(mu)
            h_sigma = self.get_sigma_input(mu, C)
            return mu - 0.5 * scsp.erfc((-self.b - h_mu) / (np.sqrt(2.) * h_sigma))
        return scop.fsolve(f, mu0)

    def get_mu_input(self, mu):
        """
        Mean input given presynaptic activity mu
        Formula (4) in Helias14
        mu: average rates
        """
        mu = np.array(mu)
        if np.shape(mu) != (2,):
            raise ValueError(
                'Mean activity needs to be given for both populations.')
        return np.dot(self.K * self.J, mu)

    def get_sigma_input(self, mu, C=None):
        """
        Standard deviation of input given presynaptic activity mu
        (and correlations C)
        For C=None: formula (6) in Helias14
        For C given: formula (13) in Helias14
        mu: averages rates
        C: average correlations
        """
        mu = np.array(mu)
        if np.shape(mu) != (2,):
            raise ValueError(
                'Mean activity needs to be given for both populations.')
        if C is None:
            C = np.array([[0., 0.],
                          [0., 0.]])
        else:
            C = np.array(C)
        if np.shape(C) != (2, 2):
            raise ValueError(
                'Correlation needs to be given for all combinations of both populations.')
        a = bhlp.get_sigma2(mu)
        sigma_shared = np.dot(self.K * self.J * self.J, a)
        sigma_corr = np.diag(
            np.dot(np.dot(self.K * self.J, C), (self.K * self.J).T))
        return np.sqrt(sigma_shared + sigma_corr)

    def get_suszeptibility(self, mu, sigma):
        """
        Suszeptibility (i.e., derivative of Gain function) for Gaussian
        input distribution
        Formula (8) in Helias14
        mu: mean of input
        sigma: std of input
        """
        return 1. / (np.sqrt(2. * np.pi) * sigma) * np.exp(-1. * (mu + self.b) ** 2 / (2. * sigma ** 2))

    def get_w_meanfield(self, mu, C=None):
        """
        Linearized population averaged weights
        Formula (10) in Helias14
        mu: average rates
        """
        h_mu = self.get_mu_input(mu)
        h_sigma = self.get_sigma_input(mu, C)
        return ((self.K * self.J).T * self.get_suszeptibility(h_mu, h_sigma)).T

    def get_c_meanfield(self, mu, C=None):
        """
        Self-consistent correlations
        Formula (24) without external input in Helias14
        mu: average rates
        """
        a = bhlp.get_sigma2(mu)
        A = np.zeros(2)
        A[0] = a[0] * 1. / self.NE if self.NE > 0 else 0.
        A[1] = a[1] * 1. / self.NI if self.NI > 0 else 0.
        W = self.get_w_meanfield(mu, C)
        M = np.array([[2. - 2. * W[0, 0], -2. * W[0, 1], 0.],
                      [-1. * W[1, 0], 2. - (W[0, 0] + W[1, 1]), -1. * W[0, 1]],
                      [0, -2. * W[1, 0], 2. - 2. * W[1, 1]]])
        B = np.array([[2. * W[0, 0], 0],
                      [W[1, 0], W[0, 1]],
                      [0, 2. * W[1, 1]]])
        rhs = np.dot(B, A)
        c = np.linalg.solve(M, rhs)
        C = np.array([[c[0], c[1]],
                      [c[1], c[2]]])
        return C

    def get_m_c_iter(self, mu0):
        """Calculate mean activity and mean correlations in a recurrent
        network iteratively, using the improved meanfield approach from
        Helias14
        mu0: initial guess for average rates
        """
        if np.shape(mu0) != (2,):
            raise ValueError('Initial guess for mean activity needs to be given for both populations.')
        Dmu = 1e10
        Dc = 1e10
        mu = mu0
        C = self.C
        while Dmu > 1e-15 and Dc > 1e-15:
            mu_old = np.sum(mu)
            c_old = np.sum(C)
            mu = self.get_mu_meanfield(mu, C)
            C = self.get_c_meanfield(mu, C)
            Dmu = abs(np.sum(mu) - mu_old)
            Dc = abs(np.sum(C) - c_old)
        self.mu = mu
        self.C = C
        return mu, C

    def get_m(self, mu0):
        """Calculate mean activity in a recurrent
        network using meanfield approach
        mu0: initial guess for average rates
        """
        if np.shape(mu0) != (2,):
            raise ValueError('Initial guess for mean activity needs to be given for both populations.')
        mu = mu0
        mu = self.get_mu_meanfield(mu)
        return mu
