# global imports
import numpy as np
import scipy.special as scsp

"""
ref Helias14:
Helias, Tetzlaff, Diesmann (2014) The Correlation Structure of
Local Neuronal Networks Intrinsically Results from Recurrent Dynamics
PLoS Comput Biol 10(1): e1003428
DOI: 10.1371/journal.pcbi.1003428
"""


class BinaryMeanfield(object):
    """this module allows one to calculate the stationary firing rates
    and pairwise correlations in a network of binary neurons with
    Heaviside activation function from the connectivity matrix and
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
        h_sigma2 = self.get_sigma2_input(mu, C)
        return 0.5 * scsp.erfc(-1. * (h_mu + self.b) / (np.sqrt(2. * h_sigma2)))

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
        return 1. / np.sqrt(2. * np.pi * h_sigma2) * np.exp(-1. * h_mu ** 2 / (2. * h_sigma2))

    def get_w_meanfield(self, mu, C):
        """
        Linearized weights
        see Formula (10) in Helias14
        mu: rate vector
        C: covariance matrix
        """
        S = np.diag(self.get_suszeptibility(mu, C))
        return np.dot(S, self.J)

    def get_corr_iter(self, mu, lamb, C0=None):
        """Calculate correlations iteratively from mean rates
        mu: rate vector
        lamb: slowness parameter (controls convergence speed)
        C0: initial guess for covariance matrix
        """
        Dc = 1e10
        if C0 is None:
            C0 = np.zeros((self.N, self.N))
        mu = mu.copy()
        C = C0.copy()
        for i, m_i in enumerate(mu):
            C[i, i] = m_i * (1. - m_i)
        S = np.diag(self.get_suszeptibility(mu, C))
        W = np.dot(S, self.J)
        while Dc > 1e-12:
            WC = np.dot(W, C)
            C_new = 0.5 * WC + 0.5 * WC.T
            for i, m_i in enumerate(mu):
                C_new[i, i] = m_i * (1. - m_i)
            Dc = np.max(abs(C - C_new))
            C = (1. - lamb) * C + lamb * C_new
        return C

    def get_m_corr_iter(self, mu0, lamb, C=None):
        """Calculate rates and correlations iteratively
        mu0: initial guess for rate vector
        lamb: slowness parameter (controls convergence speed)
        C: initial guess for covariance matrix
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
            C_new = 0.5 * WC + 0.5 * WC.T
            for i, m_i in enumerate(mu):
                C_new[i, i] = m_i * (1. - m_i)
            Dc = np.max(abs(C - C_new))
            C = (1. - lamb) * C + lamb * C_new
        return mu, C

    def get_m(self, mu0, lamb, C=None):
        """Calculate rates iteratively
        mu0: initial guess for rate vector
        lamb: slowness parameter (controls convergence speed)
        C: covariance matrix
        """
        Dmu = 1e10
        if C is None:
            C = np.zeros((self.N, self.N))
        mu = mu0.copy()
        C = C.copy()
        while Dmu > 1e-7:
            for i, m_i in enumerate(mu):
                C[i, i] = m_i * (1. - m_i)
            mu_new = self.get_mu_meanfield(mu, C)
            Dmu = np.max(abs(mu - mu_new))
            mu = (1. - lamb) * mu + lamb * mu_new
        return mu

    def get_corr_eigen(self, mu, C0=None):
        """Calculate correlations iteratively from mean rates using linearized
        approximation.
        mu: rate vector
        C0: initial guess for covariance matrix
        """

        if C0 is None:
            C0 = np.zeros((self.N, self.N))
        C = C0.copy()
        # # autocorr matrix
        # A = np.diag(mu * (1. - mu))
        for i, m_i in enumerate(mu):
            C[i, i] = m_i * (1. - m_i)

        # suszeptibility
        S = np.diag(self.get_suszeptibility(mu, C))

        # linearized coupling matrix: multiply each row with repective
        # susceptibility
        W = np.dot(S, self.J)
        M = np.eye(self.N) - W

        # diagonalize the effective coupling matrix M = 1 - W
        # determine left v and right u eigenvectors
        lmbd, U = np.linalg.eig(M)
        V = np.linalg.inv(U)

        # check whether inversion makes sense
        diag_check = np.dot(V, np.dot(M, U))
        diag_check -= np.diag(diag_check.diagonal())
        assert(abs(np.mean(diag_check)) <
               1e-15), 'Error inverting effective coupling matrix M = (1 - W)'

        # calculate covariance matrix (see e.g. tn_corr script eq. 5.6.7)
        A = np.diag(mu * (1. - mu))
        Ap = np.dot(V, np.dot(A, V.T))
        C = np.zeros((self.N, self.N), dtype=float)
        for i in xrange(0, self.N):
            for j in xrange(0, self.N):
                C += 2. * Ap[i, j] / (lmbd[i] + lmbd[j]) * \
                    np.outer(U.T[i], U.T[j])

        # set autocorr
        for i, m_i in enumerate(mu):
            C[i, i] = m_i * (1. - m_i)

        return C
