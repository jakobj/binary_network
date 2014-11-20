import numpy as np
import scipy.special as scsp
import scipy.optimize as scop

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

    def __init__(self, W, b):
        self.J = W
        self.b = b
        self.N = len(b)
        self.mu = np.zeros(self.N)


    def get_mu_meanfield(self, mu0, C):
        """
        Self-consistent rate
        Formula (7) in Helias14
        """
        h_mu = self.get_mu_input(mu0)
        h_sigma2 = self.get_sigma2_input(mu0, C)
        return 0.5*scsp.erfc((-self.b-h_mu)/(np.sqrt(2.*h_sigma2)))


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


    def get_suszeptibility(self, mu, sigma2):
        """
        Suszeptibility (i.e., derivative of Gain function) for Gaussian
        input with mean mu and standard deviation sigma
        Formula (8) in Helias14
        """
        return 1./np.sqrt(2.*np.pi*sigma2) * np.exp(-1.*(mu + self.b)**2 / (2.*sigma2))


    def get_w_meanfield(self, mu, C):
        """
        Linearized population averaged weights
        Formula (10) in Helias14
        """
        h_mu = self.get_mu_input(mu)
        h_sigma2 = self.get_sigma2_input(mu, C)
        return ((self.J).T*self.get_suszeptibility(h_mu, h_sigma2)).T


    def get_corr_iter(self, mu, lamb):
        """Calculate correlations iteratively from mean rates
        """
        Dc = 1e10
        C = np.zeros((self.N, self.N))
        for i, m_i in enumerate(mu):
            C[i, i] = m_i * (1. - m_i)
        h_mu = self.get_mu_input(mu)
        h_sigma2 = self.get_sigma2_input(mu, C)
        S = np.diag(self.get_suszeptibility(h_mu, h_sigma2))
        while Dc > 1e-12:
            W = np.dot(S, self.J)
            WC = np.dot(W, C)
            C_new = 0.5*WC + 0.5*WC.T
            for i, m_i in enumerate(mu):
                C_new[i, i] = m_i * (1. - m_i)
            Dc = np.max(abs(C - C_new))
            C = (1. - lamb) * C + lamb * C_new

        return C


    def get_m(self, mu0, lamb, C=None):
        """Calculate mean activity in a recurrent
        network using meanfield approach
        """
        Dmu = 1e10
        mu = mu0
        if C is None:
            C = np.zeros((self.N, self.N))
        while Dmu > 1e-7:
            for i, m_i in enumerate(mu):
                C[i, i] = m_i * (1. - m_i)
            mu_new = self.get_mu_meanfield(mu, C)
            Dmu = np.max(abs(mu - mu_new))
            mu = (1. - lamb) * mu + lamb * mu_new
        return mu


    def get_corr_eigen(self, mu):
        """use linearized approximation"""

        # autocorr matrix
        A = np.diag(mu * (1. - mu))

        # suszeptibility
        h_mu = self.get_mu_input(mu)
        h_sigma2 = self.get_sigma2_input(mu, A)
        S = np.diag( self.get_suszeptibility(h_mu, h_sigma2))
        
        # linearized coupling matrix: multiply each row with repective susceptibility
        W = np.dot(S, self.J)
        M = np.eye(self.N) - W

        # diagonalize the effective coupling matrix M = 1 - W
        # determine left v and right u eigenvectors
        lmbd, U = np.linalg.eig(M)
        V = np.linalg.inv(U)

        # check whether inversion makes sense
        diag_check = np.dot(V, np.dot(M, U))
        diag_check -= np.diag(diag_check.diagonal())
        assert(abs(np.mean(diag_check)) < 1e-15), 'Error inverting effective coupling matrix M = (1 - W)'

        # calculate covariance matrix (see e.g. tn_corr script eq. 5.6.7)
        Ap = np.dot(V, np.dot(A, V.T))
        C = np.zeros((self.N, self.N), dtype=float)
        for i in xrange(0, self.N):
            for j in xrange(0, self.N):
                C += 2.*Ap[i,j]/(lmbd[i]+lmbd[j]) * np.outer(U.T[i],U.T[j])

        # set autocorr
        for i,m_i in enumerate(mu):
            C[i,i] = m_i * (1. - m_i)

        return C
