import numpy as np
import scipy.special as scsp
import scipy.optimize as scop

import helper as bhlp

class binary_meanfield:

    def __init__(self, epsilon, N, gamma, g, w, b):
        self.NE = int(gamma*N)
        self.NI = N-self.NE
        self.KE = int(epsilon*self.NE)
        self.KI = int(epsilon*self.NI)
        self.K = np.array([[self.KE, self.KI],
                           [self.KE, self.KI]])
        self.J = np.array([[w, -g*w],
                           [w, -g*w]])
        self.b = b
        self.C = np.array([[0., 0.],
                           [0., 0.]])
        self.mu = np.array([0., 0.])

    def get_mu_meanfield(self, mu0, C):
        def f(mu):
            h_mu = self.get_mu_input(mu)
            h_sigma = self.get_sigma_input(mu, C)
            return mu - 0.5*scsp.erfc((-self.b-h_mu)/(np.sqrt(2.)*h_sigma))
        return scop.fsolve(f, mu0)

    def get_mu_input(self, mu):
        return np.dot(self.K*self.J, mu)

    def get_sigma_input(self, mu, C):
        a = bhlp.get_variance(mu)
        sigma_shared = np.dot(self.K*self.J*self.J, a)
        sigma_corr = np.diag( np.dot( np.dot(self.K*self.J, C), (self.K*self.J).T) )
        return np.sqrt(sigma_shared + sigma_corr)

    def get_suszeptibility(self, mu, sigma):
        return 1./(np.sqrt(2.*np.pi)*sigma) * np.exp(-1.*(mu+self.b)**2 / (2.*sigma**2))

    def get_w_meanfield(self, h_mu, h_sigma):
        return ((self.K*self.J).T*self.get_suszeptibility(h_mu, h_sigma)).T

    def get_c_meanfield(self, mu, h_mu, h_sigma):
        a = bhlp.get_variance(mu)
        A = np.zeros(2)
        A[0] = a[0] * 1./self.NE if self.NE > 0 else 0.
        A[1] = a[1] * 1./self.NI if self.NI > 0 else 0.
        W = self.get_w_meanfield(h_mu, h_sigma)
        M = np.array([[2.-2.*W[0,0], -2.*W[0,1], 0.],
                      [-1.*W[1,0], 2.-(W[0,0]+W[1,1]), -1.*W[0,1]],
                      [0, -2.*W[1,0], 2.-2.*W[1,1]]])
        B = np.array([[2.*W[0,0], 0],
                      [W[1,0], W[0,1]],
                      [0, 2.*W[1,1]]])
        rhs = np.dot(B, A)
        c = np.linalg.solve(M, rhs)
        C = np.array([[c[0], c[1]],
                      [c[1], c[2]]])
        return C


    def get_m_c_iter(self, mu0):
        """Calculate mean activity and mean correlations in a recurrent
        network iteratively, using the improved meanfield approach from
        Helias14
        """
        Dmu = 1e10
        Dc = 1e10
        mu = mu0
        C = self.C
        while Dmu > 1e-15 and Dc > 1e-15:
            mu_old = np.sum(mu)
            c_old = np.sum(C)
            mu = self.get_mu_meanfield(mu, C)
            h_mu = self.get_mu_input(mu)
            h_sigma = self.get_sigma_input(mu, C)
            C = self.get_c_meanfield(mu, h_mu, h_sigma)
            Dmu = abs(np.sum(mu)-mu_old)
            Dc = abs(np.sum(C)-c_old)
        self.mu = mu
        self.C = C
        return mu, C, h_mu, h_sigma


    def get_m(self, mu0):
        """Calculate mean activity in a recurrent
        network using meanfield approach
        """
        mu = mu0
        C = np.array([[0., 0.],
                      [0., 0.]])
        mu = self.get_mu_meanfield(mu, C)
        h_mu = self.get_mu_input(mu)
        h_sigma = self.get_sigma_input(mu, C)
        return mu, h_mu, h_sigma
