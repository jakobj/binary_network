import numpy as np
import scipy.special as scsp
import scipy.optimize as scop

import helper as bhlp

def get_mu_meanfield(epsilon, N, gamma, g, w, b, mu0, c):
    def f(mu):
        h_mu = bhlp.get_mun(epsilon, N, gamma, g, w, mu)
        h_sigma = get_sigma_input(epsilon, N, gamma, g, w, mu, c)
        return mu - 0.5*scsp.erfc((-b-h_mu)/(np.sqrt(2.)*h_sigma))
    return scop.fsolve(f, [mu0])[0]

def get_mu_meanfield_m(K, J, b, mu0, C):
    def f(mu):
        h_mu = get_mu_input(K, J, mu)
        h_sigma = get_sigma_input_m(K, J, mu, C)
        return mu - 0.5*scsp.erfc((-b-h_mu)/(np.sqrt(2.)*h_sigma))
    return scop.fsolve(f, mu0)

def get_mu_input(K, J, mu):
    return np.dot(K*J, mu)

def get_sigma_input(epsilon, N, gamma, g, w, mu, cII):
    sigma2 = bhlp.get_variance(mu)
    K = epsilon*N
    return np.sqrt((gamma + (1.-gamma)*g**2)*K*w**2*sigma2 + (1.-gamma)*K**2*w**2*g**2*cII)

def get_sigma_input_m(K, J, mu, C):
    a = bhlp.get_variance(mu)
    sigma_m = np.dot(K*J*J, a)
    sigma_c = np.diag( np.dot( np.dot(K*J, C), (K*J).T) )
    return np.sqrt(sigma_m + sigma_c)

def get_S(mu, sigma, b):
    return 1./(np.sqrt(2.*np.pi)*sigma) * np.exp(-1.*(mu+b)**2 / (2.*sigma**2))

# def get_w_meanfield(K, w, h_mu, h_sigma, b):
#     return get_S(h_mu, h_sigma, b)*w*K
def get_w_meanfield_m(K, J, h_mu, h_sigma, b):
    return ((J*K).T*get_S(h_mu, h_sigma, b)).T

# def get_c_meanfield(epsilon, N, gamma, g, w, b, mu, h_mu, h_sigma):
#     sigma = bhlp.get_std(mu)
#     wmf = get_w_meanfield(epsilon*N, -g*w, h_mu, h_sigma, b)
#     return wmf/(1.-wmf)*sigma**2/N
def get_c_meanfield_m(K, J, NE, NI, b, mu, h_mu, h_sigma):
    a = bhlp.get_variance(mu)
    A = np.zeros(2)
    A[0] = a[0] * 1./NE if NE > 0 else 0.
    A[1] = a[1] * 1./NI if NI > 0 else 0.
    W = get_w_meanfield_m(K, J, h_mu, h_sigma, b)
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

def get_m_c_iter(epsilon, N, gamma, g, w, b, mu0):
    """Calculate mean activity and mean correlations in a recurrent
    network iteratively, using the improved meanfield approach from
    Helias14
    """
    # initial guesses
    # mu = np.array([mu0, mu0])
    # c = np.array([0., 0., 0.])
    # mu = mu0
    # c = 0.
    mu = np.array([mu0, mu0]).T
    NE = int(gamma*N)
    NI = N-NE
    KE = int(epsilon*NE)
    KI = int(epsilon*NI)
    K = np.array([[KE, KI],
                  [KE, KI]])
    J = np.array([[w, -g*w],
                  [w, -g*w]])
    C = np.array([[0., 0.],
                  [0., 0.]])
    b = np.array([b, b]).T
    Dmu = 1e10
    Dc = 1e10
    while Dmu > 1e-15 and Dc > 1e-15:
        mu_old = np.sum(mu)
        c_old = np.sum(C)
        # mu = get_mu_meanfield(epsilon, N, gamma, g, w, b, mu, c)
        mu = get_mu_meanfield_m(K, J, b, mu, C)
        # h_mu = bhlp.get_mun(epsilon, N, gamma, g, w, mu)
        # h_sigma = get_sigma_input(epsilon, N, gamma, g, w, mu, c)
        h_mu = get_mu_input(K, J, mu)
        h_sigma = get_sigma_input_m(K, J, mu, C)
        # C = get_c_meanfield(epsilon, N, gamma, g, w, b, mu, h_mu, h_sigma)
        C = get_c_meanfield_m(K, J, NE, NI, b, mu, h_mu, h_sigma)
        # import ipdb;ipdb.set_trace()
        Dmu = abs(np.sum(mu)-mu_old)
        Dc = abs(np.sum(C)-c_old)
    return mu, C, h_mu, h_sigma
