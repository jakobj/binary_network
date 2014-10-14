import numpy as np
import scipy.special as scsp
import scipy.optimize as scop

import helper as bhlp

def get_mu_meanfield(epsilon, N, gamma, g, w, b, mu0, cII):
    def f(mu):
        return mu - scsp.erfc((-b-bhlp.get_mun(epsilon, N, gamma, g, w, mu))/get_sigma_meanfield(epsilon, N, gamma, g, w, mu, cII))
    return scop.fsolve(f, [mu0])[0]

def get_sigma_meanfield(epsilon, N, gamma, g, w, mu, cII):
    sigma2 = bhlp.get_variance(mu)
    K = epsilon*N
    return np.sqrt((gamma + (1.-gamma)*g**2)*K*w**2*sigma2+(1.-gamma)*K**2*w**2*g**2*cII)

def get_S(mu, sigma, b):
    return 1./(np.sqrt(2*np.pi)*sigma)*np.exp(-1.*(mu+b)**2/(2.*sigma**2))

def get_w_meanfield(K, w, h_mu, h_sigma, b):
    return get_S(h_mu, h_sigma, b)*w*K

def get_c_meanfield(epsilon, N, gamma, g, w, b, mu, h_mu, h_sigma):
    sigma = bhlp.get_std(mu)
    wmf = get_w_meanfield(epsilon*N, -g*w, h_mu, h_sigma, b)
    return wmf/(1.-wmf)*sigma**2/N
    # NE = int(gamma*N)
    # NI = N-NE
    # KE = int(epsilon*NE)
    # KI = int(epsilon*NI)
    # a_auto = np.array([get_variance(a_mu[0])*1./NE, get_variance(a_mu[1])*1./NI])
    # wEE = get_w_meanfield(KE, w, a_mu[0], 
    # A = np.array([[2.-2.*wEE, -2.*wEI, 0.],
    #               [-1.*wIE, 2.-(wEE+wII), -1.*wEI],
    #               [0, -2.*wIE, 2.-2.*wII]])
    # B = np.array([[2.*wEE, 0],
    #               [wIE, wEI],
    #               [0, 2.*wII]])
    # AINV = A.inverse()
    # return np.dot(AINV, np.dot(B, a_auto))

def get_m_c_iter(epsilon, N, gamma, g, w, b, mu0):
    """Calculate mean activity and mean correlations in a recurrent
    network iteratively, using the improved meanfield approach from
    Helias14
    """
    # initial guesses
    mu = mu0
    c = 0
    Dmu = 1e10
    Dc = 1e10
    while Dmu > 1e-15 and Dc > 1e-15:
        mu_old = mu
        c_old = c
        mu = get_mu_meanfield(epsilon, N, gamma, g, w, b, mu, c)
        h_mu = bhlp.get_mun(epsilon, N, gamma, g, w, mu)
        h_sigma = get_sigma_meanfield(epsilon, N, gamma, g, w, mu, c)
        c = get_c_meanfield(epsilon, N, gamma, g, w, b, mu, h_mu, h_sigma)
        # import ipdb;ipdb.set_trace()
        Dmu = abs(mu-mu_old)
        Dc = abs(c-c_old)
    return mu, c
