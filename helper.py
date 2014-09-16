import numpy as np
import itertools as itr

def create_BM_weight_matrix(N):
    W = 2.*(np.random.rand(N,N)-0.5)
    for i in range(N):
        for j in range(i):
            W[j, i] = W[i, j]
    W -= np.diag(W.diagonal())
    return W

def create_BM_biases(N):
    return 2.*(np.random.rand(N)-.5)

def create_connectivity_matrix(N, w, g, epsilon, gamma):
    W = np.zeros((N, N))
    NE = int(gamma*N)
    NI = int(N-NE)
    KE = int(epsilon*NE)
    KI = int(epsilon*NI)
    for i in range(N):
        if NE > 0:
            indE = np.arange(0, NE)
            indE = indE[indE != i]
            indE = np.random.permutation(indE)[:KE]
            W[i, indE] = w
        if NI > 0:
            indI = np.arange(NE, N)
            indI = indI[indI != i]
            indI = np.random.permutation(indI)[:KI]
            W[i, indI] = -g*w
    return W

def get_E(W, b, s):
    return np.sum(0.5*np.dot(s.T, np.dot(W, s)) + np.dot(b,s))

def get_states(N):
    return np.array([np.array(x) for x in itr.product([0, 1], repeat=N)])

def get_theo_marginals(W, b):
    N = len(b)
    joints = get_theo_joints(W, b)
    states = get_states(N)
    p = []
    for i in range(N):
        p.append(np.sum(joints[states[:,i] == 1]))
    return p

def get_theo_joints(W, b):
    N = len(b)
    p = []
    states = get_states(N)
    for state in states:
        p.append(np.exp(get_E(np.array(W), np.array(b), np.array(state))))
    return 1.*np.array(p)/np.sum(p)

def get_variance(mu):
    return mu*(1.-mu)

def get_std(mu):
    return np.sqrt(get_variance(mu))

def get_joints(a_s, steps_warmup):
    N = len(a_s[0])
    statetensor = np.zeros([2 for i in range(N)])
    for s in a_s[steps_warmup:]:
        statetensor[tuple(s)] += 1
    return 1.*statetensor.flatten()/len(a_s[steps_warmup:])

def get_marginals(a_s, steps_warmup):
    N = len(a_s[0])
    p = np.empty(N)
    for i in range(N):
        p[i] = np.mean(a_s[:,i])
    return p

def get_DKL(p, q):
    return np.sum([p[i]*np.log(p[i]/q[i]) for i in range(len(p))])

def theta(x):
    if np.any(abs(x)  < 1e-15):
        raise ValueError('Invalid value in ecountered in theta(x).')
    else:
        return 1./2.*(np.sign(x)+1.)

def sigma(x):
    return 1./(1. + np.exp(-x))

def sigmainv(y):
    return np.log(1./(1./y - 1.))

def get_mun(K, gamma, g, w, smu):
    return (gamma - (1.-gamma)*g)*K*w*smu

def get_sigman(K, gamma, g, w, smu):
    return np.sqrt((gamma + (1.-gamma)*g**2)*K*w**2*get_variance(smu))
