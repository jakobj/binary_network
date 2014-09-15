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
    return 2.*(np.random.rand()-.5)

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
