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

def create_noise_connectivity_matrix(N, Nnoise, gamma, g, w, epsilon):
    W = np.zeros((N, Nnoise))
    NE = int(gamma*Nnoise)
    NI = Nnoise-NE
    KE = int(epsilon*NE)
    KI = int(epsilon*NI)
    for l in range(N):
        indE = np.random.permutation(np.arange(0, NE))[:KE]
        W[l, indE] = w
        indI = np.random.permutation(np.arange(NE, Nnoise))[:KI]
        W[l, indI] = -g*w
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
    for s in a_s[int(steps_warmup):]:
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

def get_sigman(K, gamma, g, w, sigma):
    return np.sqrt((gamma + (1.-gamma)*g**2)*K*w**2*sigma**2)

def get_weight_noise(beta, sigma, K, gamma, g):
    return np.sqrt(8./(np.pi*beta**2*sigma**2*K*(gamma + (1.-gamma)*g**2)))

def Fsigma(x):
    return 0 if sigma(x) < np.random.rand() else 1

def bin_binary_data(times, a_states, tbin, time):
    a_s = a_states.T.copy()
    times_bin = np.arange(0., time+tbin, tbin)
    T = len(times_bin)
    st = np.zeros((len(a_s), T))
    for j,s in enumerate(a_s):
        for i in range(len(st[j])):
            tc = i*tbin
            idl = np.where(times <= tc)[0]
            if len(idl) > 0:
                st[j][i] = s[idl[-1]]
            else:
                pass
    return times_bin, st

def autocorrf(times_bin, st, tmax):
    times = np.hstack([-1.*times_bin[1:][::-1],0,times_bin[1::]])
    Nbins = len(st[0])
    offset_edge = np.hstack([np.arange(1, Nbins), Nbins, np.arange(1, Nbins)[::-1]])
    autof = np.mean([np.correlate(s,s, 'full')/offset_edge-np.mean(s)**2 for s in st], axis=0)
    autof = autof[abs(times) <= tmax]
    times = times[abs(times) <= tmax]
    return times, autof

def crosscorrf(times_bin, st, tmax):
    N = len(st)
    Nbins = len(times_bin)
    times = np.hstack([-1.*times_bin[1:][::-1],0,times_bin[1::]])
    crossf = np.empty((N, N, len(times)))
    a_r = np.mean(st, axis=1)
    offset_edge = np.hstack([np.arange(1,Nbins),Nbins,np.arange(1,Nbins)[::-1]])
    for i in range(N):
        for j in range(N):
            crossf[i,j] = np.correlate(st[i], st[j], 'full')/offset_edge-a_r[i]*a_r[j]
    mu_autof = np.empty((N, len(times)))
    mu_crossf = np.empty((N**2, len(times)))
    for i in range(N):
        for j in range(N):
            if i == j:
                mu_autof[i] = crossf[i,j]
            else:
                mu_crossf[i*N+j] = crossf[i,j]
    mu_autof = np.mean(mu_autof, axis=0)
    mu_crossf = 1./(N*(N-1.))*np.sum(mu_crossf, axis=0)
    mu_autof = mu_autof[abs(times) <= tmax]
    mu_crossf = mu_crossf[abs(times) <= tmax]
    times = times[abs(times) <= tmax]
    return times, mu_autof, mu_crossf
