import numpy as np
import itertools as itr
import network as bnet

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
    if abs(x) < 1e-15:
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
    st = np.zeros((len(a_s), len(times_bin)))
    Ntimes = len(times)
    for j,s in enumerate(a_s):
        idl = 0
        for i,tc in enumerate(times_bin):
            while idl < Ntimes-1 and times[idl+1] <= tc:
                idl += 1
            st[j][i] = s[idl]
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
    times_cauto, cauto = autocorrf(times_bin, [np.sum(st, axis=0)], tmax)
    times_autof, mu_autof = autocorrf(times_bin, st, tmax)
    mu_crossf = 1./(N*(N-1))*(cauto-1.*N*mu_autof)
    return times_autof, mu_autof, mu_crossf

def calibrate_noise(N, Nnoise, epsilon, gamma, g, w, tau, time, mu_target, mu_noise_target, std_noise_target):
    Nrec = N+Nnoise
    W = np.zeros((N+Nnoise, N+Nnoise))
    W[:N,N:] = create_noise_connectivity_matrix(N, Nnoise, gamma, g, w, epsilon)
    W[N:,N:] = create_connectivity_matrix(Nnoise, w, g, epsilon, gamma)
    b = np.zeros(N+Nnoise)
    b[:N] = -w/2.
    b[N:] = -1.*get_mun(epsilon*Nnoise, gamma, g, w, mu_target)-1.*w/2.
    Nact = int(mu_target*(N+Nnoise))
    sinit = np.random.permutation(np.hstack([np.ones(Nact), np.zeros(N+Nnoise-Nact)]))
    a_times, a_s, a_ui = bnet.simulate_eve(W, b, tau, sinit.copy(), time, Nrec, [N+Nnoise], [theta], record_ui=True, Nrec_ui=N)
    std_input = np.mean(np.std(a_ui, axis=0))
    w_adj = w*std_noise_target/std_input
    b_adj = mu_noise_target-1.*get_mun(epsilon*Nnoise, gamma, g, w_adj, np.mean(a_s[:,N:]))
    return w_adj, b_adj

def calibrate_poisson_noise(N, Nnoise, epsilon, gamma, g, w, tau, time, mu_target, mu_noise_target, std_noise_target):
    Nrec = N+Nnoise
    W = np.zeros((N+Nnoise, N+Nnoise))
    W[:N,N:] = create_noise_connectivity_matrix(N, Nnoise, gamma, g, w, epsilon)
    b = np.zeros(N+Nnoise)
    b[:N] = -w/2.
    b[N:] = sigmainv(mu_target)
    Nact = int(mu_target*(N+Nnoise))
    sinit = np.random.permutation(np.hstack([np.ones(Nact), np.zeros(N+Nnoise-Nact)]))
    a_times, a_s, a_ui = bnet.simulate_eve(W, b, tau, sinit.copy(), time, Nrec, [N, N+Nnoise], [theta, Fsigma], record_ui=True, Nrec_ui=N)
    std_input = np.mean(np.std(a_ui, axis=0))
    w_adj = w*std_noise_target/std_input
    b_adj = mu_noise_target-1.*get_mun(epsilon*Nnoise, gamma, g, w_adj, np.mean(a_s[:,N:]))
    return w_adj, b_adj
