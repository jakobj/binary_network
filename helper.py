import numpy as np
import itertools as itr
from collections import defaultdict


def create_BM_weight_matrix(N):
    W = 2. * (np.random.rand(N, N) - 0.5)
    for i in range(N):
        for j in range(i):
            W[j, i] = W[i, j]
    W -= np.diag(W.diagonal())
    return W


def create_BM_biases(N):
    return 2. * (np.random.rand(N) - .5)


def create_connectivity_matrix(N, w, g, epsilon, gamma):
    W = np.zeros((N, N))
    NE = int(gamma * N)
    NI = int(N - NE)
    KE = int(epsilon * NE)
    KI = int(epsilon * NI)
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
            W[i, indI] = -g * w
    return W


def create_noise_connectivity_matrix(Nbm, Nnoise, gamma, g, w, epsilon):
    W = np.zeros((Nbm, Nnoise))
    NE = int(gamma * Nnoise)
    NI = int(Nnoise - NE)
    KE = int(epsilon * NE)
    KI = int(epsilon * NI)
    for l in range(Nbm):
        indE = np.random.permutation(np.arange(0, NE))[:KE]
        W[l, indE] = w
        indI = np.random.permutation(np.arange(NE, Nnoise))[:KI]
        W[l, indI] = -g * w
    return W


def get_energy(W, b, s):
    return -1. * np.sum(0.5 * np.dot(s.T, np.dot(W, s)) + np.dot(b, s))


def get_states(N):
    return np.array([np.array(x) for x in itr.product([0, 1], repeat=N)])


def get_theo_marginals(W, b, beta):
    N = len(b)
    joints = get_theo_joints(W, b, beta)
    states = get_states(N)
    p = []
    for i in range(N):
        p.append(np.sum(joints[states[:, i] == 1]))
    return p


def get_theo_joints(W, b, beta):
    N = len(b)
    p = []
    states = get_states(N)
    for state in states:
        p.append(
            np.exp(-1. * beta * get_energy(np.array(W), np.array(b), np.array(state))))
    return 1. * np.array(p) / np.sum(p)


def get_sigma2(mu):
    return mu * (1. - mu)


def get_sigma(mu):
    return np.sqrt(get_sigma2(mu))


def get_sigma_input_from_beta(beta):
    return np.sqrt(8. / (np.pi * beta ** 2))


def get_beta_from_sigma_input(sigma_input):
    return 4. / np.sqrt(2. * np.pi * sigma_input ** 2)


def get_joints(a_s, steps_warmup):
    steps_tot = len(a_s[steps_warmup:])
    states = defaultdict(int)
    for s in a_s[steps_warmup:]:
        states[tuple(s)] += 1
    states = np.array([it[1] for it in sorted(states.items())])
    return 1. * states / steps_tot


def get_marginals(a_s, steps_warmup):
    N = len(a_s[0])
    p = np.empty(N)
    for i in range(N):
        p[i] = np.mean(a_s[steps_warmup:, i])
    return p


def get_DKL(p, q):
    return np.sum([p[i] * np.log(p[i] / q[i]) for i in range(len(p))])


def theta(x, beta=1.):
    if abs(x) < 1e-15:
        raise ValueError('Invalid value in ecountered in theta(x).')
    else:
        return 1. / 2. * (np.sign(x) + 1.)


def sigma(x, beta=1.):
    return 1. / (1. + np.exp(-beta * x))


def sigmainv(y, beta=1.):
    return 1. / beta * np.log(1. / (1. / y - 1.))


def get_mu_input(epsilon, N, gamma, g, w, mu):
    return (gamma - (1. - gamma) * g) * epsilon * N * w * mu


def get_sigma_input(epsilon, N, gamma, g, w, mu):
    sigma2 = get_sigma2(mu)
    return np.sqrt((gamma + (1. - gamma) * g ** 2) * epsilon * N * w ** 2 * sigma2)


def Fsigma(x, beta=1.):
    return 0 if sigma(x, beta) < np.random.rand() else 1


def bin_binary_data(times, a_states, tbin, tmax):
    a_s = a_states.T.copy()
    times_bin = np.arange(0., tmax + tbin, tbin)
    st = np.zeros((len(a_s), len(times_bin)))
    Ntimes = len(times)
    for j, s in enumerate(a_s):
        idl = 0
        for i, tc in enumerate(times_bin):
            while idl < Ntimes - 1 and times[idl + 1] <= tc:
                idl += 1
            st[j][i] = s[idl]
    return times_bin, st


def autocorrf(times_bin, st, tmax):
    times = np.hstack([-1. * times_bin[1:][::-1], 0, times_bin[1::]])
    Nbins = len(st[0])
    offset_edge = np.hstack(
        [np.arange(1, Nbins), Nbins, np.arange(1, Nbins)[::-1]])
    autof = np.mean(
        [np.correlate(s, s, 'full') / offset_edge - np.mean(s) ** 2 for s in st], axis=0)
    autof = autof[abs(times) <= tmax]
    times = times[abs(times) <= tmax]
    return times, autof


def crosscorrf(times_bin, st, tmax):
    N = len(st)
    times_cauto, cauto = autocorrf(times_bin, [np.sum(st, axis=0)], tmax)
    times_autof, mu_autof = autocorrf(times_bin, st, tmax)
    mu_crossf = 1. / (N * (N - 1)) * (cauto - 1. * N * mu_autof)
    return times_autof, mu_autof, mu_crossf
