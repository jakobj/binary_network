import numpy as np
import itertools as itr
import scipy.special as scsp


def binomial_outdegree_multapses(M, K, N, m):
    """probability to find a source with m outputs for choosing for M
    neurons K sources from a pool of N neurons, with allowing a source
    to be chosen more than once for a single target

    """
    return scsp.binom(M * K, m) * (1./N)**m * (1. - 1./N)**(M * K - m)


def binomial_outdegree(M, K, N, m):
    """probability to find a source with m outputs for choosing for M
    neurons K sources from a pool of N neurons, without choosing a
    source twice for a single target

    """
    return scsp.binom(M, m) * (1. * K/N)**m * (1. - 1. * K/N)**(M - m)


def binomial_shared(K, N, s):
    """distribution of choosing s shared inputs for choosing K sources of
    a pool of N sources

    """
    return scsp.binom(K, s) * (1. * K / N)**s * (1. - 1. * K / N)**(K - s)


def create_BM_weight_matrix_normal(N, muJ, sigmaJ):
    W = np.random.normal(muJ, sigmaJ, (N, N))
    for i in range(N):
        for j in range(i):
            W[j, i] = W[i, j]
    W -= np.diag(W.diagonal())
    return W


def create_BM_biases_normal(N, muJ, mu_target):
    return np.ones(N) * -1. * muJ * N * mu_target


def create_BM_weight_matrix(N, M=1):
    Ntot = M*N
    W = np.zeros((Ntot, Ntot))
    for i in range(M):
        W[i*N:(i+1)*N, i*N:(i+1)*N] = 2. * (np.random.rand(N, N) - 0.5)
    for i in range(Ntot):
        for j in range(i):
            W[j, i] = W[i, j]
    W -= np.diag(W.diagonal())
    return W


def create_BM_biases(N, M=1):
    return 2. * (np.random.rand(M*N) - .5)


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


def create_noise_connectivity_matrix_only_pairwise(Nbm, NE, NI, g, w, KE, KI):
    W = np.zeros((Nbm, NE+NI))
    if KE > 0:
        assert(Nbm * KE < 2 * NE)
    if KI > 0:
        assert(Nbm * KI < 2 * NI)
    outdegree = {}
    sources = np.arange(0, NE+NI)
    for l in xrange(Nbm):
        indegree = 0
        while indegree < KE:
            indE = np.random.permutation(sources[:KE])[0]
            if indE in outdegree.keys():
                if outdegree[indE] < 2 and abs(W[l, indE]) < 1e-12:
                    W[l, indE] = w
                    outdegree[indE] += 1
                    indegree += 1
            else:
                W[l, indE] = w
                outdegree[indE] = 1
                indegree += 1
        indegree = 0
        while indegree < KI:
            indI = np.random.permutation(sources[KE:])[0]
            if indI in outdegree.keys():
                if outdegree[indI] < 2 and abs(W[l, indI]) < 1e-12:
                    W[l, indI] = -g * w
                    outdegree[indI] += 1
                    indegree += 1
                    if outdegree[indI] == 2:
                        sources = np.delete(sources, np.where(sources == indI)[0])
            else:
                W[l, indI] = -g * w
                outdegree[indI] = 1
                indegree += 1
    return W


def create_noise_connectivity_matrix_fixed_pairwise(Nbm, Nnoise, gamma, g, w, epsilon):
    NE = int(gamma * Nnoise)
    NI = int(Nnoise - NE)
    KE = int(epsilon * NE)
    KI = int(epsilon * NI)
    KEshared = 0
    KIshared = 0
    if NE > 0:
        KEshared = int(1. * KE**2 / NE)
    if NI > 0:
        KIshared = int(1. * KI**2 / NI)
    # check whether it is possible to realize desired connectivity;
    # this translate to (Nbm - 1 ) * epsilon <= 1
    assert(KEshared * (Nbm - 1) <= KE), '[error] impossible parameter choices'
    assert(KIshared * (Nbm - 1) <= KI), '[error] impossible parameter choices'
    # the two next conditions are fullfilled if the two conditions
    # above are fullfilled; they are left in for documentation
    assert(Nbm*(KE-KEshared) <= NE)
    assert(Nbm*(KI-KIshared) <= NI)
    W = np.zeros((Nbm, NE+NI))
    for k in xrange(2):
        K = [KE, KI][k]
        Kshared = [KEshared, KIshared][k]
        wt = [w, -g*w][k]
        if K > 0:
            template = np.zeros((Nbm, K))
            l = 0
            i = 0
            Kshared_count = 0
            while l < Nbm:
                if l == 0:
                    template[l, i] = wt
                    i += 1
                    if i == K:
                        i = l
                        l += 1
                else:
                    if Kshared_count < Kshared:
                        template[l, i] = wt
                        Kshared_count += 1
                    i += 1
                    if Kshared_count == Kshared:
                        Kshared_count = 0
                        l += 1
            offset_i = k*(KE-KEshared)*Nbm
            for l in xrange(Nbm):
                if l == 0:
                    W[l:Nbm, offset_i:offset_i+K] = template
                    offset_i += K
                else:
                    if Kshared == 0:
                        W[l:Nbm, offset_i:offset_i+K-l*Kshared] = template[:-l, :]
                    else:
                        W[l:Nbm, offset_i:offset_i+K-l*Kshared] = template[:-l, :-l*Kshared]
                    offset_i += K-l*Kshared
    return W


def create_hybridnoise_connectivity_matrix(Nbm, Nnoise, gamma, g, w, epsilon):
    W = np.zeros((Nbm, Nnoise))
    NE = int(gamma * Nnoise)
    NI = int(Nnoise - NE)
    KE = int(epsilon * NE)
    KI = int(epsilon * NI)
    for l in range(Nbm):
        ind = np.random.permutation(np.arange(0, Nnoise))[:KE+KI]
        W[l, ind[:KE]] = w
        W[l, ind[KE:]] = -g * w
    return W


def create_indep_noise_connectivity_matrix(Nbm, Knoise, gamma, g, w):
    Nnoise = Nbm * Knoise
    W = np.zeros((Nbm, Nnoise))
    KE = int(gamma * Knoise)
    for l in range(Nbm):
        indE = np.arange(l*Knoise, l*Knoise + KE)
        W[l, indE] = w
        indI = np.arange(l*Knoise + KE, (l+1) * Knoise)
        W[l, indI] = -g * w
    return W


def create_noise_recurrent_connectivity_matrix(Nbm, Nnoise, epsilon):
    W = np.zeros((Nnoise, Nbm))
    K = epsilon*Nbm
    for l in range(Nnoise):
        ind = np.random.permutation(np.arange(0, Nbm))[:K]
        W[l, ind] = 2. * (np.random.rand(K) - 0.5)
    return W


def get_energy(W, b, s, beta=1.):
    return -1. * beta * np.sum(0.5 * np.dot(s.T, np.dot(W, s)) + np.dot(b, s))


def get_states(N):
    return np.array([np.array(x) for x in itr.product([0, 1], repeat=N)])


def get_theo_marginals(W, b, beta):
    N = len(b)
    joints = get_theo_joints(W, b, beta)
    states = get_states(N)
    m = np.zeros(N)
    for i in range(N):
        m[i] = np.sum(joints[states[:, i] == 1])
    return m


def get_theo_covariances(W, b, beta):
    N = len(b)
    joints = get_theo_joints(W, b, beta)
    states = get_states(N)
    m = get_theo_marginals(W, b, beta)
    cov = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cov[i, j] = np.sum(joints[np.logical_and(states[:, i] == 1, states[:, j] == 1)]) - m[i] * m[j]
    return m, cov


def get_theo_joints(W, b, beta, M=1):
    N = len(b)/M
    joints = []
    for i in range(M):
        p = []
        states = get_states(N)
        for state in states:
            p.append(
                np.exp(-1. * get_energy(np.array(W[i*N:(i+1)*N,i*N:(i+1)*N]), np.array(b[i*N:(i+1)*N]), np.array(state), beta)))
        joints.append(np.array(p)/np.sum(p))
    if M == 1:
        return joints[0]
    else:
        return joints


def get_sigma2(mu):
    """
    returns variance of activity mu
    """
    return mu * (1. - mu)


def get_sigma(mu):
    """
    returns standard deviation of activity mu
    """
    return np.sqrt(get_sigma2(mu))


def get_sigma_input_from_beta(beta):
    """returns standard deviation of input given inverse temperature
    beta, by requiring matching of erfc and sigmoidal activation
    functions at zero

    """
    return np.sqrt(8. / (np.pi * beta ** 2))


def get_beta_from_sigma_input(sigma_input):
    """returns inverse temperature beta from standard deviation of
    input, by requiring matching of erfc and sigmoidal activation
    functions at zero

    """
    return 4. / np.sqrt(2. * np.pi * sigma_input ** 2)


def get_joints(a_s, steps_warmup, M=1, prior=None):
    steps_tot = len(a_s[steps_warmup:])
    N = len(a_s[0,:])/M
    a_joints = np.empty((M, 2**N))
    possible_states = get_states(N)
    states = {}
    for i in range(M):
        if prior is None:
            for s in possible_states:
                states[tuple(s)] = 0.
        elif prior == 'uniform':
            for s in possible_states:
                states[tuple(s)] = 1.
            steps_tot += len(possible_states)
        for s in a_s[steps_warmup:, i*N:(i+1)*N]:
            states[tuple(s)] += 1
        states_sorted = np.array([it[1] for it in sorted(states.items())])
        a_joints[i,:] = 1.* states_sorted / steps_tot
    if M == 1:
        return a_joints[0,:]
    else:
        return a_joints


def get_steps_warmup(rNrec, Twarmup, tau):
    Nrec = rNrec[1] - rNrec[0]
    assert(Nrec >= 0)
    return int(np.ceil(1.*Nrec*Twarmup/tau))


def get_joints_sparse(sinit, a_s, steps_warmup, M=1, prior=None):
    steps_tot = len(a_s[steps_warmup:])
    N = len(sinit)/M
    a_joints = np.empty((M, 2**N))
    possible_states = get_states(N)
    states = {}
    for i in range(M):
        cstate = sinit.copy()
        if prior is None:
            for s in possible_states:
                states[tuple(s)] = 0.
        elif prior == 'uniform':
            for s in possible_states:
                states[tuple(s)] = 1.
            steps_tot += len(possible_states)
        for step, (idx, sidx) in enumerate(a_s):
            cstate[idx] = sidx
            if step >= steps_warmup:
                states[tuple(cstate[i*N:(i+1)*N])] += 1
        states_sorted = np.array([it[1] for it in sorted(states.items())])
        a_joints[i,:] = 1.* states_sorted / steps_tot
    if M == 1:
        return a_joints[0]
    else:
        return a_joints


def get_all_states_from_sparse(sinit, a_s, steps_warmup):
    steps_tot = len(a_s)
    N = len(sinit)
    a_s_full = np.empty((steps_tot, N))
    cstate = sinit.copy()
    for step, (idx, sidx) in enumerate(a_s):
        cstate[idx] = sidx
        if step >= steps_warmup:
            a_s_full[step] = cstate
    a_s_full = a_s_full[steps_warmup:]
    return a_s_full


def get_marginals(a_s, steps_warmup, M=1):
    N = len(a_s[0, :]) / M
    a_marginals = np.empty((M, N))
    for j in range(M):
        for i in range(N):
            a_marginals[j, i] = np.mean(a_s[steps_warmup:, j*N+i])
    if M == 1:
        return a_marginals[0]
    else:
        return a_marginals


def get_euclidean_distance(x, y):
    return np.sqrt(np.dot(x-y, x-y))


def get_DKL(p, q, M=1):
    """returns the Kullback-Leibler divergence of distributions p and q

    """
    assert(np.shape(p) == np.shape(q))
    if M == 1:
        p = [p]
        q = [q]
    DKL = []
    for j in range(M):
        if abs(np.sum(p[j]) - 1.) > 1e-12 or abs(np.sum(q[j]) - 1.) > 1e-12:
            raise ValueError('Joint densities must be normalized.')
        if np.any(p[j] <= 0) or np.any(q[j] <= 0):
            DKL.append(np.nan)
        else:
            DKL.append(np.sum([p[j][i] * np.log(p[j][i] / q[j][i]) for i in range(len(p[j]))]))
    if M == 1:
        return DKL[0]
    else:
        return DKL


def theta(x, beta=1.):
    if abs(x) < 1e-15:
        raise ValueError('Invalid value in ecountered in theta(x).')
    else:
        return 0 if x < 0 else 1


def sigma(x, beta=1.):
    """sigmoidal function

    """
    return 1. / (1. + np.exp(-beta * x))


def Fsigma(x, beta=1.):
    """sigmoidal activation function for stochastic binary neurons

    """
    return 0 if 1. / (1. + np.exp(-beta * x)) < np.random.rand() else 1


def sigmainv(y, beta=1.):
    """returns bias b that leads to mean activity y of a stochastic binary
    neuron by inverting sigmoidal activation function

    """
    return 1. / beta * np.log(1. / (1. / y - 1.))


def get_mu_input(epsilon, N, gamma, g, w, mu):
    """returns mean input for given connection statistics and presynaptic
    activity

    """
    return (gamma - (1. - gamma) * g) * epsilon * N * w * mu


def get_sigma_input(epsilon, N, gamma, g, w, mu):
    """returns standard deviation of input for given connection statistics
    and presynaptic activity

    """
    sigma2 = get_sigma2(mu)
    return np.sqrt((gamma + (1. - gamma) * g ** 2) * epsilon * N * w ** 2 * sigma2)


def get_adjusted_weights_and_bias(W, b, b_eff, beta_eff, beta):
    """return adjusted weights matrix and bias vector for a Boltzmann
    machine, given the effective offset b_eff, the effective inverse
    temperature beta_eff and the target inverse termperature beta

    """
    return beta/beta_eff*W, beta/beta_eff*b+b_eff


def bin_binary_data(times, a_states, tbin, tmin, tmax):
    """returns a binned version of the input with binsize tbin from 0 to
    tmax. input are timestamps times and networks states (or field
    recordings) a_states, which have to have the same length

    """
    a_s = a_states.T.copy()
    times_bin = np.arange(tmin, tmax + tbin, tbin)
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
    """returns the population averaged autocorrelation function of the
    binned signal st

    """
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
    """returns the population averaged autocorrelation function of the
    binned signal st

    """
    N = len(st)
    times_cauto, cauto = autocorrf(times_bin, [np.sum(st, axis=0)], tmax)
    times_autof, mu_autof = autocorrf(times_bin, st, tmax)
    mu_crossf = 1. / (N * (N - 1)) * (cauto - 1. * N * mu_autof)
    return times_autof, mu_autof, mu_crossf
