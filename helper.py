# global imports
import numpy as np
import itertools
import scipy
import scipy.special
import scipy.stats


def get_states(N):
    """return all possible states as arrays for N binary units"""
    return np.array([np.array(x) for x in itertools.product([0, 1], repeat=N)])


def get_states_as_strings(N):
    """returns all possible states as strings for N binary units"""
    return np.array([state_array_to_string(s) for s in get_states(N)])


def state_array_to_string(s):
    return ''.join(np.array(s, dtype=str))


def state_array_to_int(s):
    """translates a state s into an integer by interpreting the state as a
    binary represenation"""
    return int(state_array_to_string(s), 2)


def state_string_from_int(i, N):
    """translates an integer i into a state string by using the binary
    representation of the integer"""
    return bin(i)[2:].zfill(N)


def state_array_from_int(i, N):
    """translates an integer i into a state by using the binary
    representation of the integer"""
    return np.array([int(si) for si in state_string_from_int(i, N)])


def random_initial_condition(N):
    return np.random.randint(0, 2, N)


def adjust_time_slices(a_time, steps_warmup):
    return a_time[steps_warmup:]


def outdegree_distribution(M, K, N, m):
    """probability to find a source with m outputs for choosing for M
    neurons K sources from a pool of N neurons, without choosing a
    source twice for a single target

    """
    return scipy.stats.binom.pmf(m, M, 1. * K / N, 0)


def shared_input_distribution(K, N, s):
    """distribution of choosing s shared inputs for choosing K sources of
    a pool of N sources

    """
    return scipy.stats.binom.pmf(s, K, 1. * K / N, 0)


def create_BM_weight_matrix(N, distribution, mu_weight=None, **kwargs):
    """creates a random weight matrix for a Boltzmann machine (diagonal=0,
    and symmetric weights), with weights drawn from
    distribution. parameters for the distribution need to be passed as
    kwargs.

    """
    W = distribution(size=(N, N), **kwargs)
    # we can not just use 0.5 * (W + W.T), without altering distribution of weights
    for i in xrange(N):
        for j in xrange(i):
            W[j, i] = W[i, j]
    W -= np.diag(W.diagonal())
    if mu_weight is not None:
        W += mu_weight - 1. / (N * (N - 1)) * np.sum(W)
        W -= np.diag(W.diagonal())
    return W


def create_BM_biases(N, distribution, **kwargs):
    """create a random bias vector for a Boltzmann machine, with biases
    drawn from distribution. parameters for the distribution need to
    be passed as kwargs.

    """
    return distribution(size=N, **kwargs)


def create_multi_BM_weight_matrix(N, M, distribution, **kwargs):
    Ntot = M * N
    W = np.zeros((Ntot, Ntot))
    for i in range(M):
        W[i * N:(i + 1) * N, i * N:(i + 1) * N] = create_BM_weight_matrix(N, distribution, **kwargs)
    return W


def create_multi_BM_biases(N, M, distribution, **kwargs):
    return create_BM_biases(N * M, distribution, **kwargs)


def create_BM_biases_threshold_condition(N, muJ, mu_target):
    """create biases for a Boltzmann machine, by requiring that the
    average input from other neurons in the BM sums with the bias to
    zero. this way we can achieve an average activity in the BM of
    mu_target. for details see, e.g., Helias et al. (2014), PloS CB,
    eq. (5)

    """
    return np.ones(N) * -1. * muJ * N * mu_target


def create_BRN_weight_matrix(N, w, g, epsilon, gamma):
    """create a random realization of a weight matrix for an E/I
    network of N neurons with fixed weights.

    """
    return create_BRN_weight_matrix_fixed_indegree(N, w, g, int(epsilon * N), gamma)


def create_BRN_weight_matrix_fixed_indegree(N, w, g, K, gamma):
    """create a random realization of a weight matrix for an E/I
    network of N neurons with fixed weights.

    """
    W = np.zeros((N, N))
    NE = int(gamma * N)
    NI = int(N - NE)
    KE = int(gamma * K)
    KI = int(K - KE)
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


def create_BRN_biases_threshold_condition(N, w, g, epsilon, gamma, mu_target):
    """(see create_BM_biases_threshold_condition)"""
    return np.ones(N) * -1. * get_mu_input(epsilon, N, gamma, g, w, mu_target) - w / 2.


def create_stoch_biases_from_target_activity(N, mu_target):
    """create biases for sigmoidal units from a target activity by using
    the inverse of the sigmoid"""
    return np.ones(N) * sigmainv(mu_target)


def create_noise_weight_matrix(Nbm, Nnoise, gamma, g, w, epsilon):
    return create_noise_weight_matrix_fixed_indegree(Nbm, Nnoise, gamma, g, w, int(epsilon * Nnoise))


def create_noise_weight_matrix_fixed_indegree(Nbm, Nnoise, gamma, g, w, Knoise):
    """create a random realization of a weight matrix for Nnoise sources
    projecting to Nbm targets with E/I connections of fixed weight and
    with fixed total in degree Knoise.

    """
    W = np.zeros((Nbm, Nnoise))
    NEnoise = int(gamma * Nnoise)
    KEnoise = int(gamma * Knoise)
    KInoise = int(Knoise - KEnoise)
    for l in W:
        indE = np.random.permutation(np.arange(0, NEnoise))[:KEnoise]
        l[indE] = w
        indI = np.random.permutation(np.arange(NEnoise, Nnoise))[:KInoise]
        l[indI] = -g * w
    return W


def create_noise_weight_matrix_2dshuffle(Nbm, Nnoise, gamma, g, w, epsilon):
    return create_noise_weight_matrix_2dshuffle_fixed_indegree(Nbm, Nnoise, gamma, g, w, int(epsilon * Nnoise))


def create_noise_weight_matrix_2dshuffle_fixed_indegree(Nbm, Nnoise, gamma, g, w, Knoise):
    """create a random realizations of a weight matrix for Nnoise sources
    projecting to Nbm targets with identity of presynaptic neurons
    shuffled across E/I populations

    """
    W = np.zeros((Nbm, Nnoise))
    KE = int(gamma * Knoise)
    KI = int(Knoise - KE)
    for l in range(Nbm):
        ind = np.random.permutation(np.arange(0, Nnoise))[:KE + KI]
        W[l, ind[:KE]] = w
        W[l, ind[KE:]] = -g * w
    return W


def _generate_template(Nbm, K, Kshared, w, Ktot, N, random=False):
    assert(Nbm > 0 and K > 0)
    template = np.zeros((Nbm, K))
    l = 0
    i = 0
    Kshared_counts = np.zeros(Nbm)
    while l < Nbm:
        if l == 0:
            template[l, i] = w
            i += 1
            if i == K:
                i = l
                l += 1
                if random:
                    Kshared = scipy.random.binomial(Ktot, 1. * Ktot / N)
        else:
            if Kshared_counts[l] < Kshared:
                template[l, i] = w
                Kshared_counts[l] += 1
            i += 1
            if Kshared_counts[l] == Kshared:
                l += 1
                if random:
                    Kshared = scipy.random.binomial(Ktot, 1. * Ktot / N)
    return Kshared_counts, template


def create_noise_weight_matrix_fixed_pairwise(Nbm, Nnoise, gamma, g, w, epsilon, random_shared=False):
    return create_noise_weight_matrix_fixed_pairwise_fixed_indegree(Nbm, Nnoise, gamma, g, w, int(epsilon * Nnoise), random_shared=random_shared)


def create_noise_weight_matrix_fixed_pairwise_fixed_indegree(Nbm, Nnoise, gamma, g, w, Knoise, random_shared=False):
    """create a random realizations of a weight matrix for Nnoise sources
    projecting to Nbm targets with identity of presynaptic neurons
    shuffled across E/I populations. the number of shared sources
    (KEshared/KIshared) is fixed and equal for each pair of targets.

    """
    NE = int(gamma * Nnoise)
    NI = int(Nnoise - NE)
    KE = int(gamma * Knoise)
    KI = int(Knoise - KE)
    KEshared = 0
    KIshared = 0
    if NE > 0:
        KEshared = int(1. * KE ** 2 / NE)
    if NI > 0:
        KIshared = int(1. * KI ** 2 / NI)
    # check whether it is possible to realize desired connectivity;
    # this translate to (Nbm - 1 ) * epsilon <= 1
    assert(KEshared * (Nbm - 1) <= KE), '[error] impossible parameter choices'
    assert(KIshared * (Nbm - 1) <= KI), '[error] impossible parameter choices'
    W = np.zeros((Nbm, NE + NI))
    for k in xrange(2):
        N = [NE, NI][k]
        K = [KE, KI][k]
        Kshared = [KEshared, KIshared][k]
        wt = [w, -g * w][k]
        if K > 0:
            offset_i = k * NE
            Kshared_offset = np.zeros(Nbm)
            for l in xrange(Nbm):
                Kshared_counts, template = _generate_template(
                    Nbm - l, K - Kshared_offset[l], Kshared, wt, K, N, random_shared)
                W[l:Nbm, offset_i:offset_i + K - Kshared_offset[l]] = template
                offset_i += K - Kshared_offset[l]
                Kshared_offset[l:] += Kshared_counts
    return W


def create_indep_noise_weight_matrix(Nbm, Knoise, gamma, g, w):
    """create a weight matrix for Nbm * Knoise sources projecting to Nbm
    targets with a fixed indegree of Knoise. no shared inputs are
    allowed, hence each target receives uncorrelated input of the
    sources are uncorrelated.

    """

    Nnoise = Nbm * Knoise
    W = np.zeros((Nbm, Nnoise))
    KE = int(gamma * Knoise)
    for l in range(Nbm):
        indE = np.arange(l * Knoise, l * Knoise + KE)
        W[l, indE] = w
        indI = np.arange(l * Knoise + KE, (l + 1) * Knoise)
        W[l, indI] = -g * w
    return W


def get_energy(W, b, s, beta=1.):
    """returns the energy of a state in a boltzmann machine"""
    return -1. * beta * (0.5 * np.dot(s.T, np.dot(W, s)) + np.dot(b, s))


def get_theo_joints(W, b, beta):
    """calculate the theoretical state distribution for a Boltzmann
    machine

    """
    N = len(b)
    joints = []
    states = get_states(N)
    for s in states:
        joints.append(np.exp(-1. * get_energy(W, b, s, beta)))
    joints /= np.sum(joints)
    return joints


def get_theo_joints_multi_bm(W, b, beta, M):
    N = len(b) / M
    joints = []
    for i in range(M):
        joints.append(get_theo_joints(W[i * N:(i + 1) * N, i * N:(i + 1) * N], b[i * N:(i + 1) * N], beta))
    if M == 1:
        return joints[0]
    else:
        return joints


def get_conditionals_from_joints(N, joints, rvs, vals):
    """calculate a conditional distribution
    N: number of random variables
    joints: joint distribution
    rvs: which random variables to condition on
    vals: states of conditioned random variables

    """
    states = get_states(N)
    states_cond = []
    cond = []
    for i, s in enumerate(states):
        if np.all(s[rvs] == vals):
            states_cond.append(s)
            cond.append(joints[i])
    cond = np.array(cond) / np.sum(cond)
    return states_cond, cond


def get_marginals_from_joints(N, joints, rvs):
    """calculate marginal distributions
    N: number of random variables
    joints: joint distribution
    rvs: compute marginals for these random variables

    """
    states = get_states(N)
    m = []
    for i in rvs:
        m.append(np.sum(joints[states[:, i] == 1]))
    return rvs, m


def get_theo_marginals(W, b, beta):
    """calculate marginal distributions of all random variables"""
    N = len(b)
    joints = get_theo_joints(W, b, beta)
    rvs = np.arange(0, N)
    return get_marginals_from_joints(N, joints, rvs)


def get_theo_rates_and_covariances(W, b, beta):
    """calculate rate and covariances for Boltzmann machine from
    connectivity and biases"""
    N = len(b)
    joints = get_theo_joints(W, b, beta)
    states = get_states(N)
    rvs, m = get_theo_marginals(W, b, beta)
    cov = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cov[i, j] = np.sum(joints[np.logical_and(
                states[:, i] == 1, states[:, j] == 1)]) - m[i] * m[j]
    return m, cov


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


def get_steps_warmup(rNrec, Twarmup, tau):
    Nrec = rNrec[1] - rNrec[0]
    assert(Nrec >= 0)
    return int(np.ceil(1. * Nrec * Twarmup / tau))


def get_joints(a_s, steps_warmup, M=1, prior=None):
    """create joint distribution of network states from recorded state
    array. expected array representation of network state."""
    steps_tot = len(a_s[steps_warmup:])
    N = len(a_s[0, :]) / M
    a_joints = np.empty((M, 2 ** N))
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
        else:
            raise NotImplementedError('Unknown prior.')
        for s in a_s[steps_warmup:, i * N:(i + 1) * N]:
            states[tuple(s)] += 1
        states_sorted = np.array([it[1] for it in sorted(states.items())])
        a_joints[i, :] = 1. * states_sorted / steps_tot
        assert((np.sum(a_joints[i, :]) - 1.) < 1e-12)
    if M == 1:
        return a_joints[0]
    else:
        return a_joints


def get_joints_sparse(a_s, N, steps_warmup, prior=None):
    """create joint distribution of network states from recorded state
    array. expected integer representation of network state."""
    if prior is None:
        pass
    elif prior == 'uniform':
        # assuming a uniform prior, i.e., each state is equally
        # likely. we add one realization of each state to the state
        # vector. given no recorded data, this corresponds to a
        # uniform prior.
        for s in get_states(N):
            a_s = np.append(a_s, state_array_to_int(s))
    hist, bins = np.histogram(a_s[steps_warmup:], bins=np.arange(0, 2 ** N + 1), density=True)
    return hist


def get_joints_sparse_multi_bm(a_s, N, M, steps_warmup, prior=None):
    a_s_full = get_all_states_from_sparse(a_s, N * M, steps_warmup)
    joints = get_joints(a_s_full, steps_warmup, M=M, prior=prior)
    return joints


def get_marginals(a_s, steps_warmup):
    """calculate marginals for each unit for a list of states."""
    return np.mean(a_s[steps_warmup:], axis=0)


def get_marginals_multi_bm(a_s, steps_warmup, M):
    N = len(a_s[0, :]) / M
    a_marginals = np.empty((M, N))
    for j in range(M):
        for i in range(N):
            a_marginals[j, :] = get_marginals(a_s[:, j * N:(j + 1) * N], steps_warmup)
    if M == 1:
        return a_marginals[0]
    else:
        return a_marginals


def get_all_states_from_sparse(a_s, N, steps_warmup):
    """create array representation of list of network states from integer
    representation."""
    return np.array([state_array_from_int(s, N) for s in a_s[steps_warmup:]])


def get_euclidean_distance(x, y):
    """calculate the euclidean distance of two vectors."""
    return np.linalg.norm(x - y)


def get_DKL(p, q):
    """returns the Kullback-Leibler divergence of distributions p and q

    """
    assert(np.sum(p) - 1. < 1e-12), 'Distributions must be normalized.'
    assert(np.sum(q) - 1. < 1e-12), 'Distributions must be normalized.'
    assert(np.all(p > 0.)), 'Invalid values in distribution.'
    assert(np.all(q > 0.)), 'Invalid values in distribution.'

    return np.sum(p * np.log(p / q))


def get_DKL_multi_bm(p, q, M):
    assert(np.shape(p) == np.shape(q))
    DKL = []
    for j in range(M):
        DKL.append(get_DKL(p[j], q[j]))
    return DKL


def theta(x):
    """heaviside function."""
    return int(x >= 0)


def Ftheta(x, beta=1.):
    """deterministic activation function (McCulloch-Pitts)"""
    return int(x >= 0)


def sigma(x, beta=1.):
    """sigmoid function"""
    return 1. / (1. + np.exp(-beta * x))


def Fsigma(x, beta=1.):
    """sigmoid activation function (Ginzburg)"""
    return int(1. / (1. + np.exp(-beta * x)) > np.random.rand())


def Ferfc(x, beta=1.):
    """activation function from complementary error function for
    stochastic binary neurons (McCulloch-Pitts + white noise)"""
    return (0.5 * scipy.special.erfc(-1. / beta * x) > np.random.rand())


def sigmainv(y, beta=1.):
    """returns bias b that leads to mean activity y of a stochastic binary
    neuron by inverting sigmoidal activation function

    """
    return 1. / beta * np.log(1. / (1. / y - 1.))


def get_mu_input(epsilon, N, gamma, g, w, mu):
    """returns mean input for given connection statistics and presynaptic
    activity

    """
    return get_mu_input_fixed_indegree(int(epsilon * N), gamma, g, w, mu)


def get_mu_input_fixed_indegree(K, gamma, g, w, mu):
    """returns mean input for given connection statistics and presynaptic
    activity

    """
    return (gamma - (1. - gamma) * g) * K * w * mu


def get_sigma_input(epsilon, N, gamma, g, w, mu):
    """returns standard deviation of input for given connection statistics
    and presynaptic activity

    """
    return get_sigma_input_fixed_indegree(int(epsilon * N), gamma, g, w, mu)


def get_sigma_input_fixed_indegree(K, gamma, g, w, mu):
    """returns standard deviation of input for given connection statistics
    and presynaptic activity

    """
    sigma2 = get_sigma2(mu)
    return np.sqrt((gamma + (1. - gamma) * g ** 2) * K * w ** 2 * sigma2)


def get_adjusted_weights_and_bias(W, b, b_eff, beta_eff, beta):
    """return adjusted weights matrix and bias vector for a Boltzmann
    machine, given the effective offset b_eff, the effective inverse
    temperature beta_eff and the target inverse termperature beta

    """
    return beta / beta_eff * W, beta / beta_eff * b + b_eff


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


def autocorrf(st, tbin, tmax):
    """returns the population averaged autocorrelation function of the
    binned signal st

    """
    Nbins = len(st[0])
    times = np.arange(-Nbins * tbin + tbin, Nbins * tbin, tbin)
    offset_edge = np.hstack(
        [np.arange(1, Nbins), Nbins, np.arange(1, Nbins)[::-1]])
    autof = np.mean(
        [np.correlate(s, s, 'full') / offset_edge - np.mean(s) ** 2 for s in st], axis=0)
    autof = autof[abs(times) <= tmax]
    times = times[abs(times) <= tmax]
    return times, autof


def crosscorrf(times_bin, st, tmax):
    """returns the population averaged crosscorrelation function of the
    binned signal st

    """
    N = len(st)
    # compute autocorrelation of compound signal
    times_cauto, cauto = autocorrf(times_bin, [np.sum(st, axis=0)], tmax)
    # compute marginal autocorrelations
    times_autof, mu_autof = autocorrf(times_bin, st, tmax)
    # calculate cross correlation from compound and marginal autocorrelations
    mu_crossf = 1. / (N * (N - 1)) * (cauto - 1. * N * mu_autof)
    return times_autof, mu_autof, mu_crossf


def get_isi(a_times, a_s):
    """calculate intervals of state transitions (not updates!) for each
    unit."""
    N = len(a_s[0])
    isi = []
    t_last = []
    for i in xrange(N):
        isi.append([])
        t_last.append(0)
    s_last = a_s[0]
    for i, s in enumerate(a_s):
        if np.any(s != s_last):
            pos = np.where(s != s_last)[0][0]
            isi[pos].append(a_times[i] - t_last[pos])
            s_last = s
            t_last[pos] = a_times[i]
    return np.array(isi)


def get_transition_count(a_s, average=False):
    """calculate number of state transitions (not updates!) for each
    unit."""
    N = len(a_s[0])
    if average:
        counts = 0
    else:
        counts = np.zeros(N)
    s_last = a_s[0]
    for i, s in enumerate(a_s):
        if np.any(s != s_last):
            if average:
                counts += 1
            else:
                pos = np.where(s != s_last)[0][0]
                counts[pos] += 1
            s_last = s
    if average:
        counts *= 1. / N
    return counts
