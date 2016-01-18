# global imports
import numpy as np
import heapq as hq


def simulate(W, b, s_init, steps, rNrec, l_N, l_F, beta=1., rNrec_u=[0, 0]):
    """
    simulate a network of binary neurons (step driven)
    W: connectivity matrix
    b: bias vector
    s_init: initial state
    steps: simulation steps
    rNrec: list defining start (incl) and end unit (excl) for recording states
    l_N: list defining range of activation functions in l_F
    l_F: list of activation function for units in range l_N
    beta: inverse temperature
    rNrec_u: list defining start (incl) and end unit (excl) for recording membrane potentials
    """
    assert(len(l_N) == len(l_F))
    Nrec = rNrec[1] - rNrec[0]
    assert(Nrec >= 0)
    Nrec_u = rNrec_u[1] - rNrec_u[0]
    assert(Nrec_u >= 0)

    N = len(b)
    s = s_init.copy()

    # set up recording array for states
    mean_recsteps = Nrec * steps / N
    max_recsteps = int(np.ceil(mean_recsteps + 3 * np.sqrt(mean_recsteps)))  # Poisson process, mean + 3 * std
    a_s = np.empty((max_recsteps, Nrec), dtype=int)
    a_steps = np.empty(max_recsteps)
    recstep = 0
    # set up recording arrays for membrane potential
    mean_recsteps_u = Nrec_u * steps / N
    max_recsteps_u = int(np.ceil(mean_recsteps_u + 3 * np.sqrt(mean_recsteps_u)))  # Poisson process, mean + 3 * std
    a_u = np.empty((max_recsteps_u, Nrec_u))
    a_steps_u = np.zeros(max_recsteps_u)
    recstep_u = 0

    # build lookup tables for activation functions and recording
    F_lut = np.empty(N, dtype=int)
    rec_s_lut = np.zeros(N, dtype=bool)
    rec_u_lut = np.zeros(N, dtype=bool)
    idF = 0
    for idx in xrange(N):
        if idx >= l_N[idF]:
            idF += 1
        F_lut[idx] = idF
        if idx >= rNrec[0] and idx < rNrec[1]:
            rec_s_lut[idx] = True
        if idx >= rNrec_u[0] and idx < rNrec_u[1]:
            rec_u_lut[idx] = True

    # simulation loop
    print '[binary_network] Simulating %d nodes.' % (N)
    for step in xrange(steps):
        idx = np.random.randint(0, N)
        ui = np.dot(W[idx, :], s) + b[idx]
        if Nrec_u > 0 and rec_u_lut[idx]:
            a_u[recstep_u] = np.dot(W[rNrec_u[0]:rNrec_u[1], :], s) + b[rNrec_u[0]:rNrec_u[1]]
            a_steps_u[recstep_u] = step
            recstep_u += 1
        s[idx] = l_F[F_lut[idx]](ui, beta)
        if Nrec > 0 and rec_s_lut[idx]:
            a_s[recstep] = s[rNrec[0]:rNrec[1]]
            a_steps[recstep] = step
            recstep += 1

    if Nrec_u == 0:
        return a_steps[:recstep], a_s[:recstep]
    else:
        return a_steps[:recstep], a_s[:recstep], a_steps_u[:recstep_u], a_u[:recstep_u]


def simulate_eve_sparse(W, b, tau, s_init, Tmax, rNrec, l_N, l_F, beta=1., rNrec_u=[0, 0]):
    """
    simulate a network of binary neurons (event driven, sparse recording of states)
    W: connectivity matrix
    b: bias vector
    tau: mean update interval of units
    s_init: initial state
    time: simulation duration
    rNrec: list defining start (incl) and end unit (excl) for recording states
    l_N: list defining range of activation functions in l_F
    l_F: list of activation function for units in range l_N
    beta: inverse temperature
    rNrec_u: list defining start (incl) and end unit (excl) for recording membrane potentials
    """
    assert(len(l_N) == len(l_F))
    Nrec = rNrec[1] - rNrec[0]
    assert(Nrec >= 0)
    Nrec_u = rNrec_u[1] - rNrec_u[0]
    assert(Nrec_u >= 0)

    N = len(b)
    s = np.array(s_init, dtype=np.uint8)
    maxsteps = int(np.ceil(1. * N * Tmax / tau))

    # set up recording array for states
    mean_recsteps = Nrec * Tmax / tau
    max_recsteps = int(np.ceil(mean_recsteps + 3 * np.sqrt(mean_recsteps)))  # Poisson process, mean + 3 * std
    # record states using  np.packbits, need to select the right size and data type
    a_s = np.empty((max_recsteps, np.ceil(N / 8.)), dtype=np.uint8)
    a_times = np.zeros(max_recsteps)
    recstep = 0
    # set up recording arrays for membrane potential
    mean_recsteps_u = Nrec_u * Tmax / tau
    max_recsteps_u = int(np.ceil(mean_recsteps_u + 3 * np.sqrt(mean_recsteps_u)))  # Poisson process, mean + 3 * std
    a_u = np.empty((max_recsteps_u, Nrec_u))
    a_times_u = np.zeros(max_recsteps_u)
    recstep_u = 0

    # build lookup tables for activation functions and recording
    F_lut = np.empty(N, dtype=int)
    rec_s_lut = np.zeros(N, dtype=bool)
    rec_u_lut = np.zeros(N, dtype=bool)
    idF = 0
    for idx in xrange(N):
        if idx >= l_N[idF]:
            idF += 1
        F_lut[idx] = idF
        if idx >= rNrec[0] and idx < rNrec[1]:
            rec_s_lut[idx] = True
        if idx >= rNrec_u[0] and idx < rNrec_u[1]:
            rec_u_lut[idx] = True

    # choose initial update times
    updates = list(zip(np.random.exponential(tau, N),
                       np.random.permutation(np.arange(0, N, dtype=int))))
    hq.heapify(updates)

    # simulation loop
    print '[binary_network] Simulating %d nodes.' % (N)
    for _ in xrange(maxsteps):
        time, idx = hq.heappop(updates)
        ui = np.dot(W[idx, :], s) + b[idx]
        if Nrec_u > 0 and rec_u_lut[idx]:
            a_u[recstep_u] = np.dot(W[rNrec_u[0]:rNrec_u[1], :], s) + b[rNrec_u[0]:rNrec_u[1]]
            a_times_u[recstep_u] = time
            recstep_u += 1
        s[idx] = l_F[F_lut[idx]](ui, beta)
        if Nrec > 0 and rec_s_lut[idx]:
            a_s[recstep] = np.packbits(s)
            a_times[recstep] = time
            recstep += 1
        hq.heappush(updates, (time + np.random.exponential(tau), idx))

    if Nrec_u == 0:
        return a_times[:recstep], a_s[:recstep]
    else:
        return a_times[:recstep], a_s[:recstep], a_times_u[:recstep_u], a_u[:recstep_u]
