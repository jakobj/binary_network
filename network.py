# global imports
import numpy as np
import heapq as hq

# local imports
from . import helper as bhlp


def simulate(W, b, sinit, steps, Nrec, l_N, l_F, Nrec_ui=0, beta=1.):
    record_ui = True if Nrec_ui > 0 else False
    N = len(b)
    s = sinit
    step = 1
    a_s = np.empty((int(steps), Nrec))
    a_s[0] = s[:Nrec]
    if record_ui:
        maxrelsteps_ui = int(np.ceil(1.1 * Nrec_ui * steps))
        relstep_ui = 1
        a_rec_ui = np.empty((maxrelsteps_ui, Nrec_ui))
        a_steps_ui = np.zeros(maxrelsteps_ui)
        a_rec_ui[0] = np.dot(W[:Nrec_ui, :], s) + b[:Nrec_ui]
        a_steps_ui[0] = 0.

    while step < steps:
        idx = np.random.randint(0, N)
        idF = 0
        for Ni in l_N:
            if idx < Ni:
                break
            else:
                idF += 1
        ui = np.dot(W[idx, :], s) + b[idx]
        s[idx] = l_F[idF](ui, beta)
        if record_ui and idx < Nrec_ui:
            a_rec_ui[relstep_ui] = np.dot(W[:Nrec_ui, :], s) + b[:Nrec_ui]
            a_steps_ui[relstep_ui] = step
            relstep_ui += 1
        a_s[step] = s[:Nrec]
        step += 1
    a_steps = np.arange(steps)
    if record_ui:
        maxpos_ui = np.where(a_steps_ui > 0.)[0][-1]
        a_rec_ui = a_rec_ui[:maxpos_ui, :]
        a_steps_ui = a_steps_ui[:maxpos_ui]
        return a_steps, a_s, a_steps_ui, a_rec_ui
    else:
        return a_steps, a_s


def simulate_eve(W, b, tau, sinit, time, rNrec, l_N, l_F, Nrec_ui=0, beta=1.):
    Nrec = rNrec[1] - rNrec[0]
    record_s = True if Nrec > 0 else False
    record_ui = True if Nrec_ui > 0 else False
    N = len(b)
    print '[binary_network] Simulating %d nodes.' % (N)
    maxsteps = int(np.ceil(1. * N * time / tau))
    s = sinit
    step = 1
    maxrelsteps = int(np.ceil(1.3 * Nrec * time / tau))
    relstep = 1
    a_s = np.empty((maxrelsteps, Nrec))
    a_steps = np.zeros(maxrelsteps)
    if record_s:
        a_s[0] = s[rNrec[0]:rNrec[1]]
        a_steps[0] = 0.
    if record_ui:
        maxrelsteps_ui = int(np.ceil(1.3 * Nrec_ui * time / tau))
        relstep_ui = 1
        a_ui = np.empty((maxrelsteps_ui, Nrec_ui))
        a_steps_ui = np.zeros(maxrelsteps_ui)
        a_ui[0] = np.dot(W[:Nrec_ui, :], s) + b[:Nrec_ui]
        a_steps_ui[0] = 0.
    updates = list(zip(np.random.exponential(tau, N),
                       np.random.permutation(np.arange(0, N))))
    hq.heapify(updates)
    while step < maxsteps:
        time, idx = hq.heappop(updates)
        idF = 0
        for Ni in l_N:
            if idx < Ni:
                break
            else:
                idF += 1
        if record_ui and idx < Nrec_ui:
            a_ui[relstep_ui] = np.dot(W[:Nrec_ui, :], s) + b[:Nrec_ui]
            a_steps_ui[relstep_ui] = time
            relstep_ui += 1
        ui = np.dot(W[idx, :], s) + b[idx]
        s[idx] = l_F[idF](ui, beta)
        if record_s and idx >= rNrec[0] and idx < rNrec[1]:
            a_s[relstep] = s[rNrec[0]:rNrec[1]]
            a_steps[relstep] = time
            relstep += 1
        hq.heappush(updates, (time + np.random.exponential(tau), idx))
        step += 1
    if record_s:
        maxpos = np.where(a_steps > 0.)[0][-1]
        a_s = a_s[:maxpos, :]
        a_steps = a_steps[:maxpos]
    if record_ui:
        maxpos_ui = np.where(a_steps_ui > 0.)[0][-1]
        a_ui = a_ui[:maxpos_ui, :]
        a_steps_ui = a_steps_ui[:maxpos_ui]
        return a_steps, a_s, a_steps_ui, a_ui
    else:
        return a_steps, a_s


def simulate_eve_sparse(W, b, tau, s_init, Tmax, rNrec, l_N, l_F, beta=1., rNrec_u=[0, 0]):
    """
    simulate a network of binary neurons
    (event driven, sparse recording of states)
    W: connectivity matrix
    b: bias vector
    tau: mean update interval of units
    sinit: initial state
    time: simulation duration
    rNrec: list definig start and end unit for recording states
    l_N: list defining range of activation functions in l_F
    l_F: list of activation function for units in range l_N
    beta: inverse temperature
    """
    assert(len(l_N) == len(l_F))
    Nrec = rNrec[1] - rNrec[0]
    assert(Nrec >= 0)
    Nrec_u = rNrec_u[1] - rNrec_u[0]
    assert(Nrec_u >= 0)

    N = len(b)
    s = s_init.copy()
    maxsteps = int(np.ceil(1. * N * Tmax / tau))

    # set up recording array for states
    mean_recsteps = Nrec * Tmax / tau
    max_recsteps = int(np.ceil(mean_recsteps + 3 * np.sqrt(mean_recsteps)))  # Poisson process, mean + 3 * std
    a_s = np.empty(max_recsteps, dtype=int)
    a_times = np.zeros(max_recsteps)
    recstep = 0
    # set up recording arrays for membrane potential
    mean_recsteps_u = Nrec_u * Tmax / tau
    max_recsteps_u = int(np.ceil(mean_recsteps_u + 3 * np.sqrt(mean_recsteps_u)))  # Poisson process, mean + 3 * std
    a_u = np.empty((max_recsteps_u, Nrec_u), dtype=int)
    a_times_u = np.zeros(max_recsteps_u)
    recstep_u = 0

    # build lookup table for activation functions and recording
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
        if idx >= rNrec_u[0] and idx < rNrec[1]:
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
            a_s[recstep] = bhlp.state_array_to_int(s[rNrec[0]:rNrec[1]])
            a_times[recstep] = time
            recstep += 1
        hq.heappush(updates, (time + np.random.exponential(tau), idx))

    if Nrec_u == 0:
        return a_times[:recstep], a_s[:recstep]
    else:
        return a_times[:recstep], a_s[:recstep], a_times_u[:recstep_u], a_u[:recstep_u]


def simulate_eve_sparse_stim(W, b, tau, sinit, time, rNrec, l_N, l_F, l_pattern, beta=1.):
    Nrec = rNrec[1] - rNrec[0]
    assert(Nrec > 0)
    N = len(b)
    print '[binary_network] Simulating %d nodes.' % (N)
    maxsteps = int(np.ceil(1. * N * time / tau))
    s = sinit.copy()
    step = 0
    maxrelsteps = int(np.ceil(1.3 * Nrec * time / tau))
    relstep = 0
    a_s = np.empty((maxrelsteps, 2))
    a_steps = np.zeros(maxrelsteps)
    updates = list(zip(np.random.exponential(tau, N),
                       np.random.permutation(np.arange(0, N))))
    hq.heapify(updates)
    pattern_pos = 0
    while step < maxsteps:
        time, idx = hq.heappop(updates)
        if pattern_pos < len(l_pattern) and time > l_pattern[pattern_pos][0]:
            b[l_pattern[pattern_pos][1][0]:l_pattern[
                pattern_pos][1][1]] = l_pattern[pattern_pos][2]
            pattern_pos += 1
        idF = 0
        for Ni in l_N:
            if idx < Ni:
                break
            else:
                idF += 1
        ui = np.dot(W[idx, :], s) + b[idx]
        s[idx] = l_F[idF](ui, beta)
        if idx >= rNrec[0] and idx < rNrec[1]:
            a_s[relstep, :] = [idx, s[idx]]
            a_steps[relstep] = time
            relstep += 1
        hq.heappush(updates, (time + np.random.exponential(tau), idx))
        step += 1
    maxpos = np.where(a_steps > 0.)[0][-1]
    a_s = a_s[:maxpos, :]
    a_steps = a_steps[:maxpos]
    return sinit[rNrec[0]:rNrec[1]], a_steps, a_s
