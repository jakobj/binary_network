import numpy as np
import heapq as hq

def simulate(W, b, sinit, steps, Nrec, l_N, l_F, record_ui=False, Nrec_ui=None):
    N = len(b)
    s = sinit
    step = 1
    a_s = np.empty((int(steps), Nrec))
    a_s[0] = s[:Nrec]
    if record_ui:
        a_rec_ui = np.empty((int(steps), Nrec_ui))
        a_rec_ui[0] = np.zeros(Nrec_ui)
    while step < steps:
        idx = np.random.randint(0, N)
        idF = 0
        for Ni in l_N:
            if idx < Ni:
                break
            else:
               idF += 1
        ui = np.dot(W[idx,:], s) + b[idx]
        s[idx] = l_F[idF](ui)
        if record_ui:
            a_rec_ui[step] = np.dot(W[:Nrec_ui,:], s) + b[:Nrec_ui]
        a_s[step] = s[:Nrec]
        step += 1
    a_steps = np.arange(steps)
    if record_ui:
        return a_steps, a_s, a_rec_ui
    else:
        return a_steps, a_s


def simulate_eve(W, b, tau, sinit, time, Nrec, l_N, l_F, record_ui=False, Nrec_ui=None):
    N = len(b)
    steps = np.ceil(1.*N*time/tau)
    s = sinit
    step = 1
    a_s = np.empty((int(steps), Nrec))
    a_s[0] = s[:Nrec]
    a_steps = np.empty(int(steps))
    a_steps[0] = 0.
    if record_ui:
        a_rec_ui = np.empty((int(steps), Nrec_ui))
        a_rec_ui[0] = np.zeros(Nrec_ui)
    updates = list(zip(np.random.exponential(tau, N), np.random.permutation(np.arange(0, N))))
    hq.heapify(updates)
    while step < steps:
        time, idx = hq.heappop(updates)
        idF = 0
        for Ni in l_N:
            if idx < Ni:
                break
            else:
               idF += 1
        if record_ui:
            a_rec_ui[step] = np.dot(W[:Nrec_ui,:], s) + b[:Nrec_ui]
        ui = np.dot(W[idx,:], s) + b[idx]
        s[idx] = l_F[idF](ui)
        a_s[step] = s[:Nrec]
        a_steps[step] = time
        hq.heappush(updates, (time+np.random.exponential(tau), idx))
        step += 1
    if record_ui:
        return a_steps, a_s, a_rec_ui
    else:
        return a_steps, a_s
