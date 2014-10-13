import numpy as np
import heapq as hq

def simulate(W, b, sinit, steps, Nrec, l_N, l_F, record_ui=False, Nrec_ui=None, beta=1.):
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
        s[idx] = l_F[idF](ui, beta)
        if record_ui:
            a_rec_ui[step] = np.dot(W[:Nrec_ui,:], s) + b[:Nrec_ui]
        a_s[step] = s[:Nrec]
        step += 1
    a_steps = np.arange(steps)
    if record_ui:
        return a_steps, a_s, a_rec_ui
    else:
        return a_steps, a_s


def simulate_eve(W, b, tau, sinit, time, Nrec, l_N, l_F, Nrec_ui=0, beta=1.):
    record_s = True if Nrec > 0 else False
    record_ui = True if Nrec_ui > 0 else False
    N = len(b)
    maxsteps = int(np.ceil(1.*N*time/tau))
    s = sinit
    step = 1
    maxrelsteps = int(np.ceil(1.1*Nrec*time/tau))
    relstep = 1
    a_s = np.empty((maxrelsteps, Nrec))
    a_steps = np.zeros(maxrelsteps)
    if record_s:
        a_s[0] = s[:Nrec]
        a_steps[0] = 0.
    if record_ui:
        maxrelsteps_ui = int(np.ceil(1.1*Nrec_ui*time/tau))
        relstep_ui = 1
        a_ui = np.empty((maxrelsteps_ui, Nrec_ui))
        a_steps_ui = np.zeros(maxrelsteps_ui)
        a_ui[0] = np.dot(W[:Nrec_ui,:], s) + b[:Nrec_ui]
        a_steps_ui[0] = 0.
    updates = list(zip(np.random.exponential(tau, N), np.random.permutation(np.arange(0, N))))
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
            a_ui[relstep_ui] = np.dot(W[:Nrec_ui,:], s) + b[:Nrec_ui]
            a_steps_ui[relstep_ui] = time
            relstep_ui += 1
        ui = np.dot(W[idx,:], s) + b[idx]
        s[idx] = l_F[idF](ui, beta)
        if record_s and idx < Nrec:
            a_s[relstep] = s[:Nrec]
            a_steps[relstep] = time
            relstep += 1
        hq.heappush(updates, (time+np.random.exponential(tau), idx))
        step += 1
    if record_s:
        maxpos = np.where(a_steps > 0.)[0][-1]
        a_s = a_s[:maxpos,:]
        a_steps = a_steps[:maxpos]
    if record_ui:
        maxpos_ui = np.where(a_steps_ui > 0.)[0][-1]
        a_ui = a_ui[:maxpos_ui,:]
        a_steps_ui = a_steps_ui[:maxpos_ui]
        return a_steps, a_s, a_steps_ui, a_ui
    else:
        return a_steps, a_s
