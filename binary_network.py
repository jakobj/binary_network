import numpy as np

def simulate(W, b, sinit, steps, Nrec, l_N, l_F, calibrate=False):
    N = len(b)
    s = sinit
    step = 1
    if calibrate:
        a_act = np.empty(steps)
        a_act[0] = np.sum(s)
    else:
        a_s = np.empty((steps, Nrec))
        a_s[0] = s
    while step < steps:
        idx = np.random.randint(0, N)
        idF = 0
        for Ni in l_N:
            if idx < Ni:
                break
            else:
               idF += 1
        s[idx] = l_F[idF](np.dot(W[idx,:], s) + b[idx])
        if calibrate:
            a_act[step] = np.sum(s)
        else:
            a_s[step] = s[:Nrec]
        step += 1
    if calibrate:
        return a_act
    else:
        a_steps = np.arange(steps)
        return a_steps, a_s
