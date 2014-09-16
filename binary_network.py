import numpy as np

def simulate(W, b, sinit, steps, Nrec, l_N, l_F, calibrate=False):
    N = len(b)
    s = sinit
    step = 1
    a_s = np.empty((steps, Nrec))
    a_s[0] = s[:Nrec]
    while step < steps:
        idx = np.random.randint(0, N)
        idF = 0
        for Ni in l_N:
            if idx < Ni:
                break
            else:
               idF += 1
        s[idx] = l_F[idF](np.dot(W[idx,:], s) + b[idx])
        a_s[step] = s[:Nrec]
        step += 1
    a_steps = np.arange(steps)
    return a_steps, a_s
