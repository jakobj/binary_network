import numpy as np
import matplotlib.pyplot as plt

import correlation_toolbox.correlation_analysis as ctana

import network as bnet
import helper as hlp

def autocorrf_time(times, a_s, tmax, tbin):
    st = []
    for s in a_s:
        st.append(times[s>0])
    return ctana.autocorrfunc_time(st, tmax, tbin, times[-1], units=True)

def autocorrf(times, a_s, tmax, tbin):
    times_bin = np.arange(0., np.max(times), tbin)
    T = len(times_bin)
    st = np.zeros((len(a_s), T))
    for j,s in enumerate(a_s):
        # tup = times[s>0]
        for i in range(len(st[j])):
            tc = i*tbin
            idl = np.where(times <= tc)[0]
            if len(idl) > 0:
                st[j][i] = s[idl[-1]]
            else:
                pass

    freq, power = ctana.powerspec(st, tbin, units=True)
    return ctana.autocorrfunc(freq, power)

N = 50
sinit = np.zeros(N)
tau = 10.
Nrec = 30
time = 2e3
mu_target = 0.3
tmax = 200.

w = 0.2
g = 8.
gamma = 0.
epsilon = 0.1

W_brn = hlp.create_connectivity_matrix(N, w, g, epsilon, gamma)
b_brn = -1.*hlp.get_mun(epsilon*N, gamma, g, w, mu_target)*np.ones(N)-1.*w/2
a_times_brn, a_s_brn = bnet.simulate_eve(W_brn, b_brn, tau, sinit.copy(), time, Nrec, [N], [hlp.theta])

r_brn = np.mean(a_s_brn)
print 'brn', r_brn
offset_brn = N*1./tau*1e3*r_brn**2

W_noise = np.zeros((N, N))
b_noise = np.ones(N)*hlp.sigmainv(mu_target)
a_times_noise, a_s_noise = bnet.simulate_eve(W_noise, b_noise, tau, sinit.copy(), time, Nrec, [N], [hlp.Fsigma])

r_noise = np.mean(a_s_noise)
print 'noise', r_noise
offset_noise = N*1./tau*1e3*r_noise**2

timelag_brn, autof_brn = autocorrf_time(a_times_brn, a_s_brn.T, tmax, 1.)
timelag_brn_alt, autof_brn_alt = autocorrf(a_times_brn, a_s_brn.T, tmax, 1.)
timelag_noise, autof_noise = autocorrf_time(a_times_noise, a_s_noise.T, tmax, 1.)

offset_brn_alt = np.mean(autof_brn_alt[:15])

var_brn = autof_brn[abs(timelag_brn) < 1e-8]
var_brn_alt = np.mean(autof_brn_alt[abs(timelag_brn_alt) < 0.9*1.])
var_noise = autof_noise[abs(timelag_noise) < 1e-8]

autof_brn = 1.*(autof_brn)/var_brn
autof_brn_alt = 1.*(autof_brn_alt)/var_brn_alt
autof_noise = 1.*(autof_noise)/var_noise

plt.plot(timelag_brn, 1.*autof_brn, 'b')
plt.plot(timelag_brn_alt, 1.*autof_brn_alt, 'g')
plt.plot(timelag_noise, 1.*autof_noise, 'r')
plt.plot(timelag_noise, np.exp(-1.*abs(timelag_noise)/tau), 'k')
plt.show()


