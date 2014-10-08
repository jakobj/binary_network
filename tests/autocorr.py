import numpy as np
import matplotlib.pyplot as plt

import correlation_toolbox.correlation_analysis as ctana

import network as bnet
import helper as hlp

np.random.seed(1234)

def crosscorrf(times, a_s, tmax, tbin):
    times_bin, st = hlp.bin_binary_data(times, a_s, tbin)
    freq, cross = ctana.crossspec(st, tbin, units=True)
    return ctana.crosscorrfunc(freq, cross)

N = 60
sinit = np.zeros(N)
tau = 10.
Nrec = 40
time = 2e3
mu_target = 0.5
tmax = 300.
tbin = .6

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

timelag_brn, autof_brn = autocorrf_time(a_times_brn, a_s_brn, tmax, tbin)
timelag_brn_alt, autof_brn_alt = autocorrf(a_times_brn, a_s_brn, tmax, tbin)
timelag_noise, autof_noise = autocorrf_time(a_times_noise, a_s_noise, tmax, tbin)
timelag_noise_alt, autof_noise_alt = autocorrf(a_times_noise, a_s_noise, tmax, tbin)

timelag_brn_cross_alt, crossf_brn_alt = crosscorrf(a_times_brn, a_s_brn, tmax, tbin)
timelag_noise_cross_alt, crossf_noise_alt = crosscorrf(a_times_noise, a_s_noise, tmax, tbin)

offset_brn_alt = r_brn**2*1./tbin*1e3
offset_noise_alt = r_noise**2*1./tbin*1e3

autof_brn -= offset_brn
autof_brn_alt -= offset_brn_alt
autof_noise -= offset_noise
autof_noise_alt -= offset_noise_alt

var_brn = autof_brn[abs(timelag_brn) < 1e-8]
var_brn_alt = np.mean(autof_brn_alt[abs(timelag_brn_alt) < 0.9*tbin])
var_noise = autof_noise[abs(timelag_noise) < 1e-8]
var_noise_alt = np.mean(autof_brn_alt[abs(timelag_brn_alt) < 0.9*tbin])

autof_brn = 1.*(autof_brn)/var_brn
autof_brn_alt = 1.*(autof_brn_alt)/var_brn_alt
autof_noise = 1.*(autof_noise)/var_noise
autof_noise_alt = 1.*(autof_noise_alt)/var_noise_alt

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.plot(timelag_brn, 1.*autof_brn, 'b', label='BRN (via time)')
ax1.plot(timelag_brn_alt, 1.*autof_brn_alt, 'g', label='BRN (via FT)')
ax1.plot(timelag_noise, 1.*autof_noise, 'r', label='\"Poisson\"')
ax1.plot(timelag_noise_alt, 1.*autof_noise_alt, 'm', label='\"Poisson\" (via FT)')
ax1.plot(timelag_noise, np.exp(-1.*abs(timelag_noise)/tau), 'k', label=r'$e^{-t/\tau}$')
ax1.set_xlim([-200., 200.])
ax1.set_xlabel('Timelag (ms)')
ax1.set_ylabel('Autocorrelation (1/s)')
ax1.legend()

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.plot(timelag_brn_cross_alt, crossf_brn_alt-offset_brn_alt, 'g', label='BRN')
ax2.plot(timelag_noise_cross_alt, crossf_noise_alt-offset_noise_alt, 'm', label='\"Poisson\"')
ax2.set_xlim([-200., 200.])
ax2.set_xlabel('Timelag (ms)')
ax2.set_ylabel('Crosscorrelation (1/s)')
ax2.legend()
plt.show()
