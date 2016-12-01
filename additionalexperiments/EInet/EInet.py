# Trying to implement  E-I nets with positive output functions and Dale's law, following Mastrogiuseppe & Ostojic.


# This version has random weights, is fully dense, and SOMETIMES has permanent dynamics while non-saturating (no value reaches the maximum of 20.0) ! 
# The mean values of xs and ys don't really match the theory...?
# If you use sparse-ish connections (even 50%) with identical weights (1.0 * J), theory is matched much better and always in the dynamic, non-saturating regime ! 
# Dense connectivity with large network (800 = 2*400) is more reliable in producing the expected non-saturating chaotic regime. Theory still not observed - mean(xs) consistently lower than gamma, which shouldn't happen.. If you increase gamma, the mean xs just dips even lower...

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

NBNEUR = 200
NBI = 100
NBE = NBNEUR - NBI
#CE = NBE; CI = NBI; g = 1.1 # Dense !
#CE = NBE; CI = NBI; g = 1.05 # Dense !
CE = 50; CI = 50; g = 1.2 # Sparse-ish
J = .2

w = np.ones((NBNEUR, NBNEUR))

# Making the exitatory weights
wE = []
for nn in xrange(NBNEUR):
    wv = np.zeros((1, NBE))
    wv[0,:CE] = 1.0
    #wv[0,:CE] = .5 + 1.5 * np.random.rand(CE) 
    #wv[0,:CE] = 2.0 * np.random.rand(CE) 
    np.random.shuffle(wv[0])
    wE.append(wv)
wE = np.vstack(wE)

# Making the inhibitory weights 
wI = []
for nn in xrange(NBNEUR):
    wv = np.zeros((1, NBI))
    wv[0,:CI] = 1.0
    #wv[0,:CI] = .5 + 1.5 * np.random.rand(CI)
    #wv[0,:CI] = 2.0 * np.random.rand(CI)
    np.random.shuffle(wv[0])
    wI.append(wv)
wI = np.vstack(wI)

wI *= -g

w = np.hstack((wE, wI))
w *= J

x = .1 * np.random.randn(NBNEUR)
y = x.copy(); y[y<0] = 0

NBSTEPS = 10000
TAU = 30.0

ys=[]; xs=[]
gamma = 2.0
for numstep in xrange(NBSTEPS):
    x = x + (w.dot(y) - x) / TAU
    y[:] = x
    y[x<-gamma] = 0
    y[x>-gamma] = x[x>-gamma] + gamma
    y[x>20-gamma] = 20
    ys.append(y.copy())
    xs.append(x.copy())

ys = np.vstack(ys)
xs = np.vstack(xs)
#z = ys[::5, :10]
z = ys[:, :10]
plt.clf()
plt.plot(z)
plt.draw()
