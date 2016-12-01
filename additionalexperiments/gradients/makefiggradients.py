import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


plt.ion()
#plt.rc('font', size=10)

# Order of the gradients: gradBP(numsyn) << " " << gradDELTAX(numsyn) << " " << gradDELTAXOP(numsyn) << " " << gradDELTAX15(numsyn) << " " << gradDELTAXCU(numsyn) << " " <<gradDELTAXCUALT(numsyn) << " " <<gradNODEPERT(numsyn) <<endl; 

#plt.figure(figsize=(5, 4))
plt.figure()
plt.clf()



z = np.loadtxt('grads.txt')

NBPLOTS=4

plt.subplots_adjust(wspace=.55, hspace=.3)
plt.subplots_adjust(left=.14)

plt.subplot(2,2,1)
plt.axhline(0); plt.axvline(0)
plt.xlabel('Backprop gradient')
plt.plot(z[:,0], z[:,-1], 'or')
plt.ylabel('Node-perturb. gradient')
plt.title('a')

plt.subplot(2,2,2)
plt.axhline(0); plt.axvline(0)
plt.xlabel('Backprop gradient')
plt.plot(z[:,0], z[:,1], 'or')
plt.ylabel('Fluctuations gradient')
plt.title('b')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.subplot(2,2,3)
plt.axhline(0); plt.axvline(0)
plt.xlabel('Backprop gradient')
plt.plot(z[:,0], z[:,3], 'or')
plt.ylabel('Fluctuations gradient \n(30ms after perturbation only)')
plt.title('c')

#plt.subplot(2,2,1)
#plt.axhline(0); plt.axvline(0)
#plt.xlabel('Backprop gradient')
#plt.plot(z[:,0], z[:,4], 'or')
#plt.ylabel('Supralinear fluctuations gradient')


plt.subplot(2,2,4)
plt.axhline(0); plt.axvline(0)
plt.xlabel('Backprop gradient')
plt.plot(z[:,0], z[:,4], 'or')
plt.ylabel('Supralinear plasticity gradient')
plt.title('d')

plt.draw()
plt.savefig('figure_gradients.png', bbox_inches='tight', dpi=300)

