import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


plt.ion()
#plt.rc('font', size=10)

# Order of the gradients in each line of the file (from netRNN.cpp): 
#     gradBP(numsyn) << " " << gradDELTAX(numsyn) << " " << gradDELTAXOP(numsyn) << " " << gradDELTAX31(numsyn) << " " << gradDELTAXCU(numsyn) << " " <<gradDELTAXCUALT(numsyn) << " " << gradDELTAXSIGNSQ(numsyn) << 
#                " " << gradDELTAXSIGNSQRT(numsyn) << " " << gradEH(numsyn)  << " " <<gradNODEPERT(numsyn) <<endl;


plt.figure(figsize=(8, 10))
#plt.figure()
#plt.clf()



z = np.loadtxt('gradsRNN.txt')

NBPLOTS=6

plt.subplots_adjust(wspace=.55, hspace=.3)
plt.subplots_adjust(left=.14)

plt.subplot(3,2,1)
plt.axhline(0); plt.axvline(0)
plt.plot(z[:,-1], z[:,4], 'or')
plt.xlabel('Node-perturb. gradient')
plt.ylabel('Supralinear plasticity gradient\nx^3 (our rule)')
plt.title('a')

plt.subplot(3,2,2)
plt.axhline(0); plt.axvline(0)
plt.plot(z[:,-1], z[:,1], 'or')
plt.xlabel('Node-perturb. gradient')
plt.ylabel('Fluctuations gradient\n(E-H without real-time reward)')
plt.title('b')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.subplot(3,2,3)
plt.axhline(0); plt.axvline(0)
plt.plot(z[:,-1], z[:,3], 'or')
plt.xlabel('Node-perturb. gradient')
plt.ylabel('Fluctuations gradient \n(10ms after perturbation only)')
plt.title('c')


plt.subplot(3,2,4)
plt.axhline(0); plt.axvline(0)
plt.plot(z[:,-1], z[:,-2], 'or')
plt.xlabel('Node-perturb. gradient')
plt.ylabel('Full E-H rule\n(with real-time reward)');
plt.title('d')


plt.subplot(3,2,5)
plt.axhline(0); plt.axvline(0)
plt.plot(z[:,-1], z[:,6], 'or')
plt.xlabel('Node-perturb. gradient')
plt.ylabel('Supralinear plasticity gradient\nx*|x|')
plt.title('e')


plt.subplot(3,2,6)
plt.axhline(0); plt.axvline(0)
plt.plot(z[:,-1], z[:,7], 'or')
plt.xlabel('Node-perturb. gradient')
plt.ylabel('Sub-linear plasticity gradient\nsqrt(x)');
plt.title('f')

plt.draw()
plt.savefig('figure_gradientsRNN.png', bbox_inches='tight', dpi=300)

