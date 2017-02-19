# You must run netFF.cpp to generate the files that this program uses!

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


plt.ion()
plt.rc('font', size=10)

plt.figure(figsize=(4.5, 3.5))

x = np.loadtxt('xsnonoise1.txt')[100:]
xt = np.loadtxt('xtraceslongnonoise1.txt')[100:]

plt.subplot(2,1,1)
#plt.plot(x, 'k', label=r'$x$')
plt.plot(x, 'k', linewidth=2, label=r'$y$')
plt.plot(xt, 'k:', linewidth=2, label=r'$\overline{y}$')
plt.legend()

plt.subplot(2,1,2)
plt.fill_between(range(x.size),  0, x-xt, alpha=.25, facecolor='gray')
plt.plot(np.zeros(x.shape), 'k--')
plt.plot(x - xt, 'r', linewidth=2, label=r'$y - \overline{y}$')
plt.text(20, .5, 'Perturbation effect')
plt.arrow(50, .45, 50, -.27,  head_width=0.05, head_length=3.0, fc='k', ec='k', width=.01)
plt.text(120, .5, 'Relaxation effect')
plt.arrow(150, .45, -27, -.47,head_width=0.05, head_length=3.0 )
plt.legend()


plt.draw()
plt.savefig('figure_relax.png', bbox_inches='tight', dpi=300)
