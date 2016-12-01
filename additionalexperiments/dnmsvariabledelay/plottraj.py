# -*- coding: utf-8 -*-


from pylab import *
import numpy as nps
import scipy as sp
import glob

ion()

rlist = []
rlist.append(loadtxt('rs_long_RANDW_type0_0_G1.500000_MAXDW0.000300_ETA0.030000_ALPHAMODUL16.000000_PROBAMODUL0.003000_SQUARING1_MODULTYPE-DECOUPLED_ALPHATRACE0.500000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.050000_TAU30.000000_NDUPL5.000000_RNGSEED1.txt')[::10,:])
rlist.append(loadtxt('rs_long_type0_0_G1.500000_MAXDW0.000300_ETA0.030000_ALPHAMODUL16.000000_PROBAMODUL0.003000_SQUARING1_MODULTYPE-DECOUPLED_ALPHATRACE0.500000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.050000_TAU30.000000_NDUPL5.000000_RNGSEED1.txt')[::10,:])
rlist.append(loadtxt('rs_long_RANDW_type1_0_G1.500000_MAXDW0.000300_ETA0.030000_ALPHAMODUL16.000000_PROBAMODUL0.003000_SQUARING1_MODULTYPE-DECOUPLED_ALPHATRACE0.500000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.050000_TAU30.000000_NDUPL5.000000_RNGSEED1.txt')[::10,:])
rlist.append(loadtxt('rs_long_type1_0_G1.500000_MAXDW0.000300_ETA0.030000_ALPHAMODUL16.000000_PROBAMODUL0.003000_SQUARING1_MODULTYPE-DECOUPLED_ALPHATRACE0.500000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.050000_TAU30.000000_NDUPL5.000000_RNGSEED1.txt')[::10,:])

rs = dstack(rlist)


#fgr, sps = subplots(1, 2)

fgr = figure()
# 8.5cm = 3.3 inches for single column. 6.9 inches for two-column
fgr.set_size_inches(3.3, 3.3)
fgr.set_facecolor('white')


for numgraph in range(4):
    
    
    #ax = sps[numgraph/2,numgraph%2]
    #ax = sps[numgraph]
    ax = fgr.add_axes([(.1 + (numgraph%2) * .45), .1 + floor(numgraph/2) * .45, .40, .40])
    
    r = rs[:,:,numgraph]
    
    if numgraph == 2:
        ax.set_title('Before training:')
    if numgraph == 3:
        ax.set_title('After training:')
    if numgraph == 0:
        ax.set_ylabel('Stims. A & A')
    if numgraph == 2:
        ax.set_ylabel('Stims. A & B')

    ax.plot(r[:, 2:7])
    ax.plot(r[:,0],  'k', linewidth=2)
    ax.set_ylim([-1,1])
    ax.set_xlim([0,100])

    myxticks = [0, 20, 40, 60, 80, 100]
    ax.set_yticks([-1, 0, 1])
    ax.set_xticks(myxticks)            


    if numgraph % 2 ==0:
        ax.set_yticklabels(['-1', '0', '1'])
    else:
        ax.set_yticklabels([])            

    if floor(numgraph / 2) ==0:
        ax.set_xlabel('Time (ms)', size=10)
        ax.set_xticklabels([str(zz*10) for zz in myxticks])    
    else:
        ax.set_xticklabels([])            

savefig('figure_traj.png', bbox_inches='tight')
