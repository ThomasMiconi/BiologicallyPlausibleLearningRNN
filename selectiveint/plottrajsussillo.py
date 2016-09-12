# -*- coding: utf-8 -*-


from pylab import *
import numpy as nps
import scipy as sp
import glob



rlist = []
rlist.append(loadtxt('rs_long_type0_bias1_0.500000_bias2_-0.500000_2.txt')[::10,:])
rlist.append(loadtxt('rs_long_type1_bias1_0.500000_bias2_-0.500000_2.txt')[::10,:])
rlist.append(loadtxt('rs_long_type0_bias1_-0.500000_bias2_0.500000_2.txt')[::10,:])
rlist.append(loadtxt('rs_long_type1_bias1_-0.500000_bias2_0.500000_2.txt')[::10,:])

rs = dstack(rlist)


#fgr, sps = subplots(1, 2)

ion()

fgr = figure()
# 8.5cm = 3.3 inches for single column. 6.9 inches for two-column
#fgr.set_size_inches(3.3, 3.3)
fgr.set_facecolor('white')


for numgraph in range(4):
    
    
    #ax = sps[numgraph/2,numgraph%2]
    #ax = sps[numgraph]
    #ax = fgr.add_axes([(.1 + (numgraph%2) * .45), .1 + floor(numgraph/2) * .45, .40, .40])

    subplot(2, 2, numgraph+1)    

    r = rs[:,:,numgraph]
    
    if numgraph == 0:
        title('Attend Mod. 1:')
    if numgraph == 1:
        title('Attend Mod. 2:')
    if numgraph == 0:
        ylabel('Mod. 1 + / Mod. 2 -')
    if numgraph == 2:
        ylabel('Mod. 1 - / Mod. 2 +')

    plot(r[:, 2:10])
    plot(r[:,0],  'k', linewidth=2)
    ylim([-1,1])
    xlim([0,70])

    myxticks = [0, 20, 40, 60, 70]
    if numgraph % 2 == 0:
        yticks([-1, 0, 1], ['-1', '0', '1'])
    else:
        yticks([-1, 0, 1], [])




    if floor(numgraph / 2) ==1:
        xlabel('Time (ms)', size=10)
        xticks(myxticks, [str(zz*10) for zz in myxticks])    
    else:
        xticks(myxticks, [])            

savefig('figure_traj_sussillo.png', bbox_inches='tight')
