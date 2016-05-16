# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:32:26 2015

@author: miconi
"""
from pylab import *
from matplotlib import patches
from matplotlib.patches import Rectangle
rcParams.update({'font.size': 8})

fnames =['./accur_stim1.txt', './accur_stim2.txt', './accur_xor.txt']
titlez=['Decoding 1st stimulus', 'Decoding 2nd stimulus', 'Decoding response']
nn=0

ion()

figure()
clf()

fgr, sps = subplots(1,len(fnames))
# 8.5cm = 3.3 inches for single column. 6.9 inches for two-column
fgr.set_size_inches(6.9, 5)
fgr.set_facecolor('white')

#for fn in (fnames[0],):
for nn in range(len(fnames)):
    ee = loadtxt(fnames[nn])
    ax=sps[nn]
    
    myimg = ax.imshow(ee.T, cmap='hot', origin='lower', interpolation='nearest')
    myimg.set_clim([0, 1])
    
    ax.set_xlabel('Decoding time (ms)', verticalalignment='top')

    # Doesn't work! :
    #ax.set_xticks([0, 20, 40, 60, 80, 110], ['0', '200', '400', '600', '800',  '1100'])
    #ax.set_xticks([0, 20, 40, 60, 80, 110])
    ax.set_xticks([0, 20, 40, 60, 70, 100])
    ax.set_xticklabels( ['0', '200', '400', '600', '700',  '1000'], size=6)

    if nn==0:
        #ax.yaxis.set_ticks([0, 20, 40, 60, 80, 110])
        ax.yaxis.set_ticks([0, 20, 40, 60, 70, 100])
        ax.yaxis.set_ticklabels(['0', '200', '400', '600', '700',  '1000'])
        ax.set_ylabel('Training time (ms)', verticalalignment='bottom')
    else:
        ax.yaxis.set_ticks([0, 20, 40, 60, 70, 100])
        ax.yaxis.set_ticklabels([])
    
    

    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x',direction='out', pad=10)
    #ax.xaxis.set_label_position('top')

    #myrect1 = patches.Rectangle((-.5, 110), 20, 5, facecolor='gray', edgecolor='gray')
    myrect1 = patches.Rectangle((-.5, -9), 20.5, 8.5, facecolor='#AAAAAA', edgecolor='#AAAAAA')
    ax.add_patch(myrect1)
    myrect1.set_clip_on(False)
    #myrect2 = patches.Rectangle((40, 110), 20, 5, facecolor='gray', edgecolor='gray')
    myrect2 = patches.Rectangle((40, -9), 20.5, 8.5, facecolor='#AAAAAA', edgecolor='#AAAAAA')
    ax.add_patch(myrect2)
    myrect2.set_clip_on(False)
    #myrect3 = patches.Rectangle((80, 110), 30, 5, facecolor='gray', edgecolor='gray')
    myrect3 = patches.Rectangle((70, -9), 30, 8.5, facecolor='#AAAAAA', edgecolor='#AAAAAA')
    ax.add_patch(myrect3)
    myrect3.set_clip_on(False)

   
   
    ax.text(0, -8.5, 'Stim. 1', size=6) 
    ax.text(40, -8.5, 'Stim. 2', size=6) 
    ax.text(75, -8.5, 'Response', size=6) 
    ax.set_title(titlez[nn], size=10)
    
fgr.subplots_adjust(right=0.9)
cbar_ax = fgr.add_axes([0.92, 0.33, 0.02, 0.33])
fgr.colorbar(myimg, cax=cbar_ax,  ticks=[0, .5, 1.0])

    
#show()

savefig('figure_decode.png', bbox_inches='tight', dpi=300)
#colorbar(myimg, ax, ticks=[0, .5, 1.0]) #, fraction=0.046, pad=0.04)

