
from pylab import *
import numpy as np
import scipy as sp
import glob


from sklearn import manifold
#from sklearn.metrics import euclidean_distances

ion()

figure()


for RNGSEED in range(1, 21):

    print "Making graph for RNGSEED ", RNGSEED

    if 1:
        datalist=[] 
        labellist=[]
        for trialtype in range(4):
            print trialtype
            fnames = glob.glob('rs_long_type'+str(trialtype)+'*SEED'+str(RNGSEED)+'.txt')
            for nm in fnames:
                r = loadtxt(nm)
                #z = r.reshape((110,10,200))
                #z = sum(z,axis=1)
                z = r #r[0::10,:]
                datalist.append(z)
                #labellist.append([trialtype]*r.shape[0])
                labellist.append(trialtype)

#single:    
# not 3, not 4, not 5, maybe 6? not 7, not 8, not 9, not 10, not 11, not 12, not 13 (hmm.), not 14, not 15, 16?, not 17, not 18 (though close), not 19,  not 20
# Least bad is 16    

#singlealt:
# Not 1, not 2, not 3(only 2 fuse), not 4, not 5, not 6 (only 2 fuse), not7 (all fuse!). not 8(presumably all fuse-fail), not 9, maybe 10?, not 11, not 12
# not 13 (only 2 fuse), 14 ! , not 15 (fail), not 16,not 17,  18! 19! not 20.


    subplot(5, 4, RNGSEED)

    matdata = dstack(datalist) ; #+ .5  * standard_normal(matdata.shape) 

    matdata = matdata[:,:,::2]
    NBPTS = matdata.shape[2]

    matdata += .0 * standard_normal(shape(matdata))



    #fgr, sps = subplots(3, 2)

# 8.5cm = 3.3 inches for single column. 6.9 inches for two-column
#fgr.set_size_inches(3.3, 6)
#fgr.set_facecolor('white')
#slicetimes= [850, 900, 990, 999] #[200, 600, 900 , 850, 990, 999]
    slicetimes= [199, 599, 799, 999] #[200, 600, 900 , 850, 990, 999]
    slicetimes= [199, 599, 799, 999] 


#    for numgraph in range(4):
    
    numgraph = 3

    tslc = matdata[slicetimes[numgraph],:,:].T
    
    mds = manifold.MDS(n_components=2,  max_iter=10000, dissimilarity="euclidean")
    pos = mds.fit(tslc).embedding_
    
    #ax = sps[numgraph/2,numgraph%2]
    
    title(str(1+slicetimes[numgraph])+'ms', size=10)

    plot(pos[0:NBPTS/4-1, 0], pos[0:NBPTS/4-1, 1], 'oc', markersize=8)
    plot(pos[NBPTS/4:2*NBPTS/4-1, 0], pos[NBPTS/4:2*NBPTS/4-1, 1], 'or', markersize=8)
    plot(pos[2*NBPTS/4:3*NBPTS/4-1, 0], pos[2*NBPTS/4:3*NBPTS/4-1, 1], 'og', markersize=8)
    plot(pos[3*NBPTS/4:NBPTS-1, 0], pos[3*NBPTS/4:NBPTS-1, 1], 'oy', markersize=8)
    if numgraph==0:
        xlabel('Dimension 1', size=10)
        ylabel('Dimension 2', size=10)
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    #xlim(-10,10)
    #ylim(-10,10)
    #axes().set_aspect('equal', 'box')

#    sps[2,0].axis('off')
#    sps[2,1].axis('off')
#    sps[1,0].legend(['AA','AB', 'BA', 'BB'],  numpoints = 1, ncol= 2, loc=3, prop={'size':10}, bbox_to_anchor=(.3,-.7))
#savefig('figure_mds.png', bbox_inches='tight', dpi=300)


