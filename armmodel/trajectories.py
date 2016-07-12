from pylab import *
import glob
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import sys

ion()

for trialtype in range(2):

    if trialtype == 0:
        fn0 = glob.glob('rs_long_type0_*.txt');
        fn1 = glob.glob('rs_long_type1_*.txt');
    else:
        fn0 = glob.glob('rs_long_RANDW_type0_*.txt');
        fn1 = glob.glob('rs_long_RANDW_type1_*.txt');

# We store all the response arrays (NbTimeSteps x NbNeur) into data0l and data1l, which are lists (one element per trial). 
# Also we only use the non-input, non-output neurons.
    data0l = [loadtxt(ff)[:, 16:200] for ff in fn0]
    data1l = [loadtxt(ff)[:, 16:200]  for ff in fn1]

    #data0l = [loadtxt(ff)[:, :16] for ff in fn0]
    #data1l = [loadtxt(ff)[:, :16]  for ff in fn1]

# We also compute the mean across all trials for each condition !
    data0m = np.mean(dstack(data0l), 2);
    data1m = np.mean(dstack(data1l), 2);

# We find the axes of maximal variance in neural space, with one data point per timestep and trial, across all trials of all conditions !
    print "Data loaded, doing the pca..."
    pca = PCA()
    pca.fit(vstack((vstack(data0l), vstack(data1l))))

#sys.exit()

# We project all trials into the PCA space, obtaining a list of trajectories (NbTimeSteps x NbDim for each element in the list)
    traj0l = [pca.transform(mydata) for mydata in data0l]
    traj1l = [pca.transform(mydata) for mydata in data1l]

    ff = figure()
    ax = ff.add_subplot(111, projection='3d')
#ax.plot(rr0t[:,0], rr0t[:, 1], rr0t[:,2])

    for nn in range(5):
        tt = traj0l[nn]
        plot(tt[:,0], tt[:, 1], tt[:,2])
        tt = traj1l[nn]
        plot(tt[:,0], tt[:, 1], tt[:,2])

#plot(rr0t[:,0], rr0t[:, 1])
#plot(rr1t[:,0], rr1t[:, 1])
# To plot other projections, we need to pad one of the dimensions with some constant values (here, zeros):
#plot(rr0t[:,0], np.zeros((700,1)), rr0t[:, 2]) 
