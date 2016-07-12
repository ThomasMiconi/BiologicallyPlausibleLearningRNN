from pylab import *
import glob
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import sys

ion()

for trialtype in range(2):

    if trialtype == 0:
        fn0 = glob.glob('pos_type0_*.txt');
        fn1 = glob.glob('pos_type1_*.txt');
    else:
        fn0 = glob.glob('pos_RANDW_type0_*.txt');
        fn1 = glob.glob('pos_RANDW_type1_*.txt');

    data0l = [loadtxt(ff) for ff in fn0]
    data1l = [loadtxt(ff)  for ff in fn1]



    ff = figure()
    ax = ff.add_subplot(111, projection='3d')
#ax.plot(rr0t[:,0], rr0t[:, 1], rr0t[:,2])

    for nn in range(11):
        tt = data0l[nn].T
        plot(tt[:,0], tt[:, 2], tt[:,1])
        tt = data1l[nn].T
        plot(tt[:,0], tt[:, 2], tt[:,1])

#plot(rr0t[:,0], rr0t[:, 1])
#plot(rr1t[:,0], rr1t[:, 1])
# To plot other projections, we need to pad one of the dimensions with some constant values (here, zeros):
#plot(rr0t[:,0], np.zeros((700,1)), rr0t[:, 2]) 
