from pylab import *
import glob
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import sys

ion()

fn0 = glob.glob('pos_type0_*.txt');
fn1 = glob.glob('pos_type1_*.txt');
fn0r = glob.glob('pos_RANDW_type0_*.txt');
fn1r = glob.glob('pos_RANDW_type1_*.txt');

data0l = [loadtxt(ff) for ff in fn0]
data1l = [loadtxt(ff)  for ff in fn1]
data0lr = [loadtxt(ff) for ff in fn0r]
data1lr = [loadtxt(ff)  for ff in fn1r]



ff = figure()
ax = ff.add_subplot(111, projection='3d')
#ax.plot(rr0t[:,0], rr0t[:, 1], rr0t[:,2])

for nn in range(15):
    tt = data0l[nn].T
    plot(tt[:,0], tt[:, 1], tt[:,2], color='red')
    #plot(tt[:,0], tt[:, 1],  color='red')
    tt = data1l[nn].T
    plot(tt[:,0], tt[:, 1], tt[:,2], color='blue')
    #plot(tt[:,0], tt[:, 1],  color='blue')
    tt = data0lr[nn].T
    plot(tt[:,0], tt[:, 1], tt[:,2], color='red', linestyle='dashed')
    #plot(tt[:,0], tt[:, 1],  color='red', linestyle='dashed')
    tt = data1lr[nn].T
    plot(tt[:,0], tt[:, 1], tt[:,2], color='blue', linestyle='dashed')
    #plot(tt[:,0], tt[:, 1],  color='blue', linestyle='dashed')

xlabel('X')
ylabel('Y')
ax.set_zlabel('Z')

#plot(rr0t[:,0], rr0t[:, 1])
#plot(rr1t[:,0], rr1t[:, 1])
# To plot other projections, we need to pad one of the dimensions with some constant values (here, zeros):
#plot(rr0t[:,0], np.zeros((700,1)), rr0t[:, 2]) 
