import glob
import numpy as np
import matplotlib.pyplot as plt


def crittime(a) :   # When does that array reach the criterion (95% < 1.0 over 100 successive trials) ?
    z = a < 1.0
    span = 100
    ret = np.cumsum(z, dtype=float)
    ret[:-span] = ret[span:] - ret[:-span]
    ct = np.argmax(ret > 95)        # Returns the 1st position where ret > 95, or 0 if none...
    if ct == 0:
        ct = a.size                 # If criterion never reached, return length of array
    return ct


font = {#'family' : 'normal',
##                'weight' : 'bold',
                        'size'   : 10}
plt.rc('font', **font)

plt.ion()
np.set_printoptions(precision=3, suppress=True)





#dirz = ['trial-ETA-.01', 'trial-ETA-.01-MAXDW-2e-4']
#dirz = glob.glob('trial-*')
#dirz = ['trial-ETA-.01-ALPHAMODUL-30.0-MAXDW-2e-4-ALPHABIAS-.5']
dirz = ['.']
dirz.sort()
NBPLOTS = len(dirz)
print NBPLOTS, " graphs"
if NBPLOTS == 0:
    raise ValueException("No graph to print!") 
SS = np.ceil(np.sqrt(NBPLOTS))

#plt.figure(1,  figsize=(4, 3), dpi=100, facecolor='w', edgecolor='k')
plt.figure( figsize=(4, 3), dpi=100, facecolor='w', edgecolor='k')



nplot = 1
perfs = []
dirs = []
colorz=['b', 'b', 'b', 'r', 'g']
crittimes = []
for (num, droot) in enumerate(dirz):
    t = []
    filez = glob.glob(droot+'/errs*.txt')
    crittimes_thisdir = []
    for v in range(20):
        try:
            z = np.loadtxt(filez[v])
        except IOError:
            print "error loading "+filez[v]
            continue
        #z=z[:18000]
        #z=z[:800, :]
        if len(z) >= 20000:
            z=z[:20000]
            t.append(z)
            crittimes_thisdir.append(crittime(z))
        else:
            print filez[v]+" is too small! - size: "+str(z.size)
        #t.append(z)
    t = np.vstack(t)
    crittimes.append(crittimes_thisdir)
    tmean = np.mean(t, axis=0)
    tstd = np.std(t, axis=0)
    tmedian = np.median(t, axis=0)
    tq25 = np.percentile(t, 25, axis=0)
    tq75 = np.percentile(t, 75, axis=0)
    
    ax = plt.subplot(SS, SS, nplot)
    #ax.set_title(num)
    #plt.fill_between(range(len(tmean)), tq25, tq75, linewidth=0.0, alpha=0.3)
    plt.fill_between(range(len(tmean)), tq25, tq75, facecolor='gray', edgecolor='gray', label='IQR')
    plt.plot(tmedian, 'black')
    plt.axis([0, tmean.size, 0, 2.0])

    p1 = int(tmean.size / 3)
    p2 = 2*int(tmean.size / 3)
    p3 = -1

    print num, p1, ':', tmean[p1], p2, ':', tmean[p2], p3, ':', tmean[p3], droot, 'q25/med/q75: ', np.percentile(crittimes_thisdir, [25,   50, 75])
    perfs.append([tmean[p1], tmean[p2], tmean[p3]])
    dirs.append(droot)
    plt.show()
    plt.xlabel('Trial #') 
    plt.ylabel('Error')

    nplot += 1

print "Data read."

acrittimes = np.array(crittimes)
#mediancrittimes_o = np.median(crittimes, axis=1)
q25crittimes, q75crittimes, mediancrittimes = np.percentile(acrittimes, [25,  75, 50], axis=1)
perfs = np.array(perfs)
p = perfs[:,1]
ord = np.argsort(p)


plt.show()
plt.draw()

plt.savefig('figure_errs_selectiveint.png', bbox_inches='tight')
