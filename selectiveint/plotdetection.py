# While running the learning code, use this to find out how each run is going.
# fn = glob.glob('errs*.txt')
# [sum(loadtxt(ss)[-5000:]>1.0) for ss in fn]

from scipy.optimize import curve_fit


def sigmo(x, k, a, b, c):
    #y = -1.0 + 2.0 / (1.0 + exp(-k*x))
    #y = a + b / (1.0 + exp(-k*x))
    y = a + b / (1.0 + c * exp(-k*x))
    return y


from pylab import *
import glob

# Type 0 : modality 1 positive bias, modality 2 random, look at modality 1
# Type 1 : modality 1 negative bias, modality 2 random, look at modality 1
# Type 2 : modality 1 random, modality 2 positive bias, look at modality 2
# Type 3 : modality 1 random, modality 2 negative bias, look at modality 2 

# So Type 1 should be just the reverse of type 0, and type 2 should be reverse of type 3.

#Interactive mode 
ion()

fgr = figure()
fgr.set_size_inches(5, 5)
fgr.set_facecolor('white')

if  1:
    vals=[]
    #allfnames = glob.glob('rs_*type*.txt') 
    allfnames = glob.glob('lastr_*type*.txt') 
    for ff in allfnames:
        print ff
        z = loadtxt(ff);
        #vals.append(z[-1,0])
        vals.append(z)

    print "Files read!"
#patts = glob.glob('errs_*ETA3.0*TAU30*RNGSEED8*') 
#patts = glob.glob('errs_G1.500000_MAXDW0.000050_ETA1.500000_ALPHAMODUL4.000000_PROBAMODUL0.003000_SQUARING1_ALPHATRACE0.600000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.000000_TAU30.000000_RNGSEED8*') 
#patts = glob.glob('errs_G1.500000_MAXDW0.000050_ETA*_ALPHAMODUL4.000000_PROBAMODUL0.003000_SQUARING1_ALPHATRACE0.600000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.000000_TAU30.000000_RNGSEED8*') 
#patts = glob.glob('errs_*PROBAMODUL0.03*RNGSEED8*') 
#patts = glob.glob('../squaringworks/errs_G1.500000_MAXDW0.000050_ETA0.001000_ALPHAMODUL0.300000_PROBAMODUL0.100000_SQUARING1_SUBW0_ALPHATRACE0.600000_METHOD-DELTATOTALEXC_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.250000_RNGSEED8*.txt') 


    #allbiases = [(int(100.0*double(strn[23:29])))/100.0 for strn in allfnames]
    #trialtypes = [int(ss[15]) for ss in allfnames]
    #allbiases = [(int(100.0*double(strn[21:27])))/100.0 for strn in allfnames]
    allbias1s = [float(ss.split('_')[3]) for ss in allfnames]
    allbias2s = [float(ss.split('_')[5]) for ss in allfnames]
    trialtypes = [int(ss[10]) for ss in allfnames]
    trialnums = [int(ss.split('_')[-1][:-4]) for ss in allfnames]

    mat1 = vstack((trialtypes, allbias1s, allbias2s, trialnums, vals)).T

    mms1=[]
    mms2=[]
    for nn in unique(trialtypes):
        mm=[]
        for bb in unique(allbias1s):
            mm.append( vstack( mat1[(mat1[:,0] == nn) & (mat1[:,1] == bb), 4]) )
        mms1.append(hstack(mm))
        mm=[]
        for bb in unique(allbias2s):
            mm.append( vstack( mat1[(mat1[:,0] == nn) & (mat1[:,2] == bb), 4]) )
        mms2.append(hstack(mm))

# Now mmsX is a list of matrices (one per trial type, i.e. per attended modality), such that each matrix contains the array of results across all possible values of the bias on the Xth modality (X=0 or 1) (one bias value per column) for this trialtype.

xdata = arange(len(unique(allbias2s))) - floor(len(unique(allbias2s))/2.0) 
xdatanorm = .05 * xdata    # These are the actual x values - cannot use this as xdata because boxplot doesn't seem to like non-integer positions

for nplot in range(4):

    subplot(2, 2, nplot+1)
    if nplot == 0:
        title('Attend Modality 1')
        ylabel('Mean Response')
    if nplot == 1:
        title('Attend Modality 2')
        ylabel('Mean Response')

    if nplot < 2:
        mms = mms1
        xlabel('Bias Modality 1', verticalalignment='top', labelpad=0)
    if nplot >= 2:
        mms = mms2
        xlabel('Bias Modality 2', verticalalignment='top', labelpad=0)
    #title(str(nplot))
    ydata = np.mean(mms[nplot % 2], 0)
    popt, pcov = curve_fit(sigmo, xdata, ydata, maxfev=10000)
    print popt
    print ydata
    plot(xdata, ydata, '*', markersize=4)
    boxplot(mms[nplot % 2], positions=xdata)
    plot(xdata, sigmo(xdata, popt[0], popt[1], popt[2], popt[3]))
    yticks([-1, 1])
    xticks(xdata[::2], [str(nn) for nn in xdatanorm[::2]])

savefig('figure_detection.png', bbox_inches='tight', dpi=300)

#fnames=[]
#biases=[]
#for numtype in range(4): 
#    fnames.append([ss  for ss,nn in zip(allfnames, trialtypes) if nn==numtype])
#    biases.append([ss  for ss,nn in zip(allbiases, trialtypes) if nn==numtype])


