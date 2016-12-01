
from pylab import *
import glob

ion()


# After running this program, run this :
# [patts[vv] for vv in argsort(mediancrittimes)]
# Current best:
# errs_G1.500000_MAXDW0.000100_ETA0.100000_ALPHAMODUL8.000000_PROBAMODUL0.010000_SQUARING1_MODULTYPE-DECOUPLED_ALPHATRACE0.750000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.050000_TAU30.000000_INPUTMULT3.000000_RNGSEED8.txt


# This is the number of trials in each 'group' of parameter values. The maximum error in each group should be reported. For the same-or-diff / XOR problem, this number is 4 (one for each parameter value combination).
# If set to 1, there is no grouping - just the plain raw error per trial
NBTRIALSPERGROUP = 1

patts = glob.glob('errs_*RNGSEED8*') 
#patts = glob.glob('errs_*ETA3.0*TAU30*RNGSEED8*') 
#patts = glob.glob('errs_G1.500000_MAXDW0.000050_ETA1.500000_ALPHAMODUL4.000000_PROBAMODUL0.003000_SQUARING1_ALPHATRACE0.600000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.000000_TAU30.000000_RNGSEED8*') 
#patts = glob.glob('errs_G1.500000_MAXDW0.000050_ETA*_ALPHAMODUL4.000000_PROBAMODUL0.003000_SQUARING1_ALPHATRACE0.600000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.000000_TAU30.000000_RNGSEED8*') 
#patts = glob.glob('errs_*PROBAMODUL0.03*RNGSEED8*') 
#patts = glob.glob('../squaringworks/errs_G1.500000_MAXDW0.000050_ETA0.001000_ALPHAMODUL0.300000_PROBAMODUL0.100000_SQUARING1_SUBW0_ALPHATRACE0.600000_METHOD-DELTATOTALEXC_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.250000_RNGSEED8*.txt') 

selectstrs=[]
for patt in patts:
    selectstrs.append(patt[:-7] + '*')

#selectstrs = ['errs_G1.500000_MAXDW0.000100_ETA0.500000_ALPHAMODUL4.000000_PROBAMODUL0.010000_SQUARING1_MODULTYPE-UNIFORM_ALPHATRACE0.500000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.000000_TAU30.000000_RNGSE*',
# 'errs_G1.500000_MAXDW0.000100_ETA0.500000_ALPHAMODUL4.000000_PROBAMODUL0.010000_SQUARING1_MODULTYPE-UNIFORM_ALPHATRACE0.750000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.000000_TAU30.000000_RNGSE*',
# 'errs_G1.500000_MAXDW0.000100_ETA0.250000_ALPHAMODUL4.000000_PROBAMODUL0.010000_SQUARING1_MODULTYPE-UNIFORM_ALPHATRACE0.750000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.000000_TAU30.000000_RNGSE*']

print len(selectstrs), ' Graphs.'

# Best so far:
# 'errs_G1.500000_MAXDW0.000100_ETA0.500000_ALPHAMODUL8.000000_PROBAMODUL0.001000_SQUARING1_MODULTYPE-UNIFORM_ALPHATRACE0.500000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.000000_TAU30.000000_RNGSE*'
# Goes to lower asympt, but has more fails (3 io 1):
#'errs_G1.500000_MAXDW0.000100_ETA0.500000_ALPHAMODUL8.000000_PROBAMODUL0.003000_SQUARING1_MODULTYPE-UNIFORM_ALPHATRACE0.330000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.000000_TAU30.000000_RNGSE*'
# Less good:
#'errs_G1.500000_MAXDW0.000050_ETA1.500000_ALPHAMODUL4.000000_PROBAMODUL0.003000_SQUARING1_MODULTYPE-UNIFORM_ALPHATRACE0.330000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.000000_TAU30.000000_RNGSE*'
# 1435.5 2226.5
#'errs_G1.500000_MAXDW0.000050_ETA0.250000_ALPHAMODUL8.000000_PROBAMODUL0.003000_SQUARING1_MODULTYPE-UNIFORM_ALPHATRACE0.750000_METHOD-DELTAX_ALPHABIAS0.000000_PROBAHEBB1.000000_ATRACEEXC0.000000_TAU30.000000_RNGSE*'
# 1339.0 2588.5


figure()
clf()

LENGTH = 10000 #20000 #198000
CRITERIONPERIOD = 100
CRITERIONVALUE = 95
crittimesallplots = []

nplot = 0
for selectstr in selectstrs:
    crittimes=[]
    nplot += 1
    print nplot
    if 1:
        fnames = glob.glob(selectstr)
        errslist = []
        for nm in fnames:
            print "Loading file "+nm
            eee = loadtxt(nm)[0:LENGTH]
            if eee.size < LENGTH:
                print "Error: file "+nm+" has only "+str(eee.size)+" data points!"
                continue
            #OK, now grouping by groups of 4 (or whatever NBTRIALSPERGROUP is) and taking the max within each group
            eee = reshape(eee, (LENGTH/NBTRIALSPERGROUP, NBTRIALSPERGROUP))  # NOT 8 anymore !
            eee = amax(eee, axis = 1)
            errslist.append(eee)
            
            # When is criterion reached?
            # Note that we do this after the grouping, so we need to multiply by NBTRIALSPERGROUP.
            correctz = uint8(eee<1.0)
            cumul = np.zeros(correctz.shape)
            for nn in range(CRITERIONPERIOD):
                cumul = cumul + np.roll(correctz, nn)
            cumul[0:CRITERIONPERIOD].fill(0)
            crittime = NBTRIALSPERGROUP * np.argmax(cumul > CRITERIONVALUE)
            if crittime == 0:
                crittime = LENGTH
            crittimes.append(crittime)


        allerrs = vstack(errslist).transpose() ;

    crittimes = np.hstack(crittimes)
    crittimesallplots.append(crittimes)

    q25,q75 = percentile(allerrs, [25, 75], axis=1)
    mm = median(allerrs, axis=1)


    #figure(5, (3.2, 3))
    subplot(ceil(len(selectstrs) / 4.0 ), 4, nplot)

    rcParams.update({'font.size': 10})

    qtl = fill_between((arange(len(q75))* NBTRIALSPERGROUP), q25, q75, facecolor='gray', edgecolor='gray', label='IQR')
    mml = plot(arange(len(q75))*NBTRIALSPERGROUP, mm, 'black', linewidth=1.0, label='Median error')
    
    critml = axvline(median(crittimes))
    critq25l = axvline(percentile(crittimes, 25), linestyle=':')
    critq75l = axvline(percentile(crittimes, 75), linestyle=':')

# No legend for fill_between, at least not without some work
#mylegend = legend(handles=[ mml, qtl], prop={'size':9})
#for legobj in mylegend.legendHandles:
#    legobj.set_linewidth(2.0)

    axis([0,LENGTH,0,2.0])
    xticks(range(0,LENGTH+1,10000))
    yticks([0, 1.0, 2.0])
    xlabel('Trial #')
    ylabel('Error')
    title(selectstr[15:].replace("_SQUARING1_SUBW0_ALPHATRACE0.600000_METHOD-DELTATOTALEXC_ALPHABIAS0.000000_PROBAHEBB1.000000","").replace("_ALPHAMODUL0.000000_PROBAMODUL0.000000_SUBW0", "").replace("_METHOD-DXTRIAL","").replace("0000_","_").replace("_RNGSEED","").replace("errs_G1.500000_","").replace("000_",""), fontsize=7)

crittimesallplots = np.vstack(crittimesallplots)
mediancrittimes = np.median(crittimesallplots, axis=1)
q75crittimes = np.percentile(crittimesallplots, 75, axis=1)
print "Median times to criterion, across all runs, for each param. combination: ", mediancrittimes 
print "75th percentile times to criterion: ", q75crittimes 
orderq75 = argsort(q75crittimes)
ordermedian = argsort(mediancrittimes)
#selectstrssorted = [selectstrs[nn] for nn in orderq75[:20]]
selectstrssorted = [selectstrs[nn] for nn in ordermedian]

show() # why do I have to put this ?... 
#savefig('figure_errs.png', bbox_inches='tight')
