
from pylab import *
import glob

# This is the number of trials in each 'group' of parameter values. The maximum error in each group should be reported. For the same-or-diff / XOR problem, this number is 4 (one for each parameter value combination).
# If set to 1, there is no grouping - just the plain raw error per trial
NBTRIALSPERGROUP = 1

#Best, but slowish fall
#patts = glob.glob('errs*XDW0.001000_ETA0.100000_ALPHAMODUL16.000000_PROBAMODUL0.001000_SQUARING1_MODULTYPE-DECOUPLED_ALPHATRACE0.500000_METHOD-DELTAX_ALPHABIAS0.000000_P*SEED8.txt')

# Best
patts = glob.glob('errs*SEED8.txt')

# Good ish
#patts = glob.glob('errs*XDW0.000300_ETA0.100000_ALPHAMODUL16.000000_PROBAMODUL0.001000_SQUARING1_MODULTYPE-DECOUPLED_ALPHATRACE0.750000_METHOD-DELTAX_ALPHABIAS0.000000_P*SEED8.txt')

#Also slow fall / large residual
#patts = glob.glob('errs*XDW0.001000_ETA0.100000_ALPHAMODUL8.000000_PROBAMODUL0.010000_SQUARING1_MODULTYPE-DECOUPLED_ALPHATRACE0.500000_METHOD-DELTAX_ALPHABIAS0.000000_P*SEED8.txt')

# Somewhat irregular / large enveolpe fall, but lower residual
#patts = glob.glob('errs*XDW0.000300_ETA0.100000_ALPHAMODUL16.000000_PROBAMODUL0.003000_SQUARING1_MODULTYPE-DECOUPLED_ALPHATRACE0.750000_METHOD-DELTAX_ALPHABIAS0.000000_P*SEED8.txt')

#Meh, high is residual
#patts = glob.glob('errs*XDW0.000300_ETA0.030000_ALPHAMODUL16.000000_PROBAMODUL0.003000_SQUARING1_MODULTYPE-DECOUPLED_ALPHATRACE0.750000_METHOD-DELTAX_ALPHABIAS0.000000_P*SEED8.txt')

#Same
#patts = glob.glob('errs*XDW0.001000_ETA0.030000_ALPHAMODUL16.000000_PROBAMODUL0.003000_SQUARING1_MODULTYPE-DECOUPLED_ALPHATRACE0.500000_METHOD-DELTAX_ALPHABIAS0.000000_P*SEED8.txt')


print len(patts), ' Graphs.'
selectstrs=[]
for patt in patts:
    selectstrs.append(patt[:-7] + '*')

figure( figsize=(3.2, 3) )
clf()

LENGTH = 10000 #198000
CRITERIONPERIOD = 100
CRITERIONVALUE =95 
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
            
            # When is criterion reached (95% error < 1.0 in 100 consecutive trials?)
            # Note that we do this after the grouping, so we need to multiply by NBTRIALSPERGROUP.
            # (This code is very ugly)
            # NOTE: We are actually returning the point *after which* the previous 100 trials met criterion, rather than the point *from which* the next 100 trials will reach criterion... Not sure if good.
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
    #subplot(ceil(len(selectstrs) / 4.0 ), 4, nplot)

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
    #title(selectstr[15:].replace("_SQUARING1_SUBW0_ALPHATRACE0.600000_METHOD-DELTATOTALEXC_ALPHABIAS0.000000_PROBAHEBB1.000000","").replace("_ALPHAMODUL0.000000_PROBAMODUL0.000000_SUBW0", "").replace("_METHOD-DXTRIAL","").replace("0000_","_").replace("_RNGSEED","").replace("errs_G1.500000_","").replace("000_",""), fontsize=7)

crittimesallplots = np.vstack(crittimesallplots)
mediancrittimes = np.median(crittimesallplots, axis=1)
q25crittimes = np.percentile(crittimesallplots, 25, axis=1)
q75crittimes = np.percentile(crittimesallplots, 75, axis=1)
print "Median times to criterion, across all runs, for each param. combination: ", mediancrittimes 
print "25th percentile times to criterion: ", q25crittimes 
print "75th percentile times to criterion: ", q75crittimes 
savefig('figure_errs.png', bbox_inches='tight', dpi=300)
show() # why do I have to put this ?... 
