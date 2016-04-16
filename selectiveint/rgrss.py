import cPickle as pickle
from pylab import *
import glob
from sklearn.decomposition import PCA
from sklearn import linear_model


# Type 0 : modality 1 positive bias, modality 2 random, look at modality 1
# Type 1 : modality 1 negative bias, modality 2 random, look at modality 1
# Type 2 : modality 1 random, modality 2 positive bias, look at modality 2
# Type 3 : modality 1 random, modality 2 negative bias, look at modality 2 

# So Type 1 should be just the reverse of type 0, and type 2 should be reverse of type 3.

#Interactive mode 
ion()

#fgr = figure()
#fgr.set_size_inches(6, 6)
#fgr.set_facecolor('white')



if  1:  # After you run this code the 1st time, you can replace this with 'if 0:' to load pre-digested data from the pkl file, which saves some time
    print "Loading files..."
    resps = []
    allfnames = glob.glob('rs_*type*.txt') 
    #allfnames = glob.glob('lastr_*type*.txt') 
    cpt = 0
    for ff in allfnames:
        cpt = cpt+1
        #if cpt == 201:
        #    break
        print ff, cpt, "/", len(allfnames)
        z = loadtxt(ff)[0::10, :];  # We only sample every 10ms. Response matrices have dimensions Time x Neuron
        #vals.append(z[-1,0])
        resps.append(z)
    pickle.dump( resps, open( "resps.pkl", "wb" ) )
else:
    print "Opening file..."
    resps = pickle.load(open("resps.pkl", "rb"))
    print "Files read!"


NBTIMESLICES = resps[0].shape[0]
NBTRIALS = resps[0].shape[1]


# The vector of final network responses (i.e. last output value of neuron 0), for all trials
outz = np.array( [resps[x][-1, 0] for x in range(len(resps))] )


resps = vstack(resps)
# Now resps has dimensions (NbTriels * NbTimeslices) X NbNeur

pca = PCA()
#rv = vstack(resps)
# We can't just use zscore: we need to actually compute the mean and standard
# deviation that we will subtract and divide by, because we need them again
# during the PCA transformation.
pca.fit(resps)
PCs = pca.transform(resps)
#PCs[:, 25:] = 0
#reconstructed_data = pca.inverse_transform(PCs)

# The transformed data is the first 25 PCs of the original data.
# It has dimensions (NbTrials * Time) X NbPCs
resps_PCs = PCs[:,:25]


allfnames = glob.glob('rs_*type*.txt') 
allbias1s = np.array([float(ss.split('_')[4]) for ss in allfnames])
allbias2s = np.array([float(ss.split('_')[6]) for ss in allfnames])
trialtypes = np.array([int(ss[12]) for ss in allfnames])
trialnums = np.array([int(ss.split('_')[-1][:-4]) for ss in allfnames])


# We want to rearrange  into Time different arrays (one per time slice), each
# of which has NbTrials x NbPCs dimension - so it can be used to regress, for
# each time slice, the various aspects of the trial (mod 1, mod 2, resp) over
# PC activity 
# We can also perform the PCs-to-task-features regression at the same time, for each time slice

datapertime = []
for timeslice in range(NBTIMESLICES):
    #datapertime.append(resps_PCs[timeslice::resps[0].shape[0]])
    datapertime.append(resps[timeslice::NBTIMESLICES, :])

dd = np.array(datapertime)

# Compute the means and std. dev. of responses for each cell, across all trials and time points
meanz = np.array( [np.mean(dd[:,:,x]) for x in range(dd.shape[2])] )  
stdz = np.array( [np.std(dd[:,:,x]) for x in range(dd.shape[2])] )  


# Note: because some neurons have constant output (bias neurons !), they will
# have std 0, so we should not include them.

goodneurz = stdz > 1e-10

dd = dd[:,:,goodneurz]
meanz = meanz[goodneurz]
stdz = stdz[goodneurz]

zdata = np.divide( (dd - meanz), stdz )  # Python broadcasting FTW! 


# We need to build a new vector that contains the expected ('target') response for each trial (-1 or +1)
# The target response is the sign of the bias in the relavant modality (+1 or -1)
# If trialtype = 0 relevant modality is 1. If trialtype = 1 relevant modality is 2. So...
trialtgts = sign(allbias1s) * (1 - np.array(trialtypes)) + sign(allbias2s) *  np.array(trialtypes)
#Correlates with actual network response at ~ .90

# Important that trialtgts (the choice variable) goes first, because of the QR decomposition. See Mante Sussillo Supp Mat 6.7 
regressands = np.vstack((trialtgts, allbias1s, allbias2s, trialtypes)).T

# Here we simply regress the regressands over the whole population state, at each successive time slice, across all trials.
regmodpertime=[]
for timeslice in range(NBTIMESLICES):
    regmod = linear_model.LinearRegression()
    regmod.fit(zdata[timeslice], regressands)
    regmodpertime.append(regmod)

# Now regmodpertime is a list of regmods, i.e. regression models (one per time slice)
# capturing the regression of the z-scored population PCs over the regressands (task
# features, i.e. trial type/context, target choice, etc.)


# But that's not what Mante-Sussillo do: instead they regress the activity of each individual cell over the 'regressands' (task features).
# Then later they use the betas (for each task feature) as directions in space on which to project the population state.

regcellpertime=[]
coeffz = []
qpertime = []
interz = [] # Intercepts of the regressions
for timeslice in range(NBTIMESLICES):
    regcell = linear_model.LinearRegression()
    regcell.fit(regressands, zdata[timeslice])
    regcellpertime.append(regcell)
    coeffz.append(regcell.coef_)
    q, r = np.linalg.qr(regcell.coef_)
    qpertime.append(q)
    interz.append(regcell.intercept_)

# Make coeffz a NbCells X NbFeatures X NbTimeslices array
coeffz = dstack(coeffz)
# Make interz a NbTimeSlices X NbCells array
interz = vstack(interz)


# For every feature, we want the vector of Betas that has the largest norm across all time slices:
coeffzmax=[]
for numfeature in range(4):
    coeffzmax.append(coeffz[:, numfeature, argmax([norm(coeffz[:, numfeature , x]) for x in range(70)])])
coeffzmax = vstack(coeffzmax).T
qcoeffzmax, r = np.linalg.qr(coeffzmax)

# Showing how that the computed regression coefficients can be used for predicting cell activations from regressands / trial features
#cc = coeffz[:,:,-1]
#pv = np.dot(cc, regressands[13, :]) + interz[-1]
#print corrcoef(pv - zdata[-1, 13, :])

# Mante-Sussillo projection: predicting the regressands from the population responses (here, using only data from last timeslice)
# cc = coeffz[:,:,-1]
# pv = [np.dot(cc.T, ravel(zdata[-1, x, :]) + interz[-1,:] )[2] for x in range(100)]
# print corrcoef(pv, regressands[:100, 2])

# We find a mask for the "really-correct" trials:
correctz = ( (sign(outz) == trialtgts) & (abs(outz) > .5) )

# Let us average through a certain  sub-category of trials:
fgr = figure()
#fgr.set_size_inches(3, 3)
fgr.set_facecolor('white')
splts = []
linez = []
matplotlib.rcParams.update({'font.size': 10})
for numgraph in range(4):
    splts.append(subplot(2, 2, numgraph+1))
    title('Context: Attend Mod. ' + str(numgraph % 2 + 1) +'\n' + 'Grouping by Mod. ' + str(int(numgraph/2) +1) +' bias', fontsize=10)
    xlabel('Choice representation' )
    ylabel('Mod. ' + str(1 + int(numgraph/2)) + ' representation')
    for choice in ( -1, 1):
        for B1 in unique(allbias1s):
            if B1 == 0:
                continue
            print B1, " ", 
            #mask1 = correctz & (allbias1s == B1) & (trialtypes == 1)
            # We select the trials whose trajectories will be averaged for this particular curve
            if numgraph < 2:
                mask1 = correctz & (allbias1s == B1) & (trialtypes == numgraph % 2) & (trialtgts == choice)
            else:
                mask1 = correctz & (allbias2s == B1) & (trialtypes == numgraph % 2) & (trialtgts == choice)
            meantraj = np.mean(zdata[:, mask1, :] , axis = 1)
            predictedfeatures = []
            for timeslice in range(NBTIMESLICES):
                #predictedfeatures.append(np.dot(qpertime[-1].T, meantraj[timeslice, :] + interz[timeslice, :] ) ) # + interz[timeslice,:] ) )
                #predictedfeatures.append(np.dot(qpertime[timeslice].T, meantraj[timeslice, :] + interz[timeslice, :] ) ) # + interz[timeslice,:] ) ) # This is Bad!
                #predictedfeatures.append(np.dot(coeffz[:, :, timeslice].T, meantraj[timeslice, :] + interz[timeslice, :] ) ) # + interz[timeslice,:] ) )
                #predictedfeatures.append(np.dot(coeffz[:, :, -1].T, meantraj[timeslice, :] + interz[timeslice, :] ) ) # + interz[timeslice,:] ) )
                #predictedfeatures.append(np.dot(coeffzmax[:, :].T, meantraj[timeslice, :] + interz[timeslice, :] ) ) # + interz[timeslice,:] ) )
                #predictedfeatures.append(np.dot(coeffzmax[:, :].T, meantraj[timeslice, :] ) )
                #predictedfeatures.append(np.dot(qcoeffzmax[:, :].T, meantraj[timeslice, :] + interz[timeslice, :] ) ) # + interz[timeslice,:] ) )
                predictedfeatures.append(np.dot(qcoeffzmax[:, :].T, meantraj[timeslice, :] ) )
            z, = plot([predictedfeatures[x][0] for x in range(2, NBTIMESLICES)], [predictedfeatures[x][1 + int(numgraph/2)]  for x in range(2, NBTIMESLICES)],
                    color = [.5 + 2 * B1, .5 - 2* B1, .5 - 2*abs(B1)] )
            linez.append(z)


    print " " 
splts[0].legend([linez[0], linez[-1]], ['-0.25' , '0.25'], loc=3, fontsize=10)
tight_layout()

savefig('figure_rgrss.png', bbox_inches='tight', dpi=300)

# Prediction using the whole-population regression coefficients: super-strong signals for all features within the first 10 ms??
#predictedpertime = []
#for timeslice in range(NBTIMESLICES):
#    predictedpertime.append(regmodpertime[timeslice].predict(meantraj[timeslice,:]))
#
#pp = np.array(predictedpertime)[:, 0, :]



# Plot the predicted bias over the predicted choice, as a function of time...
#plot(predictedpertime[:, 1000, 3], predictedpertime[:, 1000, 0])
# Single trajectories are very weird.. Try the average, like mate & sussillo?




# We want to use the last-timeslice regmod to predict modality 1 bias, modality 2
# bias and target choice (+1 or -1) based on the PCs for each timeslice, then
# rearrange this to form trajectories for each trial over successive trials.

#avgb1s=[]
#avgb2s=[]
#avgtts=[]
#for b1 in unique(allbias1s):
#    for b2 in unique(allbias2s):
#        for tt in unique(trialtypes):
#            avgb1s.append(b1)
#            avgb2s.append(b2)
#            avgts.append(tt)
#            avgs.append(np.mean(dstack(






#sys.exit()

# We project all trials into the PCA space, obtaining a list of trajectories (NbTimeSteps x NbDim for each element in the list)
    #traj0l = [pca.transform(mydata) for mydata in data0l]
    #traj1l = [pca.transform(mydata) for mydata in data1l]



#    allbias1s = [float(ss.split('_')[4]) for ss in allfnames]
#    allbias2s = [float(ss.split('_')[6]) for ss in allfnames]
#    trialtypes = [int(ss[12]) for ss in allfnames]
#    trialnums = [int(ss.split('_')[-1][:-4]) for ss in allfnames]
#
#
## kkkk kkkk
#
#
#    #mat1 = vstack((trialtypes, allbias1s, allbias2s, trialnums, vals)).T
#
#    mms1=[]
#    mms2=[]
#    for nn in unique(trialtypes):
#        mm=[]
#        for bb in unique(allbias1s):
#            mm.append( vstack( mat1[(mat1[:,0] == nn) & (mat1[:,1] == bb), 4]) )
#        mms1.append(hstack(mm))
#        mm=[]
#        for bb in unique(allbias2s):
#            mm.append( vstack( mat1[(mat1[:,0] == nn) & (mat1[:,2] == bb), 4]) )
#        mms2.append(hstack(mm))
#
## Now mmsX is a list of matrices (one per trial type, i.e. per attended modality), such that each matrix contains the array of results across all possible values of the bias on the Xth modality (X=0 or 1) (one bias value per column) for this trialtype.
#
#xdata = arange(len(unique(allbias2s))) - floor(len(unique(allbias2s))/2.0) 
#xdatanorm = .1 * xdata    # Cannot use this as xdata because boxplot doesn't seem to like non-integer positions
#
#for nplot in range(4):
#
#    subplot(2, 2, nplot+1)
#    if nplot == 0:
#        title('Attend Mod. 1')
#    if nplot == 1:
#        title('Attend Mod. 2')
#
#    if nplot < 2:
#        mms = mms1
#        xlabel('Bias Mod. 1', verticalalignment='top', labelpad=0)
#    if nplot >= 2:
#        mms = mms2
#        xlabel('Bias Mod. 2', verticalalignment='top', labelpad=0)
#    #title(str(nplot))
#    ydata = np.mean(mms[nplot % 2], 0)
#    popt, pcov = curve_fit(sigmo, xdata, ydata, maxfev=10000)
#    print popt
#    print ydata
#    plot(xdata, ydata, '*', markersize=4)
#    boxplot(mms[nplot % 2], positions=xdata)
#    plot(xdata, sigmo(xdata, popt[0], popt[1], popt[2], popt[3]))
#    yticks([-1, 1])
#    xticks(xdata[::2], [str(nn) for nn in xdatanorm[::2]])
#
#
#
##fnames=[]
##biases=[]
##for numtype in range(4): 
##    fnames.append([ss  for ss,nn in zip(allfnames, trialtypes) if nn==numtype])
##    biases.append([ss  for ss,nn in zip(allbiases, trialtypes) if nn==numtype])
#
#
