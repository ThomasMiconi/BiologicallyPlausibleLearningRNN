# -*- coding: utf-8 -*-

# Decoding method from Meyers et al., also used by Stokes et al. 2013.
# Basically, try to decode a certain task feature at time point Ttest based on
# a classifer trained with data from time point Ttrain.  
#
# Before running this, you must run subtest.sh first to generate the data.
#
# For a certain number of 'splits', divide the whole (randomly permuted) dataset into a training
# set and a test set. Use the training set to compute the average activity of
# each neuron at each time point over all trials sharing a certain feature
# (same 1st stimulus, same 2nd stimulus, same final response). Then take every
# test trial and compute the correlations between population vector at each
# time point Ttest of the test trial and population vector at each time point Ttrain in each of the
# averages. At any pair of time points Ttrain, Ttest, the decoded feature value is the one
# for which the average population at time Ttrain has highest correlation with the test trial population
# vector at time Ttest.
#
# The final accuracy at any pair of time points Ttrain, Ttest is the percentage
# correct over all splits for this pair of time points.



from pylab import *
import numpy as np
import scipy as sp
from scipy import spatial
import glob

if 1:
    datalist=[] 
    labellist=[]
    for trialtype in range(4):
        print "Loading data for trial type "+str(trialtype)
        fnames = glob.glob('rs_long_type'+str(trialtype)+'*SEED1.txt')
        for nm in fnames:
            r = loadtxt(nm)
            #z = r.reshape((110,10,200))
            #z = sum(z,axis=1)
            z = r[0::10,:]
            datalist.append(z)
            #labellist.append([trialtype]*r.shape[0])
            labellist.append(trialtype)
    
#0 0
#0 1
#1 0
#1 1

matdata = dstack(datalist) ; #+ .5  * standard_normal(matdata.shape) 

# matdata contains the data in dimensions of time x neuron x trial
# 160 trials in total, 40 per 'condition' (i.e. each of 4 possible input combinations, AA AB BA BB)

conds = np.array(labellist)

NBTRIALS = conds.shape[0]
trialtime = matdata.shape[0]


# For each of all four graphs that we intend to make, we specify an appropriate label, indicating the value of the to-be-decoded quantity,
# for each of the 4 possible conditions (i.e. all 4 possible combinations of inputs, AA, AB, BA, BB):
# The precise values of the labels are unimportant. What matters is that conditions that are "the same" for this graph 
# should have the same label (e.g. if you want to decode 1st stimulus identity, conditions with same 1st stimulus but different 
# 2nd stimulus should have same label)
# For example, when decoding first stimulus, 1st and 2nd condition have same label (because they have same 1st stimulus A), and so do 3rd and 4th condition (same 1st stimulus B)
labelsforeachgraph = np.array([ [1,1,2,2], [1,2,1,2], [1,2,2,1], [1, 2, 3, 4] ])
fnames = ['accur_stim1.txt', 'accur_stim2.txt', 'accur_xor.txt', 'accur_cond.txt']

for numgraph in range(4):
    print "numgraph: " + str(numgraph)
    labels=labelsforeachgraph[numgraph,:]

    accurs = []
    for numsplit in range(100):
        print "Split:", numsplit 

        # We build a randomized-order copy of the data
        myorder = np.random.permutation(matdata.shape[2])

        matdatar = matdata[:, :, myorder]
        condsr = conds[myorder]

        correctguesses=[]
        allavgs=[]

        # Half of the data is the training set. The other half is the test set (the one in which we do the decoding).
        traindata = matdatar[:,:,::2]
        trainconds = condsr[::2]
        trainlabels = [labels[c] for c in trainconds]

        testdata = matdatar[:,:,1::2]
        testconds = condsr[1::2]
        testlabels = [labels[c] for c in testconds]
            
        avgslist=[]
        # For each label, average all the training trials with this label
        allperlabellist = []
        for lbl in unique(trainlabels):
            avgslist.append(mean(traindata[:,:,trainlabels == lbl], axis=2))
            allperlabellist.append(traindata[:,:,trainlabels == lbl])
        avgs = vstack(avgslist)
        #allavgs.append(avgs)
        
        # Here, normally avgslist should be a list of matrices Time x Neur, one per possible label, each containing the average of all trials for this label.
        # avgs represents the same data but stacked vertically, i.e. (NbClasses*Time) x Neur 

        correctguesses = []
        guessedclasses = []

        # Compute the correlation between the rows of each test trial and the average trials for each label.
        # This measures the correlation between the population at any time point in the test trial and the population at every time point in the average, conveniently stored as a square matrix.
        for numtrial in range(testdata.shape[2]):  # Iterate over all test trials
            distsperclasslist=[]
            for numclass in range(len(avgslist)):
                # This compares the test trial vs the average of all training trials for this particular class:
                distsperclasslist.append(sp.spatial.distance.cdist(avgslist[numclass], testdata[:,:, numtrial], 'correlation'))   # Shape of output: Time in the Average for this class x Time in the test trial
                # This compares the test trial vs a single randomly chosen training trial for this particular class:
                #distsperclasslist.append(sp.spatial.distance.cdist(allperlabellist[numclass][:,:,np.random.randint(allperlabellist[numclass].shape[2])], testdata[:,:, numtrial]))   # Shape of output: Time in the Aveage for this class x Time in the test trial
            distsperclass = dstack(distsperclasslist)
            
            # At every point TimeAvg x TimeTestTrial, find the label / class at that TimeAvg that has minimum distance from this trial at TimeTestTrial. Determine whether it is correct and store the resulting matrix.
            guessedclass = argmin(distsperclass, axis=2)
            #correctguess =unique(labelslearn)[guessedclass] == labelstest[numtest]
            correctguess =unique(trainlabels)[guessedclass] == testlabels[numtrial] 
            correctguesses.append(correctguess)
            guessedclasses.append(guessedclass)
        
        cg = dstack(correctguesses)
        accur = mean(cg, axis=2)      
        accurs.append(accur)  # Store the mean accuracy matrix over all test trials for this split

     
    savetxt(fnames[numgraph], np.mean(dstack(accurs), axis=2)) 
