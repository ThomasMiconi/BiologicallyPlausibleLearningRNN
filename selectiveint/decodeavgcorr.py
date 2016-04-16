# -*- coding: utf-8 -*-


# In this version, we don't attempt to classify - rather, we simply avrage (over many splits) the correlation between a 
# 'test' population vector and the average of population vectors at a given time for the correct to-be-decoded quantity
# Problem: There's a lot of high-correlation over the graphs !


from pylab import *
import numpy as np
import scipy as sp
import glob

if 1:
    datalist=[] 
    labellist=[]
    for trialtype in range(4):
        print "Loading data for trial type "+str(trialtype)
        fnames = glob.glob('rs_long_type'+str(trialtype)+'*.txt')
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
# 160 trials in total, 40 per 'condition' (i.e. each of 4 possible input combinations)


# The random noise is there to avoid the large effects of small fluctuations caused by a few individual trials when the averages are supposed 
#to be zero - but actually are not, in a way that spuriously correlates with future, significant trajectories
# To understand better: look at matshow(corrcoef(avgs[:,:,1]-avgs[:,:,0])), for sufficiently large values of splitsize, w/o noise
                
matdata += 0.0 * standard_normal(shape(matdata))

conds = np.array(labellist)

NBTRIALS = conds.shape[0]
trialtime = matdata.shape[0]
SPLITSIZE = 3

# 18, not 19, 14 (big dark patches!), 10 seems usable even without adding Noise !?
# 10 is best...

# For each of all four graphs that we intend to make, we specify an appropriate label for all 4 conditions:
# The precise values of the labels are unimportant. What matters is that conditions that are "the same" for this graph 
# should have the same label (e.g. if you want to decode 1st stimulus identity, conditions with same 1st stimulus but different 
# 2nd stimulus should have same label)
labelsforeachgraph = np.array([ [10,10,11,11], [10,11,10,11], [10,11,11,10], [10, 11,12,13] ])
fnames = ['accur_stim1.txt', 'accur_stim2.txt', 'accur_xor.txt', 'accur_cond.txt']

for numgraph in range(4):
    print "numgraph: " + str(numgraph)
    labels=labelsforeachgraph[numgraph,:]
    correctguesses=[]
    allavgs=[]
    corrswithcorrect = []
    for numiter in range(100):
        #print numiter
        learnallconds = []
        testallconds=[]
        labelslearnlist=[]
        labelstestlist=[]
        for trialtype in range(4):
            # Collect all the trials for this particular condition
            datathiscond = matdata[:,:,(conds == trialtype)]; 
            selects = permutation(arange(datathiscond.shape[2]))[0:SPLITSIZE]  # Select 5 random trials for this condition
            # 4 of those will be the 'learning' trials
            learnthiscond = datathiscond[:,:,selects[1:]]
            # 1 of those will be the 'test' trial
            testthiscond = datathiscond[:,:,selects[0]]
            # Append the learning/testing trials to learnallconds / testallconds...
            learnallconds.append(learnthiscond)
            testallconds.append(testthiscond)
            # And give the proper label to these trials (depending on the condition, and what we want to decode)
            labelslearnlist.append((labels[trialtype],) * (SPLITSIZE - 1))
            labelstestlist.append(labels[trialtype])
        
        labelstest = np.array(labelstestlist)
        labelslearn = hstack(labelslearnlist)
        learndata = dstack(learnallconds)
        testdata = dstack(testallconds)
        
        avgslist=[]
        # For each label, average all the training trials with this label
        for lbl in unique(labelslearn):
            avgslist.append(mean(learndata[:,:,labelslearn == lbl], axis=2))
        avgs = dstack(avgslist)
        #allavgs.append(avgs)
        
        # Compare each row of test trial to each row of all avgs (either 2 or 4 of them). 
        # Remember that a row is a population activity vector at a given time, which is what you want: you compare the population vector of test 
        # to the average population vectors of the training set for each condition label.
        # The result is a set of NON-symmetric matrices in which row,col indicates the correlation b/w avg-train at time row and test 
        # pop vector at time col (it's not symmetric because we extract the relavant quadrant from the overall, symmetric corrlation matrix b/w the rows 
        # of avg-train and test)
        # row,col = corr coeff between row 'row' of x1 and row 'col' of x2 (when we extract this quadrant). Verify by using simple matricesin which one row (different in each 
        # matrix) has identical values
        for numtest in range(testdata.shape[2]):
            corrsperclasslist=[]
            for numlearn in range(avgs.shape[2]):
                corrsperclasslist.append(corrcoef(avgs[:,:,numlearn], testdata[:,:,numtest]) [0:trialtime, trialtime:])
            corrsperclass = dstack(corrsperclasslist)
            corrswithcorrect.append(corrsperclass[:,:,unique(labelslearn) == labelstest[numtest]])
    
    cwc = dstack(corrswithcorrect)
    accur = mean(cwc, axis=2)      
    savetxt(fnames[numgraph], accur)
