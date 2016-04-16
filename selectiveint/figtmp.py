splts = []
linez = []
matplotlib.rcParams.update({'font.size': 10})
for numgraph in range(4):
    splts.append(subplot(2, 2, numgraph+1))
    title('Context: Attend Mod. ' + str(numgraph % 2 + 1) +'\n' + 'Grouping by Mod. ' + str(int(numgraph/2) +1) +' bias')
    xlabel('Choice axis' )
    ylabel('Mod. ' + str(1 + int(numgraph/2)) + ' axis')
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

