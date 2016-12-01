# Submit jobs to the cluster. 

# /opt/python-2.7.10/bin/python


import sys
import os
import shutil

allopts = [

        # ALPHAMODUL 8.0 is clearly bad !

        #"ETA .01",   # This is the original
        #"ETA .01 TAUINPUT 3.0",   
        #"ETA .01 MAXDW 2e-4",   
        #"ETA .01 MAXDW 2e-4 TAUINPUT 3.0",   
        #"ETA 0.003 MAXDW 5e-5 TAUINPUT 1.0",
        #"ETA 0.003 MAXDW 5e-5 TAUINPUT 3.0",
        #"ETA .03",   
        #"ETA .03 TAUINPUT 3.0",   

        
        #"ETA .01 ALPHAMODUL 30.0",   
        #"ETA .01 ALPHAMODUL 30.0 TAUINPUT 3.0",   
        #
        #"ETA .01 ALPHAMODUL 30.0 MAXDW 2e-4",   
        #"ETA .01 ALPHAMODUL 30.0 MAXDW 2e-4 TAUINPUT 3.0",   

        #"ETA .005 ALPHAMODUL 30.0",   
        #"ETA .005 ALPHAMODUL 30.0 MAXDW 2e-4",   
        #"ETA .005 ALPHAMODUL 30.0 MAXDW 1e-4",   

       
        "ETA .01 ALPHAMODUL 30.0 ALPHABIAS .5",
        "ETA .01 ALPHAMODUL 30.0 MAXDW 2e-4 ALPHABIAS .5" ,
        "ETA .005 ALPHAMODUL 30.0 ALPHABIAS .5",
        "ETA .005 ALPHAMODUL 30.0 MAXDW 2e-4 ALPHABIAS .5",
        "ETA .005 ALPHAMODUL 30.0 MAXDW 1e-4 ALPHABIAS .5",


        ]


for optionz in allopts:

    dirname = "trial-" + optionz.replace(' ', '-')

    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    os.chdir(dirname)
    print os.getcwd()

    for v in range(20):
        #os.mkdir("v"+str(v))
        #os.chdir("v"+str(v))
        CMD = "bsub -q short -W 8:00 -eo e.txt -g /net ../net " + optionz + " RNGSEED " + str(v)
        #CMD = "bsub -q short -W 4:00 -eo e.txt -oo o.txt -g /rnn /opt/python-2.7.10/bin/python ../../rnn.py " + optionz + " RNGSEED " + str(v)
        #CMD = "bsub -q short -W 6:00 -eo e.txt -oo o.txt -g /rnn /opt/python-2.7.10/bin/python ../../min-char-rnn-param.py " + optionz + " RNGSEED " + str(v) # For fixed-size
        #print CMD
        retval = os.system(CMD)
        print retval
        #os.chdir('..') 
    
    os.chdir('..') 


    #print dirname
    #for RNGSEED in range(2):
    #st = "python rnn.py COEFFMULTIPNORM " + str(CMN) + " DELETIONTHRESHOLD " + str(DT) + " MINMULTIP " \
    #+ str(MMmultiplierofDT*DT) + " PROBADEL " + str(PD) + " PROBAADD " + str(PAmultiplierofPD * PD) \
    #+ " RNGSEED " + str(RNGSEED) + " NUMBERMARGIN " + str(NM)




