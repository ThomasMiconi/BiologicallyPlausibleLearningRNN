This is the code for the selective attention and integration experiment, based on Mante,
Sussillo, Shenoy and Newsome, Nature 2013 (see also Song, Yang and Wang, PLOS Comp. Biol.
2016).

As in the dnms/ directory, the main code is in `net.cpp`, with other files
mostly concerned with analysis and figure plotting. You can compile and run the
program as is. It will train a network to perform the Mante-Sussillo task and
store weights and errors in various output files.

##How to generate the figures:

(Note that these scripts require access to an LSF cluster)

    g++ -O3 -lm -I ../../Eigen net.cpp -o net -std=c++11    # To compile the code. Adapt as needed.
    ./subsmall.sh               # Runs the code with 20 different random seeds. Requires an LSF cluster!
    python makefigsingle.py     # Plots the training error curve, using the median and inter-quartile range across 20 runs
    ./subtest.sh                # Uses the stored weights of the trained network to generate test data over many trials. Requires an LSF cluster!
    python plotdetection.py     # Generates the psychometric curves
    python rgrss.py             # Plots the  Mante-Sussillo trajectories



In IPython:

    %run plotdecode.py
    %run rgrss.py

