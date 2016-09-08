This is the code for the selective attention and integration experiment, based on Mante,
Sussillo, Shenoy and Newsome, Nature 2013 (see also Song, Yang and Wang, PLOS Comp. Biol.
2016).

As in the dnms/ directory, the main code is in `net.cpp`, with other files mostly concerned with analysis and figure plotting.

##How to generate the data for the figures:

    g++ -O3 -lm -I ../../Eigen net.cpp -o net -std=c++11
    bsub -q short -g /net -oo output1.txt -eo error1.txt  -W 10:00 ./net 

    ./subtest.sh

In IPython:
    %run plotdecode.py
    %run rgrss.py

