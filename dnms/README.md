This is the code for the delayed nonmatch-to-sample task.
The main source code is in `net.cpp`. Other files are either analysis and processing files, or for submission to a cluster.

`net.cpp` implements the method described in http://biorxiv.org/content/early/2016/06/07/057729. However, it also contains (commented-out) code for for the
faster, but less biologically plausible node-perturbation method (Fiete & Seung
2006 - see also http://arxiv.org/abs/1507.08973).

The node-perturbation code is commented out by default. If you are only
interested in training RNNs and don't care much about biological plausibility,
*you should use node-perturbation*, since it is considerably faster and should
produce similar performance (the other method largely reproduces
node-perturbation through more biologically plausible means). See the code for more details.


## How to run the code to generate the figures from the preprint:

The code can be compiled with `g++ -O3 -lm -I ../../Eigen net.cpp -o net -std=c++11`. Be sure to adapt the directory where you installed the Eigen library.

To run the code, just type `./net`. It should be reaching criterion (95% of errors < .5) after a few minutes (by 1000 trials), but let it go to the full ~20000 trials to obtain good convergence.

To generate 20 runs for the error curve, I use `./subsingle.sh` which submits 20 jobs to a local cluster. You probably want to adapt it. 

Then run `./net test` to generate output data files (it should complete in a few minutes)

In python (IPython):

First, type `ls -ltr rs_long_type1_1_*SEED1.txt`. There should only be one result! 

This generates the data files for the dynamic encoding figure
`%run decodetomdist.py`

Generate actual figures:

````
%run ploterr_single.py  
(note: the above should output "1 Graphs")
%run plotdecode.py
%run plotmds.py
````

