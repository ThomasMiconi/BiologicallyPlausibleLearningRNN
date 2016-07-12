This is the source code for the preprint ["Biologically plausible learning in recurrent neural networks for flexible decision tasks"](http://biorxiv.org/content/early/2016/06/07/057729).

This code implements a simple, biologically plausible plasticity rule for chaotic recurrent neural networks (RNNs), that can learn nontrivial tasks based solely on delayed, sparse rewards.

The source code is in C++ and requires a C++11-compliant compiler. It also
requires the [Eigen
library](http://eigen.tuxfamily.org/index.php?title=Main_Page) (version 3),
which is very easy to install since it is a source-only library. In addition,
Python files are used to generate test data and plot the figures.

The arm-model experiment also requires downloading and compiling the [Opensim 3.3 library](https://simtk.org/frs/?group_id=91) *from source* (and its prerequisite, the SimBody library), which is *not at all* easy to install. Attempt at your own risk.

The `dnms` directory contains the delayed non-match to sample experiment. In
addition to the method described in the preprint, this code also contains an implementation of the 
node-perturbation method (introduced by
Fiete and Seung 2006, and applied to RNNs as described in [this previous
preprint](https://arxiv.org/abs/1507.08973)), which is much faster, but less biologically plausible. See the code for details.

The
`selectiveint` directory contains the selective attention/integration
experiment. 

The `armmodel` directory contains the arm control experiment. 


