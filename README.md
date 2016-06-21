This is the source code for the paper ["Biologically plausible learning in recurrent neural networks for flexible decision tasks"](http://biorxiv.org/content/early/2016/06/07/057729).

The source code is in C++ and requires the Eigen library (don't worry, it's a source-only library so installation is a breeze) and a C++11 compliant compiler.

The `dnms` directory contains the delayed non-match to sample experiment. The
`selectiveint` directory contains the selective attention/integration
experiment. In both case, the main program is the `net.cpp` file. Python files
in both directories are for analysis and figures.

The arm control experiment is not currently included but will be soon.

More information to come!

