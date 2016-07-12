This is the code for the arm-model experiment.

In addition to the Eigen library, you will need to download and compile the Opensim 3.3 library (and its prerequisite, the SimBody library). You need the 3.3 version - it will not work with any other version. 

Be warned: this is a major pain. 

If you somehow manage to compile the OpenSim library and run some of the example programs included with it (in particular the ExampleMain program, including the graphics visualizer), you should be ready to go.

1. Unzip BONEFILE.zip

2. To compile, type `g++ -lm armmodel.cpp -o armmodel -std=c++11 -I ../../../opensim3.3/include/SimTK/ -I ../../include    -L/opensim3.3/lib  -L/Users/miconi/exp/muscle/opensim3.3/bin  -losimSimulation -losimActuators -losimCommon -losimAnalyses -losimTools -lSimTKcommon -lSimTKmath -lSimTKsimbody` (be sure to adapt the directories to wherever you installed Opensim 3.3).


3. To visualize the results of the saved connection matrices included (J.dat, J.txt, win.dat and win.txt) run `./armmodel TEST` (case sensitive). You should see a window pop up with an arm-waving skeleton in it.

4. To start the learning, just run `./armmodel`. It will periodically save the learned connectivity in J.dat and win.dat. If it crashes too much, try to adapt and use `runwithreloads.sh`. NOTE: the program is extremely slow and takes several days to run to completion.


The main code is in armmodel.cpp. The file MyArm26.osim contains a simplified version of various Opensim-compatible models of the human upper arm, as cited in the file.

