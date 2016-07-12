
#include <OpenSim/OpenSim.h>
#include "../../../../../../orchestra/Eigen/Eigen/Dense"
#include <ctime>  // clock(), clock_t, CLOCKS_PER_SEC
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <string>
#include <random>

#define TESTING 777
#define LEARNING 666


#define NBTRIALTYPES 2


using namespace OpenSim;
using namespace SimTK;
using namespace Eigen;
using namespace std;

int stepCount = 0;
double initialTime = 0.0;
double finalTime = .700;

void saveWeights(MatrixXd& m, string fname);
void readWeights(MatrixXd& m, string fname);
void randJ(MatrixXd& m);


int NBNEUR = 400;
int NBIN = 2;  
int NBOUT = -1;
double PROBACONN = 1.0;
double G = 1.5;
string METHOD = "DELTAX"; // "DELTAX"; //"DELTATOTALEXC"; //"DXTRIAL";
string MODULTYPE = "DECOUPLED";
int RNGSEED = 1;

double ALPHAACTIVPEN = .8;

int DEBUG = 0;

double PROBAMODUL = .003;
double ALPHAMODUL = 10.0;

double PROBAHEBB = 1.0;
double ALPHABIAS = .0; //.01

double ALPHATRACE = .5;
double ALPHATRACEEXC = 0.0;

int SQUARING = 1;

double MAXDW = 5e-5 ; //* 1.5;
double ETA =  .5 ; // * 1.5;  // Learning rate
double INPUTMULT = 2.0;
//double STIMVAL = .5;

int NBTRIALS = 1000;
int RELOAD = 0;
int RANDW = 0;

std::default_random_engine myrng;
std::normal_distribution<double> Gauss(0.0,1.0);
std::uniform_real_distribution<double> Uniform(0.0,1.0);

int main(int argc, char* argv[])
{

    fstream myfile;

    double dt = 1.0;
    double tau = 30.0;

    int PHASE=LEARNING;
    if (argc > 1)
        for (int nn=1; nn < argc; nn++)
        {
            if (strcmp(argv[nn], "TEST") == 0) { PHASE = TESTING; cout << "Test mode. " << endl; }
            if (strcmp(argv[nn], "RANDW") == 0) { RANDW = 1; cout << "Random (i.e. initial) weights. " << endl; }
            if (strcmp(argv[nn], "RELOAD") == 0) { RELOAD = 1; cout << "Reloading! " << RELOAD << endl; } 
            if (strcmp(argv[nn], "METHOD") == 0) { METHOD = argv[nn+1]; }
            if (strcmp(argv[nn], "MODULTYPE") == 0) { MODULTYPE = argv[nn+1]; }
            if (strcmp(argv[nn], "SQUARING") == 0) { SQUARING = atoi(argv[nn+1]); }
            if (strcmp(argv[nn], "NBTRIALS") == 0) { NBTRIALS = atoi(argv[nn+1]); }
            if (strcmp(argv[nn], "DEBUG") == 0) { DEBUG = atoi(argv[nn+1]); }
            if (strcmp(argv[nn], "G") == 0) { G = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "ALPHABIAS") == 0) { ALPHABIAS = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "ETA") == 0) { ETA = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "TAU") == 0) { tau = atof(argv[nn+1]); }
            //if (strcmp(argv[nn], "STIMVAL") == 0) { STIMVAL = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "INPUTMULT") == 0) { INPUTMULT = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "ALPHAMODUL") == 0) { ALPHAMODUL = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "PROBAMODUL") == 0) { PROBAMODUL = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "PROBAHEBB") == 0) { PROBAHEBB = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "ALPHATRACE") == 0) { ALPHATRACE = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "ALPHATRACEEXC") == 0) { ALPHATRACEEXC = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "RNGSEED") == 0) { RNGSEED = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "MAXDW") == 0) { MAXDW = atof(argv[nn+1]); }
        }

    string SUFFIX = "_G" + to_string(G) + "_MAXDW" + to_string(MAXDW) + "_ETA" + to_string(ETA) + "_ALPHAMODUL" + to_string(ALPHAMODUL) + "_PROBAMODUL" + to_string(PROBAMODUL) + "_SQUARING" +to_string(SQUARING) + "_MODULTYPE-" + MODULTYPE +   
        "_ALPHATRACE" + to_string(ALPHATRACE) + "_METHOD-" + METHOD + "_ALPHABIAS" + to_string(ALPHABIAS) + "_PROBAHEBB" + to_string(PROBAHEBB) + "_ATRACEEXC" + to_string(ALPHATRACEEXC) + "_TAU" + to_string(tau) 
        + "_INPUTMULT" + to_string(INPUTMULT)
        + "_RNGSEED" + to_string(RNGSEED);
    cout << SUFFIX << endl;

    myrng.seed(RNGSEED);
    srand(RNGSEED);

    int TRIALTIME = 700;
    //int STARTSTIM2 = 400, TIMESTIM2 = 200; 
    /*int TRIALTIME = 1000;
      int STARTSTIM1 = 1, TIMESTIM1 = 200; // 200
      int STARTSTIM2 = 400, TIMESTIM2 = 200; */

    int marker =0;
    if (PHASE == TESTING) 
        NBTRIALS = 40*NBTRIALTYPES;

    MatrixXd dJ(NBNEUR, NBNEUR);
    MatrixXd win(NBNEUR, NBIN); 
    //cout << win.col(0).head(5) << endl;
    MatrixXd J(NBNEUR, NBNEUR);

    VectorXd errs(NBTRIALS); errs.setZero();
    VectorXd lateral_input;
    VectorXd total_exc(NBNEUR), total_exc_prev(NBNEUR);
    VectorXd delta_r(NBNEUR), delta_r_sq(NBNEUR), r_trace(NBNEUR), r_trace2(NBNEUR);
    r_trace.fill(0); r_trace2.fill(0);
    VectorXd delta_x(NBNEUR), delta_x_sq(NBNEUR), x_trace(NBNEUR), x_trace2(NBNEUR), delta_x_cu(NBNEUR);
    total_exc.fill(0); total_exc_prev.fill(0); x_trace.fill(0); x_trace2.fill(0);
    VectorXd modul(NBNEUR); modul.setZero();
    VectorXd modul_trace(NBNEUR); modul_trace.setZero();
    VectorXd dxthistrial(NBNEUR);  dxthistrial.setZero();
    MatrixXd rs(NBNEUR, TRIALTIME); rs.setZero();
    MatrixXd hebb(NBNEUR, NBNEUR);  
    VectorXd x(NBNEUR), r(NBNEUR), rprev(NBNEUR), dxdt(NBNEUR), k(NBNEUR), 
             input(NBIN), deltax(NBNEUR);
    x.fill(0); r.fill(0);

    VectorXd meanerrtrace(NBTRIALTYPES);

    MatrixXd dJtmp, Jprev, Jr;


    double hebbmat[NBNEUR][NBNEUR];
    double rprevmat[NBNEUR], dx2[NBNEUR];
    double xmat[NBNEUR], xtracemat[NBNEUR];

    double dtdivtau = dt / tau;

    
    // Positions of the targets
    double PosTgt1[3] = { .4, 0, .2 };
    double PosTgt2[3] = { .4, 0, -.2 };

    // Initializations for the muscle stuff
    
    int VISUAL = 0;
    if (PHASE == TESTING)
        VISUAL = 1;

    Model osimModel("MyArm26.osim");
    //Model osimModel("tmp.osim");
    if (VISUAL)
        osimModel.setUseVisualizer(true);
    State& si = osimModel.initSystem();

    PrescribedController *muscleController = new PrescribedController();
    muscleController->setActuators(osimModel.updActuators());

    const Set<Actuator>& actuators = osimModel.getActuators();
    int NBMUSCLES = actuators.getSize();
    NBOUT = NBMUSCLES;


    //win.setRandom(); win.topRows(NBOUT).setZero() ; 
    win.setRandom(); win.topRows(NBNEUR/2).setZero() ; 


    for(int i=0; i<actuators.getSize(); ++i){
        muscleController->prescribeControlForActuator(i, new Constant(.01));
    }
    // Add the muscle controller to the model
    osimModel.addController(muscleController);
    VectorXd MIFs(NBMUSCLES);
    const Set<Muscle> &muscleSet = osimModel.getMuscles();
    if (muscleSet.getSize() != NBMUSCLES)
        throw std::runtime_error("Number of muscles different from number of actuators!");
    for(int i=0; i< muscleSet.getSize(); i++ ){
        double MIF = muscleSet[i].getMaxIsometricForce();
        cout << "Max Isometric Force for muscle " << i << ": " << MIF << endl;
        MIFs(i) = MIF;
    }
    MIFs /= MIFs.sum();

    osimModel.equilibrateMuscles(si);
    //SimTK::RungeKuttaMersonIntegrator integrator(osimModel.getMultibodySystem());
    /*
    // OK but too loose? Still quite a lot of nans, no t1>t2 error.
    SimTK::SemiExplicitEuler2Integrator integrator(osimModel.getMultibodySystem());
    integrator.setAccuracy(1.0e-2);
    integrator.setMinimumStepSize(1.0e-6);
    */
    /*
    //Blocks on test
    SimTK::ExplicitEulerIntegrator integrator(osimModel.getMultibodySystem(), 1e-5); 
    integrator.setAccuracy(1.0e-5); 
    */
    
    //SimTK::RungeKuttaMersonIntegrator integrator(osimModel.getMultibodySystem());
    //integrator.setAccuracy(1.0e-5); 
    // This (with a .9 multiplier at net-to-muscle translation) seems to minimize the funky stuff while having good perf?.. Still occasional funky stuff (but no NaNs)
    // Still occasionally blocks !
    /*SimTK::SemiExplicitEuler2Integrator integrator(osimModel.getMultibodySystem());
    integrator.setAccuracy(1.0e-3);
    integrator.setMinimumStepSize(1.0e-7);*/

    SimTK::SemiExplicitEuler2Integrator integrator(osimModel.getMultibodySystem());
    integrator.setAccuracy(1.0e-3);
    integrator.setMinimumStepSize(1.0e-6);
    // Separating the forces for both obstacles didn't either...
    // Try increasing precision from 1e-5 to 1e-6: Still error at  187th trial !!! 
    // Reduce stiffness of the contact force from 10eN to .8: Simply cuts through the spheres..
    // Try with 1e-7: fails even earlier ! (133 i o 187... but after a few trial where it failed later...)
    Manager manager(osimModel,  integrator);



    // Position of the finger tip in the reference frameof the ulna-radius-hand body
    SimTK::Vec3 fingertippos(.02, -.4, .1);

    /*
    PointKinematics* fingerkin = new PointKinematics(&osimModel);
    fingerkin->setName("fingerkin");
    fingerkin->setBodyPoint("r_ulna_radius_hand", fingertippos);
    fingerkin->setPointName("pelvis");
    model.addAnalysis(fingerkin);
*/

    PointKinematics fingerkin(&osimModel);
    fingerkin.setName("fingerkin");
    fingerkin.setBodyPoint("r_ulna_radius_hand", fingertippos);
    fingerkin.setPointName("fingertip");
    osimModel.addAnalysis(&fingerkin);


    osimModel.printDetailedInfo(si, cout);

    // Put the target balls in the visualizer. 
    // Apparently y is the vertical coordinate. x is the 'straight ahead' direction (from the skeleton's perspective), z the lateral one. Right-hand rule seems respected.
    if (VISUAL)
    {
        //osimModel.updVisualizer().updSimbodyVisualizer().addDecoration(GroundIndex, SimTK::Transform(Vec3(0,0,0)), DecorativeSphere(.05).setColor(Blue).setOpacity(1));
        osimModel.updVisualizer().updSimbodyVisualizer().addDecoration(GroundIndex, SimTK::Transform(Vec3(PosTgt1[0], PosTgt1[1], PosTgt1[2])), DecorativeSphere(.05).setColor(Cyan).setOpacity(1));
        osimModel.updVisualizer().updSimbodyVisualizer().addDecoration(GroundIndex, SimTK::Transform(Vec3(PosTgt2[0], PosTgt2[1], PosTgt2[2])), DecorativeSphere(.05).setColor(Red).setOpacity(1));
    }




    randJ(J);
    if ( ((PHASE == TESTING) && (RANDW == 0)) ||  (RELOAD) )  {
        cout << "Reading from files " << "J" + SUFFIX + ".dat" << " and " << "win" + SUFFIX + ".dat" << endl;
        readWeights(J, "J.dat");
        readWeights(win, "win.dat");
        //cout << win.row(NBNEUR-2) << endl;
        //readWeights(J, "J" + SUFFIX + ".dat");
        //readWeights(win, "win" + SUFFIX + ".dat"); // win doesn't change over time.
    }
    meanerrtrace.setZero();


        // Main loop

        for (int numtrial=0; numtrial < NBTRIALS; numtrial++)
        {
            cout << "Trial " << numtrial << ": ";

            try{
                int trialtype = numtrial % NBTRIALTYPES;
                for (int n1=0; n1 < NBNEUR; n1++)
                    for (int n2=0; n2 < NBNEUR; n2++)
                        hebbmat[n1][n2] = 0;
                hebb.setZero();
                //input = patterns.col(trialtype);
                input.setZero();
                


                // Initial randomization
                //x.fill(0.0); 
                x.setRandom(); x *= .1; 
                //x.setRandom(); x *= 1.0; 
                
                
                // Special init for the muscle outputs, since they will be shifted up later..
                //x.head(NBMUSCLES).fill(-2.0);
                

                
                // Biases
                x(NBNEUR-3)=1.0; x(NBNEUR-4)=1.0; x(NBNEUR-5)=-1.0; //x(12) = 1.0; 
                //x(NBOUT + 3)=1.0; x(NBOUT + 4)=1.0; x(NBOUT + 5)=-1.0; //x(12) = 1.0; 

                for (int nn=0; nn < NBNEUR; nn++)
                    r(nn) = tanh(x(nn));


                // First, run the network!
                for (int numiter=0; numiter < TRIALTIME;  numiter++)
                {
                    // Left- or Right- trial?
                    input(trialtype) = 1.0; input(1-trialtype) = 0.0; 
 
                    rprev = r;
                    lateral_input =  J * r;

                    total_exc =  lateral_input  + win * input ;
                    //total_exc =  lateral_input /*+ wfb * zout *//* + win * input + dxthistrial */  ; 

                    // Modulation !
                    modul.setZero();
                    if (MODULTYPE == "UNIFORM")
                    {
                        if ( (Uniform(myrng) < PROBAMODUL)  && (marker == 0) // No back-to-back perturbations !
                                && (numiter> 3))
                        {
                            modul.setRandom();
                            modul *= ALPHAMODUL;
                            total_exc +=   modul;
                            marker = 1;
                        }
                        else
                            marker = 0;
                    }
                    else if (MODULTYPE == "DECOUPLED")
                    {
                        for (int nn=0; nn < NBNEUR; nn++)
                            if ( (Uniform(myrng) < PROBAMODUL)
                                    && (numiter> 3)
                               )
                            {
                                //total_exc(nn) += ALPHAMODUL * (-1.0 + 2.0 * Uniform(myrng));
                                total_exc(nn) += ALPHAMODUL * (Gauss(myrng));
                            }
                    }
                    else { throw std::runtime_error("Which modulation type?"); }


                    x += dtdivtau * (-x + total_exc);



                    //x(NBOUT + 3)=1.0; x(NBOUT + 4)=1.0; x(NBOUT + 5)=-1.0; //x(12) = 1.0; 
                x(NBNEUR-3)=1.0; x(NBNEUR-4)=1.0; x(NBNEUR-5)=-1.0; //x(12) = 1.0; 
                    for (int nn=0; nn < NBNEUR; nn++)
                    {
                        r(nn) = tanh(x(nn));
                    }


                    // Muscle activators are semi-rectified, strictly positive
                    r.head(NBMUSCLES) = r.head(NBMUSCLES).cwiseMax(0);
                    //r.head(NBMUSCLES).array() += 1.0;
                    //r.head(NBMUSCLES).array() /= 2.0;
                   
                   // Inputs delivered to only a few neurons. 
                    //r(NBNEUR-1)=0; r(NBNEUR-2)=0;
                    //r(NBNEUR-1 - trialtype) = 1.0 * INPUTMULT;

                    //r.tail(2 * 5).setZero();
                    //r.tail(2*5).head(5).fill(trialtype);
                    //r.tail(5).fill(1 - trialtype);

                    // Store the responses of all neurons for this timestep
                    rs.col(numiter) = r;


                    delta_x =  x  - x_trace ;
                    delta_x_sq = delta_x.array() * delta_x.array().abs(); // Sign-preserving squaring. Also works, but need to adjust parameters
                    delta_x_cu = delta_x.array() * delta_x.array() * delta_x.array();
                    x_trace = ALPHATRACEEXC * x_trace + (1.0 - ALPHATRACEEXC) * x;


                    if (DEBUG > 0)
                    {
                        if ((marker == 1)  || (DEBUG == 2))
                            cout << delta_x_sq.norm() << " " << modul.norm() << " " << total_exc.norm() << " " << x.norm() << " " << (total_exc - x).norm() << " Alignment deltax/total_exc (inc. modul):" 
                                << delta_x.dot(total_exc) / (delta_x.norm() * total_exc.norm()) 
                                   << " Alignment deltax/modul:" << delta_x.dot(modul) / (delta_x.norm() * modul.norm()) << " Align deltax_sq/modul:" << delta_x_sq.dot(modul) / (delta_x_sq.norm() * modul.norm()) <<  endl;
                    }

                    if ((PHASE == LEARNING) && (Uniform(myrng) < PROBAHEBB)  && (numiter> 2))
                    {
                        if (METHOD == "DELTAX")
                        {
                            for (int n1=0; n1 < NBNEUR; n1++)
                            {
                                rprevmat[n1] = rprev(n1);
                                dx2[n1] = delta_x_cu(n1);
                            }
                            for (int n1=0; n1 < NBNEUR; n1++)
                                for (int n2=0; n2 < NBNEUR; n2++)
                                    hebbmat[n1][n2] += rprevmat[n1] * dx2[n2]; //rprev(n1) * delta_x_sq(n2);
                        }
                        else { cout << "Which method??" << endl; return -1; }
                    }


                }

                // Now we need to evaluate this network run. So we take out the stored activations of the output neurons over the course of the run, and use it to guide the muscles of the musculo-skeletal model.

                // Load the initial state (arm extended along the thorax, palm front-facing)
                State s = si;

                for(int i=0; i < NBMUSCLES; i++){
                    int NBSTEPS = 100;
                    double time[NBSTEPS];
                    double value1[NBSTEPS];
                    for (int nn=0; nn < NBSTEPS; nn++)
                    {
                        value1[nn] = 0;



                        time[nn] = nn * (double)finalTime / NBSTEPS;
                        time[NBSTEPS-1] = finalTime;
                        value1[nn] = .9 * rs(i, nn * floor(TRIALTIME / NBSTEPS));  // The network response of the appropriate output neuron at the appropriate time
                        //value1[nn] = .5 * ( .75 + rs(i, nn * floor(TRIALTIME / NBSTEPS)));  // The network response of the appropriate output neuron at the appropriate time
                        if (value1[nn] < 0) value1[nn] = 0;
                        
                            /*if ((i == 5)) // || (i == 3) || (i == 4)) 
                                value1[nn] = 1.0;*/
                            /*else
                                value1[nn] = 0;
                                */
                        //cout << value1[nn] << ", " << nn << ", " << NBSTEPS << ", " << TRIALTIME << ", " << floor(TRIALTIME / NBSTEPS) << ", " << nn * floor(TRIALTIME / NBSTEPS) << endl;
                    }


                    muscleController->prescribeControlForActuator(i, new PiecewiseLinearFunction(NBSTEPS,time,value1));



                    // DEBUGGING

                    /*for (int nn=0; nn < NBSTEPS; nn++)
                      cout << value1[nn] << ", ";
                      cout << endl;
                      for (int nn=0; nn < TRIALTIME; nn++)
                      cout << rs(NBNEUR - 1, nn ) << ", ";
                      cout << endl;
                      return 0;*/

                }

                // Make sure the muscles states are in equilibrium
                osimModel.equilibrateMuscles(s);

                // Integrate from initial time to final time
                manager.setInitialTime(initialTime);
                manager.setFinalTime(finalTime);
                //cout<<"\nIntegrating from "<<initialTime<<" to "<<finalTime<<endl;

                manager.integrate(s);

                osimModel.getMultibodySystem().realize(s, Stage::Acceleration); // Needed to read the position values
                
                /*
                // Obtain the hand-arm center-of-mass position !
                SimTK::Vec3 com, handpos;
                const OpenSim::Body& HandBody =  osimModel.getBodySet().get("r_ulna_radius_hand");
                HandBody.getMassCenter(com);
                osimModel.getSimbodyEngine().getPosition(s, HandBody, com, handpos);
                */
                
                // Obtain the position corresponding to the 'hand' marker (hopefully) !
                SimTK::Vec3 handpos;
                const OpenSim::Body& HandBody =  osimModel.getBodySet().get("r_ulna_radius_hand");
                osimModel.getSimbodyEngine().getPosition(s, HandBody, fingertippos, handpos);
                cout << handpos << endl;

                double elbowflex = osimModel.getCoordinateSet().get("r_elbow_flex").getValue(s) ;
                cout << "Elbow flex: " << elbowflex << endl;


                double *PosTgt;
                if (trialtype == 0)
                    PosTgt = PosTgt1;
                else
                    PosTgt = PosTgt2;

                //err =  3.0 - elbowflex ;   // The 3.0 is probably unnecessary since we are using err - mean(err)...

                /*
                double err = sqrt (  (handpos[0] - PosTgt[0]) * (handpos[0] - PosTgt[0])  +  (handpos[1] - PosTgt[1]) * (handpos[1] - PosTgt[1]) + (handpos[2] - PosTgt[2]) * (handpos[2] - PosTgt[2]) )
                                              + .05 * (double)rs.topRows(NBMUSCLES).cwiseMax(0).sum() / (double)rs.topRows(NBMUSCLES).size(); // Muscle activity penalty !
                                        ;
                */
                
                // Now we recover the data about hand position that was stored during the simulation (through the 'fingerkin' analysis).
                OpenSim::Array<double> colX, colY, colZ, colT;
                Storage *stor = fingerkin.getPositionStorage();
                int nbst = stor->getSize();
                VectorXd vecX(nbst -1), vecY(nbst -1), vecZ(nbst -1), vecT(nbst -1);  
                stor->getTimeColumn(colT);  // I had to dig in the API to find out that this existed...
                stor->getDataColumn("fingertip_X", colX);
                stor->getDataColumn("fingertip_Y", colY);
                stor->getDataColumn("fingertip_Z", colZ);
                for (int nn=0; nn < nbst - 1; nn++){
                    vecX(nn) = colX[nn + 1] ; //- PosTgt[0];
                    vecY(nn) = colY[nn + 1] ; //- PosTgt[1];
                    vecZ(nn) = colZ[nn + 1] ; //- PosTgt[2];
                    vecT(nn) = colT[nn + 1]  - colT[nn];  // The samples are not taken at regular times, so we need the time elapsed at each sample.
                }
                //cout << stor->getColumnLabels() << stor->getSize() << colX.size() << colX[0] << " " << colX[2] << endl;
                //cout << vecX(nbst-2) << " " << vecY(nbst - 2) << " " << vecZ(nbst - 2) << " " << vecT(nbst - 2) <<  endl;
                //cout  << vecT.head(20).transpose() << endl;

                VectorXd vectdists = ((vecX.array() - PosTgt[0]).square() + (vecY.array() - PosTgt[1]).square() + (vecZ.array() - PosTgt[2]).square()).sqrt(); // The distance from target at each sample time...

                double errposMean = vectdists.dot(vecT) / vecT.sum();  // Take the time-averaged distance from target
                double errposMeanLate = vectdists.tail(nbst/2).dot(vecT.tail(nbst/2)) / vecT.tail(nbst/2).sum();  // Same thing, but only for the latter half of the trial
                double errposLastTimeStep =  vectdists(vectdists.size() - 1); //  Only take into account distance at the last timestep.
                
                double errpos = errposLastTimeStep;
                //double errpos = errposMeanLate;

                // Using simple averages may cause difficulties when the sampling rate in different phases of the motion is very uneven !
                // double errpos =  (vecX.array().square() + vecY.array().square() + vecZ.array().square()).sqrt().sum() / (double) nbst; 
                // double errpos =  (vecX.cwiseAbs() + vecY.cwiseAbs() + vecZ.cwiseAbs() ).sum() / (double) nbst; 


                                        
                double activationpenalty = ALPHAACTIVPEN * (rs.topRows(NBMUSCLES).cwiseMax(0).transpose() * MIFs).sum() / (double)rs.cols();  // Should do it !
                //double activationpenalty = .4 * (rs.topRows(NBMUSCLES).cwiseMax(0).transpose() * MIFs).sum() / (double)rs.cols();  // Should do it !
                //double activationpenalty = (rs.topRows(NBMUSCLES).cwiseMax(0).transpose() * VectorXd::Ones(NBMUSCLES)).sum() / (double)rs.topRows(NBMUSCLES).size() ;  // Should be equivalent to the old version
                //double err = sqrt (  (handpos[0] - PosTgt[0]) * (handpos[0] - PosTgt[0])  +  (handpos[1] - PosTgt[1]) * (handpos[1] - PosTgt[1]) + (handpos[2] - PosTgt[2]) * (handpos[2] - PosTgt[2]) ) 
                double err = errpos
                          + activationpenalty;


                /*err = sqrt (  (handpos[0] - PosTgt[0]) * (handpos[0] - PosTgt[0])  +  (handpos[1] - PosTgt[1]) * (handpos[1] - PosTgt[1]) + (handpos[2] - PosTgt[2]) * (handpos[2] - PosTgt[2]) ) 
                        - elbowflex;
                        ;*/
                cout << "Err (position, activation pen., total): " 
                    //<< sqrt (  (handpos[0] - PosTgt[0]) * (handpos[0] - PosTgt[0])  +  (handpos[1] - PosTgt[1]) * (handpos[1] - PosTgt[1]) + (handpos[2] - PosTgt[2]) * (handpos[2] - PosTgt[2]) ) 
                    //<< " + " << .1 * (double)rs.cwiseMax(0).sum() / (double)rs.size() << " = " 
                    << errpos << " " << activationpenalty << " " << err  
                    << endl;

                if (isnan(err))
                {
                    throw std::runtime_error("NaN value in the coordinates!");
                }

                for (int n1=0; n1 < NBNEUR; n1++)
                    for (int n2=0; n2 < NBNEUR; n2++)
                        hebb(n1, n2) = hebbmat[n1][n2];


                if ((PHASE == LEARNING) && (numtrial> 50)
                        // && (numtrial %2 == 1)
                   )
                {
                    dJ = (  -  ETA * meanerrtrace(trialtype) * (hebb.array() * (err - meanerrtrace(trialtype)))).transpose().cwiseMin(MAXDW).cwiseMax(-MAXDW);
                    J +=  dJ;
                }

                // osimModel.updAnalysisSet().get("fingerkin").printResults("test"); // Outputs results of analyses (position, vel, acc of fingertip) to files.
                 
                //ArrayPtrs<Storage>& zz = osimModel.updAnalysisSet().get("fingerkin").getStorageList()   ;
                //cout << "Size of storage list: "<< zz.size() << endl;
                //Storage *zz = osimModel.updAnalysisSet().get("fingerkin").getStorageList()[0]   ;
                //cout << " Name of 1st storage : " << zz->getName() << " of size " << zz->getSize() << endl;

                //cout << osimModel.updAnalysisSet().get("fingerkin").getColumnLabels () << endl;
                //cout << osimModel.updAnalysisSet().get("fingerkin").getStorageList() << endl; // <-- returns empty!
                //cout << osimModel.updAnalysisSet().get("fingerkin")._pStore << endl; // doesn't work

                


                meanerrtrace(trialtype) = ALPHATRACE * meanerrtrace(trialtype) + (1.0 - ALPHATRACE) * err; 
                //meanerrtrace(trialtype) = meanerr; 
                errs(numtrial) = err;


                int ERRSTORINGPERIOD = 100;
                //cout << (numtrial % ERRSTORINGPERIOD) << "!" <<endl;
                if (PHASE == LEARNING)
                {
                    if ((numtrial % ERRSTORINGPERIOD  == 0) && ((numtrial > 0) || (RANDW == 1)))
                    {
                        cout << "Saving..." <<endl;
                        //myfile.open("J_"+std::to_string(numtrial)+".txt", ios::trunc | ios::out);  myfile << J << endl; myfile.close();
                        //saveWeights(J, "J_"+std::to_string(numtrial)+".dat");
                        //myfile.open("J" + SUFFIX + ".txt", ios::trunc | ios::out);  myfile << J << endl; myfile.close();
                        //myfile.open("win" + SUFFIX + ".txt", ios::trunc | ios::out);  myfile << win << endl; myfile.close();
                        //saveWeights(J, "J" + SUFFIX + ".dat");
                        //saveWeights(win, "win" + SUFFIX + ".dat"); // win doesn't change over time.
                        if (RANDW){
                        saveWeights(J, "J_RANDW.dat");
                        saveWeights(win, "win_RANDW.dat"); // win doesn't change over time, shouldn't be affected by RANDW.
                        myfile.open("J_RANDW.txt", ios::trunc | ios::out);  myfile << endl << J << endl; myfile.close();
                        }
                        else {
                        saveWeights(J, "J.dat");
                        saveWeights(win, "win.dat"); // win doesn't change over time.
                        myfile.open("J.txt", ios::trunc | ios::out);  myfile << endl << J << endl; myfile.close();
                        }

                        if (RELOAD) {
                            cout << "Appending " << RELOAD << endl;
                            //myfile.open("errs" + SUFFIX + ".txt", ios::app | ios::out);  myfile << endl << errs.head(numtrial).tail(ERRSTORINGPERIOD) << endl; myfile.close();
                            myfile.open("errs.txt", ios::app | ios::out);  myfile << endl << errs.head(numtrial).tail(ERRSTORINGPERIOD) << endl; myfile.close();
                        }
                        else{
                            cout << "Truncating "  << RELOAD << endl;
                            //myfile.open("errs" + SUFFIX + ".txt", ios::trunc | ios::out);  myfile << endl << errs.head(numtrial) << endl; myfile.close();
                            myfile.open("errs.txt", ios::trunc | ios::out);  myfile << endl << errs.head(numtrial) << endl; myfile.close();
                        }

                    }

                    if (numtrial % (100) <  4*NBTRIALTYPES)
                    {    
                        cout << numtrial << "- trial type: " << trialtype;
                        cout << ", err: " << err;
                        cout << ", r(0,1:6): " << r.transpose().head(6) ; 
                        cout << ", dJ(0,1:4): " << dJ.row(0).head(4)  ;
                        cout << endl;
                    }
                }
                else if (PHASE == TESTING) {
                    //myfile.open("J.txt", ios::trunc | ios::out);  myfile << endl << J << endl; myfile.close();
                    cout << numtrial << "- trial type: " << trialtype;
                    cout << " r[0]: " << r(0);
                    cout << endl;
                    if (RANDW)
                    {
                        myfile.open("rs_long_RANDW_type"+std::to_string(trialtype)+"_"+std::to_string(int(numtrial/NBTRIALTYPES))+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
                        myfile.open("pos_RANDW_type"+std::to_string(trialtype)+"_"+std::to_string(int(numtrial/NBTRIALTYPES))+".txt", ios::trunc | ios::out);  
                    }
                    else
                    {
                        myfile.open("rs_long_type"+std::to_string(trialtype)+"_"+std::to_string(int(numtrial/NBTRIALTYPES))+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
                        myfile.open("pos_type"+std::to_string(trialtype)+"_"+std::to_string(int(numtrial/NBTRIALTYPES))+".txt", ios::trunc | ios::out);  
                    }
                        myfile << endl << vecX.transpose() << endl; 
                        myfile << endl << vecY.transpose() << endl; 
                        myfile << endl << vecZ.transpose() << endl; 
                        myfile.close();
                    //myfile.open("rs_long_type"+std::to_string(trialtype)+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
                }
            }
            catch (const std::exception& ex)
            {
                errs(numtrial) = 1.0;
                std::cout << ex.what() << std::endl;
                //return 1;
            }


        }
    cout << "Done learnin' ..." << endl;
    return 0;
}

void saveWeights(MatrixXd& J, string fname)
{
    double wdata[J.rows() * J.cols()];
    int idx=0;
    for (int cc=0; cc < J.cols(); cc++)
        for (int rr=0; rr < J.rows(); rr++)
            wdata[idx++] = J(rr, cc);
    ofstream myfile(fname, ios::binary | ios::trunc);
    if (!myfile.write((char*) wdata, J.rows() * J.cols() * sizeof(double)))
        throw std::runtime_error("Error while saving matrix of weights.\n");
    myfile.close();
}
void readWeights(MatrixXd& J, string fname)
{
    double wdata[J.cols() * J.rows()];
    int idx=0;
    cout << endl << "Reading weights from file " << fname << endl;
    ifstream myfile(fname, ios::binary);
    if (!myfile.read((char*) wdata, J.cols() * J.rows() * sizeof(double)))
        throw std::runtime_error("Error while reading matrix of weights.\n");
    myfile.close();
    for (int cc=0; cc < J.cols() ; cc++)
        for (int rr=0; rr < J.rows(); rr++)
            J(rr, cc) = wdata[idx++];
    cout << "Done!" <<endl;
}


void randJ(MatrixXd& J)
{
    for (int rr=0; rr < J.rows(); rr++)
        for (int cc=0; cc < J.cols(); cc++)
        {
            if (Uniform(myrng) < PROBACONN)
                J(rr, cc) =  G * Gauss(myrng) / sqrt(PROBACONN * NBNEUR);
            else
                J(rr, cc) = 0.0;
        }
}

