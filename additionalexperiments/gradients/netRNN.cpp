// Task: a recurrent network with 200 neurons receives a random input vector
// for 100 ms, then (after 100ms delay), during the 100ms response period, the
// (arbitrarily chosen) output neuron must output 1 if the input vector had
// positive mean, or -1 i it had negative mean.  
// No learning actually occurs : we just compute the gradients that would be
// applied to a randomly chosen lateral weight of the output neuron, according
// to various methods.



#include <iostream>
#include <stdexcept>
#include <fstream>
#include <string>
#include <random>
#include "../../../Eigen/Eigen/Dense"

#define TESTING 777
#define LEARNING 666


#define NBPATTERNS  4

using namespace std;
using namespace Eigen;
void saveWeights(MatrixXd& m, string fname);
void readWeights(MatrixXd& m, string fname);
void randJ(MatrixXd& m);
void randVec(VectorXd & m);
void randVecGauss(VectorXd & m);
void randMat(MatrixXd& m);


int NBNEUR = 200;
int NBIN = 10;  // Number of inputs. Input 0 is reserved for a 'go' signal that is not used here.
int NBOUT = 1;    // Only 1 output neuron
double PROBACONN = 1.0;  // Dense connectivity
double G = 1.5;   // Early chaotic regime. Chaos ma non troppo.



// Here you choose which learning method to use:

// For faster, but less biologically plausible method (simple node-perturbation
// method, similar to Fiete & Seung 2006), uncomment this:
// It is faster because you only compute the
// Hebbian increment on the few timesteps where a perturbation actually
// occurs.
/*
string METHOD = "NODEPERT"; 
double ETA = .001 ; //  Learning rate
*/

//For slower, but more biologically plausible method (as described in the paper, see http://biorxiv.org/content/early/2016/06/07/057729 ):
string METHOD = "DELTAX"; 
double ETA = .1;  //  Learning rate


// == PARAMETERS ==

string MODULTYPE = "DECOUPLED"; // Modulations (exploratory perturbations) are applied independently to each neuron.
double ALPHAMODUL = 16.0;  // Note that TAU = 30ms, so the real ALPHAMODUL is 16.0 * dt / TAU ~= .5
double MAXDW = 3e-4 ; 
double PROBAMODUL = .003;

int RNGSEED = 0;

int DEBUG = 0;



double ALPHATRACE = .75;
double ALPHATRACEEXC = 0.05;

int SQUARING = 1;


//double INPUTMULT = 5.0;


std::default_random_engine myrng;
std::normal_distribution<double> Gauss(0.0,1.0);
std::uniform_real_distribution<double> Uniform(0.0,1.0);

int main(int argc, char* argv[])
{


    fstream myfile, gradfile;
        gradfile.open("gradsRNN.txt", ios::trunc | ios::out);  



    double dt = 1.0;
    double tau = 30.0;

    int PHASE=LEARNING;
    int RANDW = 0;
    if (argc > 1)
        for (int nn=1; nn < argc; nn++)
        {
            if (strcmp(argv[nn], "test") == 0) { PHASE = TESTING; cout << "Test mode. " << endl; }
            if (strcmp(argv[nn], "RANDW") == 0) { RANDW = 1; } // Randomize weights. Only useful for 'test' mode.
            if (strcmp(argv[nn], "METHOD") == 0) { METHOD = argv[nn+1]; }
            if (strcmp(argv[nn], "MODULTYPE") == 0) { MODULTYPE = argv[nn+1]; }
            if (strcmp(argv[nn], "SQUARING") == 0) { SQUARING = atoi(argv[nn+1]); }
            if (strcmp(argv[nn], "DEBUG") == 0) { DEBUG = atoi(argv[nn+1]); }
            if (strcmp(argv[nn], "G") == 0) { G = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "ETA") == 0) { ETA = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "TAU") == 0) { tau = atof(argv[nn+1]); }
            //if (strcmp(argv[nn], "INPUTMULT") == 0) { INPUTMULT = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "ALPHAMODUL") == 0) { ALPHAMODUL = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "PROBAMODUL") == 0) { PROBAMODUL = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "ALPHATRACE") == 0) { ALPHATRACE = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "ALPHATRACEEXC") == 0) { ALPHATRACEEXC = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "RNGSEED") == 0) { RNGSEED = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "MAXDW") == 0) { MAXDW = atof(argv[nn+1]); }
        }

    string SUFFIX = "_G" + to_string(G) + "_MAXDW" + to_string(MAXDW) + "_ETA" + to_string(ETA) + "_ALPHAMODUL" + to_string(ALPHAMODUL) + "_PROBAMODUL" + to_string(PROBAMODUL) + "_SQUARING" +to_string(SQUARING) + "_MODULTYPE-" + MODULTYPE +   
        "_ALPHATRACE" + to_string(ALPHATRACE) + "_METHOD-" + METHOD +  "_ATRACEEXC" + to_string(ALPHATRACEEXC) + "_TAU" + to_string(tau) + 
        //"_INPUTMULT" +to_string(INPUTMULT) + 
        "_RNGSEED" + 
        to_string(RNGSEED);
    cout << SUFFIX << endl;

    myrng.seed(RNGSEED);
    srand(RNGSEED);


    int trialtype;


    int NBTRIALS = 507;
    int TRIALTIME = 300;
    int TIMEINPUT = 100;
    int TIMEMODUL = 210;
    int TIMERESP = 200;

    VectorXi modulmarker(NBNEUR); modulmarker.setZero();

    if (PHASE == TESTING) 
    {
        NBTRIALS = 20*NBPATTERNS;
        //NBTRIALS = 40*NBPATTERNS;
        //TRIALTIME = 1500;
    }



    MatrixXd dJ(NBNEUR, NBNEUR); dJ.setZero();
    MatrixXd win(NBNEUR, NBIN); win.setRandom(); //win.row(0).setZero(); // Input weights are uniformly chosen between -1 and 1, except possibly for output cell (not even necessary). No plasticity for input weights.

    
    win *= .2;



    MatrixXd J(NBNEUR, NBNEUR);


    cout << Uniform(myrng) << endl;
    randJ(J); // Randomize recurrent weight matrix, according to the Sompolinsky method (Gaussian(0,1), divided by sqrt(ProbaConn*N) and multiplied by G - see definition of randJ() below).

    // If in the TESTING mode, read the weights from a previously saved file:
    if (PHASE == TESTING){
        if (RANDW == 0){
        //readWeights(J, "J.dat");
        //readWeights(win, "win.dat");
        readWeights(J, "J" + SUFFIX + ".dat");
        readWeights(win, "win" + SUFFIX + ".dat"); // win doesn't change over time.
        }
        else cout << "Using random weights." << endl;
    }


//cout << J(0,0) << " " << win(1,1) << endl;

    VectorXd meanabserrs(NBTRIALS); meanabserrs.setZero();
    VectorXd lateral_input;
    VectorXd total_exc(NBNEUR), delta_total_exc(NBNEUR), delta_total_exc_sq(NBNEUR), total_exc_prev(NBNEUR);
    VectorXd delta_r(NBNEUR), delta_r_sq(NBNEUR), r_trace(NBNEUR), r_trace2(NBNEUR);
    r_trace.fill(0); r_trace2.fill(0);
    VectorXd delta_x(NBNEUR), delta_x_sq(NBNEUR), x_trace(NBNEUR), x_trace2(NBNEUR), delta_x_cu(NBNEUR);
    total_exc.fill(0); total_exc_prev.fill(0); x_trace.fill(0); x_trace2.fill(0);
    VectorXd modul(NBNEUR); modul.setZero();
    VectorXd modul_trace(NBNEUR); modul_trace.setZero();
    MatrixXd rs(NBNEUR, TRIALTIME); rs.setZero();
    MatrixXd hebb(NBNEUR, NBNEUR);  
    VectorXd x(NBNEUR), r(NBNEUR), rprev(NBNEUR), dxdt(NBNEUR), k(NBNEUR), 
             input(NBIN), deltax(NBNEUR);
    VectorXd         etraceDELTAX(NBNEUR), etraceDELTAXCUALT(NBNEUR), etraceDELTAXCU(NBNEUR), etraceNODEPERT(NBNEUR), etraceDELTAXOP(NBNEUR), etraceDELTAX31(NBNEUR), etraceDELTAXSIGNSQRT(NBNEUR), etraceDELTAXSIGNSQ(NBNEUR), etraceEH(NBNEUR);
    VectorXd         gradDELTAX(NBNEUR), gradDELTAXCUALT(NBNEUR), gradDELTAXCU(NBNEUR), gradNODEPERT(NBNEUR), gradDELTAXOP(NBNEUR), gradBP(NBNEUR), gradDELTAX31(NBNEUR), gradDELTAXSIGNSQRT(NBNEUR), gradDELTAXSIGNSQ(NBNEUR), gradEH(NBNEUR);
    x.fill(0); r.fill(0);


    double modul0;
    double rew, rew_trace, delta_rew;
    double tgtresp;
    double predictederr;

    VectorXd err(TRIALTIME); 
    VectorXd meanabserrtrace(NBPATTERNS);
    double meanabserr;

    MatrixXd dJtmp, Jprev, Jr;


    // Auxiliary variables for speeding things up a little bit.
    double hebbmat[NBNEUR][NBNEUR];
    double rprevmat[NBNEUR], dx2[NBNEUR];
    double xmat[NBNEUR], xtracemat[NBNEUR];

    double dtdivtau = dt / tau;



    meanabserrtrace.setZero();




    for (int numtrial=0; numtrial < NBTRIALS; numtrial++)
    {



        // We use native-C array hebbmat for fast computations within the loop, then transfer it back to Eigen matrix hebb for plasticity computations
        hebb.setZero();
        for (int n1=0; n1 < NBNEUR; n1++)
            for (int n2=0; n2 < NBNEUR; n2++)
                hebbmat[n1][n2] = 0;
        r.setZero();
        x.setZero();

        modul0 = 0;

        if (numtrial %2 == 0)
            input.setRandom();
        
        tgtresp  = input.sum() > 0 ? 1.0 : -1.0;

        etraceDELTAX.setZero();
        etraceEH.setZero();
        etraceDELTAXCU.setZero();
        etraceDELTAXSIGNSQ.setZero();
        etraceDELTAXSIGNSQRT.setZero();
        etraceDELTAXCUALT.setZero();
        etraceDELTAXOP.setZero();
        etraceDELTAX31.setZero();
        etraceNODEPERT.setZero();

        rew = 0;
        rew_trace = 0;

        // Let's start the trial:
        for (int numiter=0; numiter < TRIALTIME;  numiter++)
        {

            rprev = r;
            lateral_input =  J * r;

            if (numiter < TIMEINPUT)
                total_exc =  lateral_input + win * input ;
            else
                total_exc =  lateral_input ;




            // Exploratory perturbations
/*            modul.setZero();
            if (MODULTYPE == "UNIFORM")
            {
                // Apply a modulation to the entire network with probability PROBAMODUL - Not used for these simulations.
                if ( (Uniform(myrng) < PROBAMODUL)
                        && (numiter> 3)
                   )
                {
                    randVec(modul);
                    modul *= ALPHAMODUL;
                    total_exc +=   modul;
                    modulmarker.fill(1);
                }
                else
                    modulmarker.setZero();
            }
            else if (MODULTYPE == "DECOUPLED")
            {
                // Perturb each neuron independently with probability PROBAMODUL
                modulmarker.setZero();
                for (int nn=0; nn < NBNEUR; nn++)
                    if ( (Uniform(myrng) < PROBAMODUL)
                            && (numiter> 3)
                       )
                    {
                        modulmarker(nn) = 1;
                        modul(nn) = (-1.0 + 2.0 * Uniform(myrng));
                    }
                modul *= ALPHAMODUL;
                total_exc +=   modul;
            }
            else { throw std::runtime_error("Which modulation type?"); }
*/
            
            if (numtrial % 2 == 1)
            {
                if (numiter == TIMEMODUL)
                {
                    modul0 =  Uniform(myrng) < .5 ? ALPHAMODUL : -ALPHAMODUL;  // Fixed-magnitude modulations: sem to align much better to BackProp gradients
                    //modul0 =  (1.0 - 2.0 * Uniform(myrng))  * ALPHAMODUL ;  
                    
                    total_exc(0) += modul0;

                    etraceNODEPERT = r * modul0;
                }
            }

            
            
            // Compute network activations

            x += dtdivtau * (-x + total_exc);

            x(1)=1.0; x(10)=1.0;x(11)=-1.0; //x(12) = 1.0;  // Biases


            // Actual responses = tanh(activations)
            for (int nn=0; nn < NBNEUR; nn++)
            {
                r(nn) = tanh(x(nn));
            }


            rs.col(numiter) = r;


            // Okay, now for the actual plasticity.

            // First, compute the fluctuations of neural activity (detrending / high-pass filtering)
            // NOTE: Remember that x is the current excitation of the neuron (i.e. what is passed to tanh to produce the actual output r) - NOT the input!
            delta_x =  x  - x_trace ;
            //delta_x_sq = delta_x.array() * delta_x.array().abs();
            //delta_x_cu = delta_x.array() * delta_x.array() * delta_x.array();
            x_trace = ALPHATRACEEXC * x_trace + (1.0 - ALPHATRACEEXC) * x;

            // This is for the full exploratory-hebbian rule, which requires a continuous, real-time reward signal (and its running average, with same time constant as that of x)
            rew = -abs(r(0) - tgtresp);
            delta_rew =  rew  - rew_trace ;
            //delta_x_sq = delta_x.array() * delta_x.array().abs();
            //delta_x_cu = delta_x.array() * delta_x.array() * delta_x.array();
            rew_trace = ALPHATRACEEXC * rew_trace + (1.0 - ALPHATRACEEXC) * rew;


            if (numiter > TIMERESP)
            {

                    // Note that r = r(t-1) = the current lateral input to each neuron! 

                    // Eligibility trace as the accumulated product of inputs times fluctuations in output (i.e. like the Exploratory Hebbian rule, without the continuous real-time reward signal) (should not work)
                    etraceDELTAX += rprev * delta_x(0);
             
                    // For Exploratory-Hebbian rule, the eligibility trace is essentially the gradient:
                    etraceEH += rprev * delta_x(0) * delta_rew;
                    
                    // Eligibility trace computed according to our rule (accumulated product of inputs time fluctuations, passed through a supralinear function - here, the cubic function)
                    etraceDELTAXCU.array() +=  (rprev * delta_x(0)).array().cube();
                    

                    // Slight variant: cube applied only to the fluctuations, not to the total product 
                    double deltaxcu =  delta_x(0) * delta_x(0) * delta_x(0);
                    etraceDELTAXCUALT += deltaxcu * rprev;

                    // Signed-square nonlinearity (supralinear, so should work)
                    etraceDELTAXSIGNSQ.array() +=  (rprev * delta_x(0)).array() * (rprev * delta_x(0)).array().abs();
                    
                    // Signed-square-root nonlinearity (sublinear, so should NOT work)
                    etraceDELTAXSIGNSQRT.array() +=  ((r * delta_x(0)).array() > 0).select( (r * delta_x(0)).array().abs().sqrt(), -(r * delta_x(0)).array().abs().sqrt() ) ;

                    
                    // Product of inputs times fluctuation, but only at the time of the modulation (should work, illustating that the problem is with the post-perturbation relaxation effects)
                    if (numiter == TIMEMODUL)
                    {
                        etraceDELTAXOP += rprev * delta_x(0);
                    }
                    
                    // Same thing, but accumulated for a few ms after the modulation
                    if ((numiter >= TIMEMODUL) && (numiter < TIMEMODUL + 11))
                    //if (numiter >= TIMEMODUL)
                        etraceDELTAX31 += rprev * delta_x(0);
                    

            }


        }  // Trial finished!


        // Compute error for this trial

        int EVALTIME = TRIALTIME - TIMERESP; 

        err = rs.row(0).array() - tgtresp;
        err.head(TIMERESP).setZero(); // Error is only computed over the response period, i.e. from TIMERESP onward

        meanabserr =  err.cwiseAbs().sum() / (double)EVALTIME;

        if (numtrial % 2 == 0)
            predictederr = meanabserr;
        
        // How to compute predicted error in the absence of perturbation (i.e. R0)? Normally we simply use a running average over previous Rs for this particular trial type, but here there is no trial type !
        // Solution is to run each trial twice, once with and once without the perturbation.

        if (numtrial % 2 == 1)
        {
            gradDELTAX = -ETA * (meanabserr -   predictederr) * etraceDELTAX;
            gradEH = ETA * etraceEH ;  // The reward/error signal is already included in the e-trace for the EH rule
            gradDELTAXOP = -ETA * (meanabserr - predictederr) * etraceDELTAXOP;
            gradDELTAX31 = -ETA * (meanabserr - predictederr) * etraceDELTAX31;
            gradDELTAXCU = -ETA * (meanabserr - predictederr) * etraceDELTAXCU;
            gradDELTAXCUALT = -ETA * (meanabserr - predictederr) * etraceDELTAXCUALT;
            gradDELTAXSIGNSQ = -ETA * (meanabserr - predictederr) * etraceDELTAXSIGNSQ;
            gradDELTAXSIGNSQRT = -ETA * (meanabserr - predictederr) * etraceDELTAXSIGNSQRT;
            gradNODEPERT = -ETA * (meanabserr - predictederr) * etraceNODEPERT;
            // For backpropagation, the gradient is simply error * inputs. We
            // only consider the gradient around the time of the modulation to
            // ensure a fair comparison with other measures. We use TIMEMODUL-1
            // as the closest approximation not affected by the modulation
            // itself.
            gradBP = -ETA * err(TIMEMODUL -1)  * rs.col(TIMEMODUL-1);
        }

        // We re-transfer the values back from the C arrays to the Eigen matrix
        /*
        for (int n1=0; n1 < NBNEUR; n1++)
            for (int n2=0; n2 < NBNEUR; n2++)
                hebb(n1, n2) = hebbmat[n1][n2];


        // Compute the actual weight change, based on eligibility trace and the relative error for this trial:

        if ((PHASE == LEARNING) && (numtrial> 100)
           )
        {
            // Note that the weight change is the summed Hebbian increments, multiplied by the relative error, AND the mean of recent errors for this trial type - this last multiplication may help to stabilize learning.
            dJ = (  -  ETA * meanabserrtrace(trialtype) * (hebb.array() * (meanabserr - meanabserrtrace(trialtype)))).transpose().cwiseMin(MAXDW).cwiseMax(-MAXDW);
            J +=  dJ;

        }


        meanabserrtrace(trialtype) = ALPHATRACE * meanabserrtrace(trialtype) + (1.0 - ALPHATRACE) * meanabserr; 
        meanabserrs(numtrial) = meanabserr;
        */


        // Display stuff, save files.
        VectorXd etrace = etraceDELTAX;

        if (numtrial % 2 == 1)
        {
            // Note that the weight change is the summed Hebbian increments, multiplied by the relative error, AND the mean of recent errors for this trial type - this last multiplication may help to stabilize learning.
            //dwff = (  -  ETA * meanabserrtrace(trialtype) * (etrace.array() * (meanabserr - meanabserrtrace(trialtype)))).cwiseMin(MAXDW).cwiseMax(-MAXDW);
            //wff +=  dwff;

            int numsyn = (int)floor(Uniform(myrng) * NBNEUR);
            gradfile << gradBP(numsyn) << " " << gradDELTAX(numsyn) << " " << gradDELTAXOP(numsyn) << " " << gradDELTAX31(numsyn) << " " << gradDELTAXCU(numsyn) << " " <<gradDELTAXCUALT(numsyn) << " " << gradDELTAXSIGNSQ(numsyn) << 
                " " << gradDELTAXSIGNSQRT(numsyn) << " " << gradEH(numsyn)  << " " <<gradNODEPERT(numsyn) <<endl;
        }


            if (numtrial % 100 <  8)
            {    
                cout << numtrial ; // << "- trial type: " << trialtype;
                //cout << ", responses : " << zout;
                //cout << ", time-avg responses for each pattern: " << zouttrace ;
                //cout << ", sub(abs(wout)): "  << wout.cwiseAbs().sum() ;
                //cout << ", hebb(0,1:3): " << hebb.col(0).head(4).transpose();
                cout << ", meanabserr: " << meanabserr;
                //cout << ", wout(0,1:3): " << wout.row(0).head(5) ; 
                cout << ", r: " << r.head(5).transpose();
                cout << ", modul: " << modul0;
                cout << ", input: " << input.head(4).transpose();
                //cout << ", wff: " << wff.transpose();
                //cout << ", dwff: " << dwff.transpose();
                cout<<endl;
/*                cout << ", gradCU: " << gradDELTAXCU.transpose();
                cout<<endl;
                cout << ", gradCUalt: " << gradDELTAXCUALT.transpose();
                cout<<endl;
                cout << ", gradDX: " << gradDELTAX.transpose();
                cout<<endl;
                cout << ", gradOP: " << gradDELTAXOP.transpose();
                cout<<endl;
                cout << ", gradNP: " << gradNODEPERT.transpose();
                cout << endl;
                cout << ", gradBP: " << gradBP.transpose();
                cout << endl;
        cout << gradNODEPERT.dot(gradBP) / (gradNODEPERT.norm() * gradBP.norm());
        cout << " " << gradDELTAX.dot(gradBP) / (gradDELTAX.norm() * gradBP.norm());
        cout << " " << gradDELTAXOP.dot(gradBP) / (gradDELTAXOP.norm() * gradBP.norm());
        cout << " " << gradDELTAXCU.dot(gradBP) / (gradDELTAXCU.norm() * gradBP.norm());*/
        cout << endl;
            }


    }

    cout << "Done learning ..." << endl;


    cout << J.mean() << " " << J.cwiseAbs().sum() << " " << J.maxCoeff() << endl;
    //cout << wout.mean() << " " << wout.cwiseAbs().sum() << " " << wout.maxCoeff() << endl;


    cout << endl;
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

void randVec(VectorXd& M)
{
    for (int nn = 0; nn < M.size(); nn++)
        M.data()[nn] = -1.0 + 2.0 * Uniform(myrng);
}
void randVecGaussian(VectorXd& M)
{
    for (int nn = 0; nn < M.size(); nn++)
        M.data()[nn] = -1.0 + 2.0 * Gauss(myrng);
}
void randMat(MatrixXd& M)
{
    for (int nn = 0; nn < M.size(); nn++)
        M.data()[nn] = -1.0 + 2.0 * Uniform(myrng);
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

