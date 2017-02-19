// Task: Simple logistic regression of a single neuron with random inputs, must determine whether the mean of inputs is positive or not.
// Strictly feedforward. The paper uses netFF for the gradients figure, but this is used to generate the data for makefigrelax.py (the relaxation figure)
// No learning actually occurs, we just want to see the gradients.


#include <iostream>
#include <stdexcept>
#include <fstream>
#include <string>
#include <random>
#include "Eigen/Dense"

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


int NBNEUR = 1;  
int NBIN = 10;

// The following are ignored for this code:
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
double ALPHAMODUL = 10.0; 
double MAXDW = 3e-4 ; 

int RNGSEED = 0;

int DEBUG = 0;



double ALPHATRACE = .75;

double ALPHATRACEEXC = 0.05;
//double ALPHATRACEEXC = 0.9;

int SQUARING = 1;


//double INPUTMULT = 5.0;


std::default_random_engine myrng;
std::normal_distribution<double> Gauss(0.0,1.0);
std::uniform_real_distribution<double> Uniform(0.0,1.0);

int main(int argc, char* argv[])
{

    fstream myfile, gradfile;
        gradfile.open("grads.txt", ios::trunc | ios::out);  


    double dt = 1.0;
    double tau = 10.0;

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
            /*if (strcmp(argv[nn], "PROBAMODUL") == 0) { PROBAMODUL = atof(argv[nn+1]); }*/
            if (strcmp(argv[nn], "ALPHATRACE") == 0) { ALPHATRACE = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "ALPHATRACEEXC") == 0) { ALPHATRACEEXC = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "RNGSEED") == 0) { RNGSEED = atof(argv[nn+1]); }
            if (strcmp(argv[nn], "MAXDW") == 0) { MAXDW = atof(argv[nn+1]); }
        }

    string SUFFIX = "_G" + to_string(G) + "_MAXDW" + to_string(MAXDW) + "_ETA" + to_string(ETA) + "_ALPHAMODUL" + to_string(ALPHAMODUL) +
        //"_PROBAMODUL" + to_string(PROBAMODUL) + 
        "_SQUARING" +to_string(SQUARING) + "_MODULTYPE-" + MODULTYPE +   
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
    int TIMEMODUL = 200;

    VectorXi modulmarker(NBNEUR); modulmarker.setZero();


    VectorXd wff(NBIN);
    wff.setRandom(); wff *= .2;
    VectorXd dwff(NBIN);


    //randJ(J); // Randomize recurrent weight matrix, according to the Sompolinsky method (Gaussian(0,1), divided by sqrt(ProbaConn*N) and multiplied by G - see definition of randJ() below).


//cout << J(0,0) << " " << win(1,1) << endl;

    VectorXd meanerrs(NBTRIALS); meanerrs.setZero();
    VectorXd lateral_input;
    double ff_input;
    double delta_x, delta_x_sq, x_trace, x_trace_long, x_trace_long_nonoise, delta_x_cu;
    double total_exc = 0;
    VectorXd modul(NBNEUR); modul.setZero();
    double modul0;
    VectorXd modul_trace(NBNEUR); modul_trace.setZero();
//MatrixXd rs(NBNEUR, TRIALTIME); rs.setZero();
    VectorXd rs(TRIALTIME), xs(TRIALTIME), xtraces(TRIALTIME), xtraceslong(TRIALTIME), xtraceslongnonoise(TRIALTIME), xsnonoise(TRIALTIME); 
    MatrixXd hebb(NBNEUR, NBNEUR);  
    //VectorXd x(NBNEUR), r(NBNEUR), rprev(NBNEUR), dxdt(NBNEUR), k(NBNEUR), 

    double x, r, rprev, dxdt;
    VectorXd         input(NBIN), etraceDELTAX(NBIN), etraceDELTAXCUALT(NBIN), etraceDELTAXCU(NBIN), etraceNODEPERT(NBIN), etraceDELTAXOP(NBIN), etraceDELTAX31(NBIN);
    VectorXd         gradDELTAX(NBIN), gradDELTAXCUALT(NBIN), gradDELTAXCU(NBIN), gradNODEPERT(NBIN), gradDELTAXOP(NBIN), gradBP(NBIN), gradDELTAX31(NBIN);

    x=0; r = 0;

    VectorXd err(TRIALTIME); 
    VectorXd meanerrtrace(NBPATTERNS);
    double meanerr;

    MatrixXd dJtmp, Jprev, Jr;


    // Auxiliary variables for speeding things up a little bit.
    double hebbmat[NBNEUR][NBNEUR];
    double rprevmat[NBNEUR], dx2[NBNEUR];
    double xmat[NBNEUR], xtracemat[NBNEUR];

    double dtdivtau = dt / tau;



    meanerrtrace.setZero();



    for (int numtrial=0; numtrial < NBTRIALS; numtrial++)
    {

        //trialtype = numtrial % NBPATTERNS;
        trialtype = 0;   // Only 1 trial type.




        input.setRandom();
        r = 0;
        x=0;
        double x_nonoise=0;
        double r_nonoise=0;
        x_trace = 0;
        x_trace_long = 0;
        x_trace_long_nonoise = 0;
        etraceDELTAX.setZero();
        etraceDELTAXCU.setZero();
        etraceDELTAXCUALT.setZero();
        etraceDELTAXOP.setZero();
        etraceDELTAX31.setZero();
        etraceNODEPERT.setZero();

        for (int numiter=0; numiter < TRIALTIME;  numiter++)
        {


            
            // Note that in this version, only neuron 0 has any nonzero lateral input weights!
            
            rprev = r;

            //ff_input = wff.dot(input) ;
            ff_input = wff.dot(input + VectorXd::Random(NBIN) * .1) ;

            total_exc =  ff_input;



            modul0 = 0;
            if (numiter == TIMEMODUL)
            {
                //modul0 = (-1.0 + 2.0 * Uniform(myrng)) * ALPHAMODUL;
                modul0 =  Uniform(myrng) < .5 ? ALPHAMODUL : -ALPHAMODUL;
                total_exc += modul0;
            }

            
            // Compute neuron 0 activation

            x += dtdivtau * (-x + total_exc);
            x_nonoise += dtdivtau * (-x_nonoise +  wff.dot(input) + modul0);


            r = tanh(x);


            rs(numiter) = r;
            xs(numiter) = x;
            xsnonoise(numiter) = x_nonoise;
            xtraces(numiter) = x_trace; // Note that this is stored before the updating of x_trace, as appropriate since this is value used in computation of delta_x
            xtraceslong(numiter) = x_trace_long;
            xtraceslongnonoise(numiter) = x_trace_long_nonoise;


            // Okay, now for the actual plasticity.

            // First, compute the fluctuations of neural activity (detrending / high-pass filtering)
            delta_x =  x  - x_trace ;
            //delta_x_sq = delta_x.array() * delta_x.array().abs();
            //delta_x_cu = delta_x.array() * delta_x.array() * delta_x.array();
            x_trace = ALPHATRACEEXC * x_trace + (1.0 - ALPHATRACEEXC) * x;
            x_trace_long = .9 * x_trace_long + .1 * x;
            x_trace_long_nonoise = .9 * x_trace_long_nonoise + .1 * x_nonoise;

            // Compute the Hebbian increment to be added to the eligibility trace (i.e. potential weight change) for this time step, based on inputs and fluctuations of neural activity
                    // Method from the paper. Slow, but biologically plausible (-ish).
                    //
                    // The Hebbian increment at every timestep is the inputs (i.e. rprev) times the (cubed) fluctuations in activity for each neuron. 
                    // More plausible, but slower and requires a supralinear function to be applied to the fluctuations (here cubing, but sign-preserving square also works)

            if (numiter > 149)
            {

                    etraceDELTAX += input * delta_x;
                    
                    etraceDELTAXCU.array() +=  (input * delta_x).array().cube();

                    double deltaxcu =  delta_x * delta_x * delta_x;
                    etraceDELTAXCUALT += deltaxcu * input;

                    if (numiter == TIMEMODUL)
                        etraceDELTAXOP += input * delta_x;
                    
                    if ((numiter >= TIMEMODUL) && (numiter < TIMEMODUL + 31))
                    //if (numiter >= TIMEMODUL)
                        etraceDELTAX31 += input * delta_x;



                    /*double incr;
                   for (int n1=0; n1 < NBNEUR; n1++)
                        for (int n2=0; n2 < NBNEUR; n2++)
                        {
                            incr = rprev(n1) * delta_x(n2);
                            hebbmat[n1][n2] +=  incr * incr * incr;
                        }*/

                    // Node-perturbation. 
                    //
                    // The Hebbian increment is the inputs times the
                    // perturbation itself. Node-perturbation method, similar to 
                    // Fiete & Seung 2006. Much faster 
                    // because you only compute the Hebbian
                    // increments in the few timesteps at which a
                    // perturbation actually occurs.

                    /*for (int n2=0; n2 < NBNEUR; n2++)
                        if (modulmarker(n2) != 0)
                            for (int n1=0; n1 < NBNEUR; n1++)
                                hebbmat[n1][n2] += rprev(n1) * modul(n2); */

                    etraceNODEPERT += input * modul0;
            }



        }  // Trial finished!


        // Compute error for this trial

        int EVALTIME = 120; 

        // The expected response is 1 if the input for this trial had positive sum, or -1 otherwise.

        double tgtresp  = input.sum() > 0 ? 1.0 : -1.0;
        err = rs.array() - tgtresp;
        err.head(TRIALTIME - EVALTIME).setZero(); // Error is only computed over the response period, i.e. the last EVALTIME ms.

        meanerr =  err.cwiseAbs().sum() / (double)EVALTIME;
        double meanerrraw = rs(rs.size()-1) - tgtresp;

        // Normally, the predicted error is the mean of previous erros for this trial type. This won't work here because there are no trial types and the inputs are too variable. 
        // Fortunately, we have a predictor of what the error would be in the absence of perturbation - the abs-value actual error on the last step of the trial, when perturbations have washed out.
        double predictederr = abs(meanerrraw);

        gradBP = -meanerrraw * input; // * (1 - (rs(rs.size-1)) * (rs(rs.size-1)) 
        gradDELTAX = -ETA * (meanerr -   predictederr) * etraceDELTAX;
        gradDELTAXOP = -ETA * (meanerr - predictederr) * etraceDELTAXOP;
        gradDELTAX31 = -ETA * (meanerr - predictederr) * etraceDELTAX31;
        gradDELTAXCU = -ETA * (meanerr - predictederr) * etraceDELTAXCU;
        gradDELTAXCUALT = -ETA * (meanerr - predictederr) * etraceDELTAXCUALT;
        gradNODEPERT = -ETA * (meanerr - predictederr) * etraceNODEPERT;

        // Compute the actual weight change, based on eligibility trace and the relative error for this trial:

//        VectorXd etrace = etraceNODEPERT;
        VectorXd etrace = etraceDELTAX;

        if (numtrial > 100)
        {
            // Note that the weight change is the summed Hebbian increments, multiplied by the relative error, AND the mean of recent errors for this trial type - this last multiplication may help to stabilize learning.
            //dwff = (  -  ETA * meanerrtrace(trialtype) * (etrace.array() * (meanerr - meanerrtrace(trialtype)))).cwiseMin(MAXDW).cwiseMax(-MAXDW);
            //wff +=  dwff;

            int numsyn = (int)floor(Uniform(myrng) * NBIN);
            gradfile << gradBP(numsyn) << " " << gradDELTAX(numsyn) << " " << gradDELTAXOP(numsyn) << " " << gradDELTAX31(numsyn) << " " << gradDELTAXCU(numsyn) << " " <<gradDELTAXCUALT(numsyn) << " " <<gradNODEPERT(numsyn) <<endl;
        }

        /*
        cout << etraceDELTAX.dot(etraceNODEPERT) / (etraceDELTAX.norm() * etraceNODEPERT.norm());
        cout << " " << etraceDELTAXCU.dot(etraceNODEPERT) / (etraceDELTAXCU.norm() * etraceNODEPERT.norm());
        cout << endl;
        */

        meanerrtrace(trialtype) = ALPHATRACE * meanerrtrace(trialtype) + (1.0 - ALPHATRACE) * meanerr; 
        meanerrs(numtrial) = meanerr;


        // Display stuff, save files.

        if (PHASE == LEARNING)
        {
            if (numtrial % 3000 < 8) 
            {
                //myfile.open("rs"+std::to_string((numtrial/2)%4)+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
                //myfile.open("rs"+std::to_string(trialtype)+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();

                myfile.open("rs"+std::to_string(numtrial % 3000)+".txt", ios::trunc | ios::out);  myfile << endl << rs << endl; myfile.close();
                myfile.open("xs"+std::to_string(numtrial % 3000)+".txt", ios::trunc | ios::out);  myfile << endl << xs << endl; myfile.close();
                myfile.open("xsnonoise"+std::to_string(numtrial % 3000)+".txt", ios::trunc | ios::out);  myfile << endl << xsnonoise << endl; myfile.close();
                myfile.open("xtraces"+std::to_string(numtrial % 3000)+".txt", ios::trunc | ios::out);  myfile << endl << xtraces << endl; myfile.close();
                myfile.open("xtraceslong"+std::to_string(numtrial % 3000)+".txt", ios::trunc | ios::out);  myfile << endl << xtraceslong << endl; myfile.close();
                myfile.open("xtraceslongnonoise"+std::to_string(numtrial % 3000)+".txt", ios::trunc | ios::out);  myfile << endl << xtraceslongnonoise << endl; myfile.close();
            }
            if ((numtrial % 1000 == 0) || (numtrial == NBTRIALS -1))
            {
                myfile.open("errs" + SUFFIX + ".txt", ios::trunc | ios::out);  myfile << endl << meanerrs.head(numtrial) << endl; myfile.close();

            }


            //if (numtrial % (NBPATTERNS * 100) <  2*NBPATTERNS)
            if (numtrial % 100 <  2*NBPATTERNS)
            {    
                cout << numtrial << "- trial type: " << trialtype;
                //cout << ", responses : " << zout;
                //cout << ", time-avg responses for each pattern: " << zouttrace ;
                //cout << ", sub(abs(wout)): "  << wout.cwiseAbs().sum() ;
                //cout << ", hebb(0,1:3): " << hebb.col(0).head(4).transpose();
                cout << ", meanerr: " << meanerr;
                //cout << ", wout(0,1:3): " << wout.row(0).head(5) ; 
                cout << ", r: " << r;
                //cout << ", wff: " << wff.transpose();
                //cout << ", dwff: " << dwff.transpose();
                cout<<endl;
                cout << ", gradCU: " << gradDELTAXCU.transpose();
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
        cout << " " << gradDELTAXCU.dot(gradBP) / (gradDELTAXCU.norm() * gradBP.norm());
                /*
                cout << ", etraceCU: " << etraceDELTAXCU.transpose();
                cout<<endl;
                cout << ", etraceDX: " << etraceDELTAX.transpose();
                cout<<endl;
                cout << ", etraceOP: " << etraceDELTAXOP.transpose();
                cout<<endl;
                cout << ", etraceNP: " << etraceNODEPERT.transpose();
                cout << endl;
        cout << etraceDELTAX.dot(etraceNODEPERT) / (etraceDELTAX.norm() * etraceNODEPERT.norm());
        cout << " " << etraceDELTAXOP.dot(etraceNODEPERT) / (etraceDELTAXOP.norm() * etraceNODEPERT.norm());
        cout << " " << etraceDELTAXCU.dot(etraceNODEPERT) / (etraceDELTAXCU.norm() * etraceNODEPERT.norm());
        */
        cout << endl;
            }
        }


    }

    gradfile.close();
    cout << "Done learning ..." << endl;




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

