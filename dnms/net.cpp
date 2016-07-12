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


int NBNEUR = 200;
int NBIN = 3;  // Input 0 is reserved for a 'go' signal that is not used here.
int NBOUT = 1;
double PROBACONN = 1.0;  // Dense connectivity
double G = 1.5;   // Early chaotic regime. Chaos ma non troppo.



// For *fast*, but less biologically plausible method (simple node-perturbation
// method, similar to Fiete & Seung 2006).
// It is faster because you only compute the
// Hebbian increment on the few timesteps where a perturbation actually
// occurs.
/*
string METHOD = "NODEPERT"; 
double ETA = .001 ; //  Learning rate
*/

//For slower, but more biologically plausible method (based on detrended
//post-synaptic activities and nonlinearized Hebbian increments to 'extract'
//the exploratory perturbations - see http://biorxiv.org/content/early/2016/06/07/057729 :
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

    fstream myfile;

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


    int NBTRIALS = 20407; // ~10K trials sufficient to get good convergence (95% correct on a binary criterion is reached within ~1000 trials, but performance keeps improving after that). Should really be 100K if you have time.
    int TRIALTIME = 1000;
    int STARTSTIM1 = 1, TIMESTIM1 = 200; 
    int STARTSTIM2 = 400, TIMESTIM2 = 200; 
    /*int TRIALTIME = 1000;
      int STARTSTIM1 = 1, TIMESTIM1 = 200; // 200
      int STARTSTIM2 = 400, TIMESTIM2 = 200; */

    VectorXi modulmarker(NBNEUR); modulmarker.setZero();

    if (PHASE == TESTING) 
    {
        NBTRIALS = 20*NBPATTERNS;
        //NBTRIALS = 40*NBPATTERNS;
        //TRIALTIME = 1500;
    }


    MatrixXd patterns[NBPATTERNS];
    MatrixXd tgtresps[NBPATTERNS];


    // Remember that input channel 0 is reserved for the (unused) 'go' signal


    // For the sequential-XOR problem (NBPATTERNS to 4, TRIALTIME and eval. time as appropriate): 
    // We encode the input patterns as matrices with NBIN rows and TRIALTIME columns, which we fill with the appropriate input values at every time step and for each input channel
    

    patterns[0] = MatrixXd::Zero(NBIN, TRIALTIME); patterns[0].row(1).segment(STARTSTIM1, TIMESTIM1).fill(1.0); patterns[0].row(1).segment(STARTSTIM2, TIMESTIM2).fill(1.0);
    patterns[1] = MatrixXd::Zero(NBIN, TRIALTIME); patterns[1].row(1).segment(STARTSTIM1, TIMESTIM1).fill(1.0); patterns[1].row(2).segment(STARTSTIM2, TIMESTIM2).fill(1.0);
    patterns[2] = MatrixXd::Zero(NBIN, TRIALTIME); patterns[2].row(2).segment(STARTSTIM1, TIMESTIM1).fill(1.0); patterns[2].row(1).segment(STARTSTIM2, TIMESTIM2).fill(1.0);
    patterns[3] = MatrixXd::Zero(NBIN, TRIALTIME); patterns[3].row(2).segment(STARTSTIM1, TIMESTIM1).fill(1.0); patterns[3].row(2).segment(STARTSTIM2, TIMESTIM2).fill(1.0);

    // Target responses - what the network ought to produce (note that only the last EVALTIME timesteps are actually relevant - see below)
    tgtresps[0] = MatrixXd::Zero(1, TRIALTIME); tgtresps[0].fill(-.98);
    tgtresps[1] = MatrixXd::Zero(1, TRIALTIME); tgtresps[1].fill(.98);
    tgtresps[2] = MatrixXd::Zero(1, TRIALTIME); tgtresps[2].fill(.98);
    tgtresps[3] = MatrixXd::Zero(1, TRIALTIME); tgtresps[3].fill(-.98);




    MatrixXd dJ(NBNEUR, NBNEUR); dJ.setZero();
    MatrixXd win(NBNEUR, NBIN); win.setRandom(); win.row(0).setZero(); // Input weights are uniformly between -1 and 1, except possibly for output cell (not even necessary). No plasticity for input weights.



    MatrixXd J(NBNEUR, NBNEUR);


    randJ(J); // Randomize recurrent weight matrix, according to the Sompolinsky method (Gaussian(0,1), divided by sqrt(ProbaConn*N) and multiplied by G - see definition of randJ() below).

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

    VectorXd meanerrs(NBTRIALS); meanerrs.setZero();
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
    x.fill(0); r.fill(0);

    VectorXd err(TRIALTIME); 
    VectorXd meanerrtrace(NBPATTERNS);
    double meanerr;

    MatrixXd dJtmp, Jprev, Jr;


    double hebbmat[NBNEUR][NBNEUR];
    double rprevmat[NBNEUR], dx2[NBNEUR];
    double xmat[NBNEUR], xtracemat[NBNEUR];

    double dtdivtau = dt / tau;



    meanerrtrace.setZero();



    for (int numtrial=0; numtrial < NBTRIALS; numtrial++)
    {

        trialtype = numtrial % NBPATTERNS;


        // We use native-C array hebbmat for fast computations within the loop, then transfer it back to Eigen matrix hebb for plasticity computations
        hebb.setZero();
        for (int n1=0; n1 < NBNEUR; n1++)
            for (int n2=0; n2 < NBNEUR; n2++)
                hebbmat[n1][n2] = 0;
        r.setZero();
        input.setZero();


        // Initialization with moderate random noise. Decreases performance a bit, but more realistic.

        x.setRandom(); x *= .1; 
        x(1)=1.0; x(10)=1.0;x(11)=-1.0; //x(12) = 1.0;  // Biases
        for (int nn=0; nn < NBNEUR; nn++)
            r(nn) = tanh(x(nn));


        for (int numiter=0; numiter < TRIALTIME;  numiter++)
        {

            input.setZero();
            input(1) = patterns[trialtype].row(1)(numiter);
            input(2) = patterns[trialtype].row(2)(numiter);
            rprev = r;
            lateral_input =  J * r;

            total_exc =  lateral_input + win * input ;




            // Exploratory perturbations
            modul.setZero();
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


            x += dtdivtau * (-x + total_exc);

            x(1)=1.0; x(10)=1.0;x(11)=-1.0; //x(12) = 1.0;  // Biases


            for (int nn=0; nn < NBNEUR; nn++)
            {
                r(nn) = tanh(x(nn));
            }


            rs.col(numiter) = r;

            // Compute the fluctuations of neural activity (detrending / high-pass filtering)
            delta_x =  x  - x_trace ;
            //delta_x_sq = delta_x.array() * delta_x.array().abs();
            //delta_x_cu = delta_x.array() * delta_x.array() * delta_x.array();
            x_trace = ALPHATRACEEXC * x_trace + (1.0 - ALPHATRACEEXC) * x;


            if (DEBUG > 0)
            {
                if (( modulmarker.any() )  || (DEBUG == 2))
                    cout << delta_x_sq.norm() << " " << modul.norm() << " " << total_exc.norm() << " " << x.norm() << " " << (total_exc - x).norm() << " Alignment deltax/total_exc (inc. modul):" 
                        << delta_x.dot(total_exc) / (delta_x.norm() * total_exc.norm()) 
                           << " Alignment deltax/modul:" << delta_x.dot(modul) / (delta_x.norm() * modul.norm()) << " Align deltax_sq/modul:" << delta_x_sq.dot(modul) / (delta_x_sq.norm() * modul.norm()) <<  endl;
            }



            // Computing the Hebbian increment for this time step

            if ( (PHASE == LEARNING) 
                    && (numiter> 2) 
               )
            {
                if (METHOD == "DELTAX")
                {
                    // Method from the paper. Slow, but biologically plausible (-ish).
                    //
                    // The Hebbian increment at every timestep is the inputs (i.e. rprev) times the (cubed) fluctuations in activity for each neuron. 
                    // More plausible, but slower and requires a supralinear function to be applied to the fluctuations (here cubing, but sign-preserving square also works)

                    double incr;
                   for (int n1=0; n1 < NBNEUR; n1++)
                        for (int n2=0; n2 < NBNEUR; n2++)
                        {
                            incr = rprev(n1) * delta_x(n2);
                            hebbmat[n1][n2] +=  incr * incr * incr;
                        }

                }
                else if (METHOD == "NODEPERT")
                {
                    // Node-perturbation. 
                    //
                    // The Hebbian increment is the inputs times the
                    // perturbation itself. Node-perturbation method, similar to 
                    // Fiete & Seung 2006. Much faster 
                    // because you only compute the Hebbian
                    // increments in the few timesteps at which a
                    // perturbation actually occurs.



                    for (int n2=0; n2 < NBNEUR; n2++)
                        if (modulmarker(n2) != 0)
                            for (int n1=0; n1 < NBNEUR; n1++)
                                hebbmat[n1][n2] += rprev(n1) * modul(n2); 


                }
                else { cout << "Which method??" << endl; return -1; }
            }


        }

        int EVALTIME = 200; 

        err = rs.row(0) - tgtresps[trialtype].row(0);
        err.head(TRIALTIME - EVALTIME).setZero();

        meanerr =  err.cwiseAbs().sum() / (double)EVALTIME;

        // We re-transfer the values back from the C arrays to the Eigen matrix
        for (int n1=0; n1 < NBNEUR; n1++)
            for (int n2=0; n2 < NBNEUR; n2++)
                hebb(n1, n2) = hebbmat[n1][n2];


        if ((PHASE == LEARNING) && (numtrial> 100)
           )
        {
            // Note that the weight change is the summed Hebbian increments, multiplied by the mean of recent errors for this trial type - this multiplication may help to stabilize learning.
            dJ = (  -  ETA * meanerrtrace(trialtype) * (hebb.array() * (meanerr - meanerrtrace(trialtype)))).transpose().cwiseMin(MAXDW).cwiseMax(-MAXDW);
            J +=  dJ;


        }


        meanerrtrace(trialtype) = ALPHATRACE * meanerrtrace(trialtype) + (1.0 - ALPHATRACE) * meanerr; 
        meanerrs(numtrial) = meanerr;


        // Display stuff, save files.

        if (PHASE == LEARNING)
        {
            if (numtrial % 3000 < 8) 
            {
                //myfile.open("rs"+std::to_string((numtrial/2)%4)+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
                //myfile.open("rs"+std::to_string(trialtype)+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();

                //myfile.open("rs"+std::to_string(numtrial % 3000)+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
            }
            if ((numtrial % 1000 == 0) || (numtrial == NBTRIALS -1))
            {
                if (numtrial == 0){
                    // Store the initial (random) weights.
                    myfile.open("J_"+std::to_string(numtrial)+SUFFIX+".txt", ios::trunc | ios::out);  myfile << J << endl; myfile.close();
                    saveWeights(J, "J_"+std::to_string(numtrial)+SUFFIX+".dat");
                }
                myfile.open("J" + SUFFIX + ".txt", ios::trunc | ios::out);  myfile << J << endl; myfile.close();
                myfile.open("win" + SUFFIX + ".txt", ios::trunc | ios::out);  myfile << win << endl; myfile.close();
                saveWeights(J, "J" + SUFFIX + ".dat");
                saveWeights(win, "win" + SUFFIX + ".dat"); // win doesn't change over time.

                myfile.open("errs" + SUFFIX + ".txt", ios::trunc | ios::out);  myfile << endl << meanerrs.head(numtrial) << endl; myfile.close();

            }


            //if (numtrial % (NBPATTERNS * 100) <  2*NBPATTERNS)
            if (numtrial % 200 <  2*NBPATTERNS)
            {    
                cout << numtrial << "- trial type: " << trialtype;
                //cout << ", responses : " << zout;
                //cout << ", time-avg responses for each pattern: " << zouttrace ;
                //cout << ", sub(abs(wout)): "  << wout.cwiseAbs().sum() ;
                //cout << ", hebb(0,1:3): " << hebb.col(0).head(4).transpose();
                cout << ", meanerr: " << meanerr;
                //cout << ", wout(0,1:3): " << wout.row(0).head(5) ; 
                cout << ", r(0,1:6): " << r.transpose().head(6) ; 
                cout << ", dJ(0,1:4): " << dJ.row(0).head(4)  ;
                cout << endl;
            }
        }
        else if (PHASE == TESTING) {
            cout << numtrial << "- trial type: " << trialtype;
            cout << " r[0]: " << r(0);
            cout << endl;
            string israndw=""; if (RANDW) israndw = "_RANDW";
            myfile.open("rs_long"+israndw+"_type"+std::to_string(trialtype)+"_"+std::to_string(int(numtrial/NBPATTERNS)) + SUFFIX + ".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
            //myfile.open("rs_long_type"+std::to_string(trialtype)+"_"+std::to_string(int(numtrial/NBPATTERNS)) +  ".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
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

