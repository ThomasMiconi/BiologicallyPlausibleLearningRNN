
// bsub -q short -g /net -oo output2.txt -eo error.txt  -W 10:00 ./net ALPHAINPUTBIAS .5 ALPHAMODUL 16.0 ETA .03 MAXDW 2e-5  <== Good !

// For training, this is like the others:  4 trial types (attend mod1 or mod2, bias of the relevant modality positive or negative [other modality has random-sign bias])
// Variable, randomly chosen input biases at each trial: ALPHAINPUTBIAS just sets the maximum.
// For testing, you set BIAS1 and BIAS2 directly; Only 2 trial types (attend mod1 or mod2), 10 trials for each type (20 trials total).

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <string>
#include <random>
#include "Eigen/Dense"

#define TESTING 777
#define LEARNING 666



using namespace std;
using namespace Eigen;
void saveWeights(MatrixXd& m, string fname);
void readWeights(MatrixXd& m, string fname);
void randJ(MatrixXd& m);


int NBNEUR = 200;
int NBIN = 5;  // Input 0 is reserved for a 'go' signal
int NBOUT = 1;
double PROBACONN = 1.0;
double G = 1.5;
string METHOD = "DELTAX"; // "DELTAX"; //"DELTATOTALEXC"; //"DXTRIAL";
string MODULTYPE = "DECOUPLED";
int RNGSEED = 10;

int DEBUG = 0;

double PROBAMODUL = .003;
double ALPHAMODUL = 16.0;

double PROBAHEBB = 1.0;

double ALPHATRACE = .75;
double ALPHATRACEEXC = 0.05;

double ALPHAINPUTBIAS = 0.5;

int SQUARING = 1;

double MAXDW = 1e-4; // 2e-5 ; //* 1.5;
double ETA =  .01 ; //.03 ; // * 1.5;  // Learning rate
double INPUTMULT = 3.0;

//double STIMVAL = .5;

double BIAS1 = 100000;
double BIAS2 = 100000;

std::default_random_engine myrng;
std::normal_distribution<double> Gauss(0.0,1.0);
std::uniform_real_distribution<double> Uniform(0.0,1.0);

int main(int argc, char* argv[])
{

    fstream myfile;

    double dt = 1.0;
    double tau = 30.0;

    int NBTRIALTYPES = 4; // 2 if Testing

    int PHASE=LEARNING;
    if (argc > 1)
       for (int nn=1; nn < argc; nn++)
       {
           if (strcmp(argv[nn], "TEST") == 0) { PHASE = TESTING; cout << "Test mode. " << endl; }
           if (strcmp(argv[nn], "METHOD") == 0) { METHOD = argv[nn+1]; }
           if (strcmp(argv[nn], "MODULTYPE") == 0) { MODULTYPE = argv[nn+1]; }
           if (strcmp(argv[nn], "SQUARING") == 0) { SQUARING = atoi(argv[nn+1]); }
           if (strcmp(argv[nn], "DEBUG") == 0) { DEBUG = atoi(argv[nn+1]); }
           if (strcmp(argv[nn], "G") == 0) { G = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "ALPHAINPUTBIAS") == 0) { ALPHAINPUTBIAS = atof(argv[nn+1]);  if (PHASE == TESTING) { std::runtime_error( "In Testing phase, do not specify ALPHAINPUTBIAS!");  } }
           if (strcmp(argv[nn], "ETA") == 0) { ETA = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "TAU") == 0) { tau = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "INPUTMULT") == 0) { INPUTMULT = atof(argv[nn+1]); }
           //if (strcmp(argv[nn], "STIMVAL") == 0) { STIMVAL = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "ALPHAMODUL") == 0) { ALPHAMODUL = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "PROBAMODUL") == 0) { PROBAMODUL = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "PROBAHEBB") == 0) { PROBAHEBB = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "BIAS1") == 0) { BIAS1 = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "BIAS2") == 0) { BIAS2 = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "ALPHATRACE") == 0) { ALPHATRACE = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "ALPHATRACEEXC") == 0) { ALPHATRACEEXC = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "RNGSEED") == 0) { RNGSEED = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "MAXDW") == 0) { MAXDW = atof(argv[nn+1]); }
       }

    if (PHASE == TESTING)
    {
        NBTRIALTYPES = 2; // Only 2 trial types - the negative biases don't constitute separate trial types for testing purposes (makes data collection easier)
        if ((BIAS1 > 1000) || (BIAS2 > 1000))
             throw std::runtime_error( "In testing phase, must specify BIAS1 and BIAS2 "); 
    }

    string SUFFIX = "_G" + to_string(G) + "_MAXDW" + to_string(MAXDW) + "_ETA" + to_string(ETA) + "_ALPHAMODUL" + to_string(ALPHAMODUL) + "_PROBAMODUL" + to_string(PROBAMODUL) + "_SQUARING" +to_string(SQUARING) 
        + "_MODULTYPE-" + MODULTYPE +   
         "_ALPHATRACE" + to_string(ALPHATRACE) + "_METHOD-" + METHOD + "_ALPHAINPUTBIAS" + to_string(ALPHAINPUTBIAS) + "_PROBAHEBB" + to_string(PROBAHEBB) + "_ATRACEEXC" + to_string(ALPHATRACEEXC) + "_TAU" + to_string(tau) +
        "_INPUTMULT" + to_string(INPUTMULT) + 
        "_RNGSEED" + to_string(RNGSEED);
    cout << SUFFIX << endl;

    myrng.seed(RNGSEED);
    srand(RNGSEED);

    int trialtype;

    int NBTRIALS =  50401; 
    int TRIALTIME = 700;
    int STARTSTIM1 = 1, TIMESTIM1 = 500; 
    //int STARTSTIM2 = 400, TIMESTIM2 = 200; 
    /*int TRIALTIME = 1000;
    int STARTSTIM1 = 1, TIMESTIM1 = 200; // 200
    int STARTSTIM2 = 400, TIMESTIM2 = 200; */

    int marker =0;


    


    MatrixXd dJ(NBOUT, NBNEUR); dJ.setZero();
     MatrixXd win(NBNEUR, NBIN); win.setRandom();  win.row(0).setZero();  // win.topRows(NBNEUR/2).setZero(); // randMat(win); //win.setRandom();// win.row(0).setZero(); // Uniformly between -1 and 1, except possibly for output cell (not even necessary).
    // cout << win.col(0).head(5) << endl;
    MatrixXd J(NBNEUR, NBNEUR);
    
    randJ(J);
    

    //J.rightCols(NBIN).setRandom();



    if (PHASE == TESTING){
        readWeights(J, "J.dat");
        readWeights(win, "win.dat");
        NBTRIALS = 10;  // In total, across all trial types.
    }



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
    VectorXd meanerrtrace(NBTRIALTYPES);
    double meanerr;

    MatrixXd dJtmp, Jprev, Jr;


    double hebbmat[NBNEUR][NBNEUR];
    double rprevmat[NBNEUR], dx2[NBNEUR];
    double xmat[NBNEUR], xtracemat[NBNEUR];

    double dtdivtau = dt / tau;



    meanerrtrace.setZero();



    for (int numtrial=0; numtrial < NBTRIALS; numtrial++)
    {

    
        trialtype = numtrial % NBTRIALTYPES;

        
        for (int n1=0; n1 < NBNEUR; n1++)
            for (int n2=0; n2 < NBNEUR; n2++)
                hebbmat[n1][n2] = 0;
        hebb.setZero();
        dJ.setZero();
        //input = patterns.col(trialtype);
        input.setZero();
        
        // Initialize network activations with small random noise 
        //x.fill(0.0); 
        x.setRandom(); x *= .1; 
        x(1)=1.0; x(10)=1.0; x(11)=-1.0; // Biases 
        //x += dtdivtau * win * input;
        for (int nn=0; nn < NBNEUR; nn++)
            r(nn) = tanh(x(nn));

        


        double biasmodality1, biasmodality2, tgtresp;
        
        biasmodality1=0; biasmodality2 = 0;

        if (trialtype == 0){  // modality 1 positive bias, modality 2 random, look at modality 1
            input(3) = 1.0; input(4) = 0.0;
            biasmodality1 = 1.0;
            tgtresp = 1.0;
            biasmodality2 = Uniform(myrng) < .5 ? 1.0 : -1.0;
        } 
        else if (trialtype == 1){  // modality 1 random, modality 2 positive bias, look at modality 2
            input(3) = 0.0; input(4) = 1.0;
            biasmodality2 = 1.0;
            tgtresp = 1.0;
            biasmodality1 = Uniform(myrng) < .5 ? 1.0 : -1.0;
        } 
        else if (trialtype == 2){  // modality 1 negative bias, modality 2 random, look at modality 1
            input(3) = 1.0; input(4) = 0.0;
            biasmodality1 = -1.0;
            tgtresp = -1.0;
            biasmodality2 = Uniform(myrng) < .5 ? 1.0 : -1.0;
        } 
        else if (trialtype == 3){  // modality 1 random, modality 2 negative bias, look at modality 2
            input(3) = 0.0; input(4) = 1.0;
            biasmodality2 = -1.0;
            tgtresp = -1.0;
            biasmodality1 = Uniform(myrng) < .5 ? 1.0 : -1.0;
        } 
        else { cout << "Which trial type?" << endl; return -1;}
        
        /*
         // This doesn't work ! You really need to maintain one reward predictor for each possible stimulus combination
        if (trialtype == 0){  //Look at modality 1
            input(3) = 1.0; input(4) = 0.0;
            biasmodality2 = Uniform(myrng) < .5 ? 1.0 : -1.0;
            if (Uniform(myrng) < .5){
                biasmodality1 = 1.0;
                tgtresp = .99;
            }
            else{
                biasmodality1 = -1.0;
                tgtresp = -.99;
            }
        } 
        else if (trialtype == 1){ // Look at modality 2 
            input(3) = 0.0; input(4) = 1.0;
            biasmodality1 = Uniform(myrng) < .5 ? 1.0 : -1.0;
            if (Uniform(myrng) < .5){
                biasmodality2 = 1.0;
                tgtresp = .99;
            }
            else{
                biasmodality2 = -1.0;
                tgtresp = -.99;
            }
        } 
        else { cout << "Which trial type?" << endl; return -1;}
        */


        double biasmodulator1;
        double biasmodulator2;


        biasmodulator1 = .2;
        biasmodulator2 = .2;
        
        biasmodality1 *= biasmodulator1 ;
        biasmodality2 *= biasmodulator2 ;
        
        if (PHASE == TESTING)
        {
            biasmodality1 = BIAS1;
            biasmodality2 = BIAS2;
        }




        for (int numiter=0; numiter < TRIALTIME;  numiter++)
        {

            input(0) = 0.0; input(1) = 0.0; input(2) = 0.0;
            if (numiter >= STARTSTIM1  & numiter <  STARTSTIM1 + TIMESTIM1)
            {
                input(1) =  .5 * ( Gauss(myrng) + biasmodality1 );
                input(2) =  .5 * ( Gauss(myrng) + biasmodality2 );
            }

            rprev = r;
            lateral_input =  J * r;
           


 
            total_exc =  lateral_input   + win * input;

            
            modul.setZero();

            if (MODULTYPE == "DECOUPLED")
            {
                for (int nn=0; nn < NBNEUR; nn++)
                    if ( (Uniform(myrng) < PROBAMODUL)
                            && (numiter> 3)
                       )
                    {
                        total_exc(nn) += ALPHAMODUL * (-1.0 + 2.0 * Uniform(myrng));
                    }
            }
            else { throw std::runtime_error("Which modulation type?"); }

            delta_total_exc =  total_exc - total_exc_prev;
            delta_total_exc_sq = delta_total_exc.array() * delta_total_exc.array().abs();
            total_exc_prev = ALPHATRACEEXC * total_exc_prev + (1.0 - ALPHATRACEEXC) * total_exc; 
            
            x += dtdivtau * (-x + total_exc);
           
            x(1)=1.0; x(10)=1.0;x(11)=-1.0; //x(12) = 1.0; 



            for (int nn=0; nn < NBNEUR; nn++)
            {
                r(nn) = tanh(x(nn));
            }
            
/*
            //r.tail(NBIN) =  INPUTMULT * input;
            r.tail(NBIN) =  input;
            r.tail(2*NBIN).head(NBIN) = input;
            r.tail(3*NBIN).head(NBIN) = input;
            r.tail(4*NBIN).head(NBIN) = input;
*/
            
            rs.col(numiter) = r;
         



            delta_x =  x  - x_trace ;
            delta_x_sq = delta_x.array() * delta_x.array().abs();
            delta_x_cu = delta_x.array() * delta_x.array() * delta_x.array();
            x_trace = ALPHATRACEEXC * x_trace + (1.0 - ALPHATRACEEXC) * x;


            if (DEBUG > 0)
            {
                if ((marker == 1)  || (DEBUG == 2))
                cout << delta_x_sq.norm() << " " << modul.norm() << " " << total_exc.norm() << " " << x.norm() << " " << (total_exc - x).norm() << " Alignment deltax/total_exc (inc. modul):" 
                    << delta_x.dot(total_exc) / (delta_x.norm() * total_exc.norm()) 
                << " Alignment deltax/modul:" << delta_x.dot(modul) / (delta_x.norm() * modul.norm()) << " Align deltax_sq/modul:" << delta_x_sq.dot(modul) / (delta_x_sq.norm() * modul.norm()) <<  endl;
            }

            

                if ((PHASE == LEARNING) && (Uniform(myrng) < PROBAHEBB) 
                        && (numiter> 2) 
                        //&& (abs(numiter - 200) > 5)
                        //&& (abs(numiter - 400) > 5) && (abs(numiter - 600) > 5) 
                        )
                {
                    if (METHOD == "DELTAX")
                    {
                            for (int n1=0; n1 < NBNEUR; n1++)
                            {
                                rprevmat[n1] = rprev(n1);
                                //if (marker == 1){
                                dx2[n1] = delta_x_cu(n1);
                            }
                            for (int n1=0; n1 < NBNEUR; n1++)
                                for (int n2=0; n2 < NBNEUR; n2++)
                                    hebbmat[n1][n2] += rprevmat[n1] * dx2[n2]; //rprev(n1) * delta_x_sq(n2);
                    }
                    else { cout << "Which method??" << endl; return -1; }
                }

            
        }
       
       int EVALTIME = 200; 

        err = rs.row(0).array() - tgtresp;
        err.head(TRIALTIME - EVALTIME).setZero();

        //meanerr = 2.0   * err.cwiseAbs().sum();
        meanerr =  err.cwiseAbs().sum() / (double)EVALTIME;

        for (int n1=0; n1 < NBNEUR; n1++)
            for (int n2=0; n2 < NBNEUR; n2++)
                hebb(n1, n2) = hebbmat[n1][n2];


        if ((PHASE == LEARNING) && (numtrial> 100)
                // && (numtrial %2 == 1)
           )
            {
             
                
                // dJ = -.0001 * meanerr.sum() * (hebb.array() * (meanerr.sum() - meanerrtrace.col(trialtype).sum())).transpose().cwiseMin(5e-4).cwiseMax(-5e-4); << Version that worked
                //dJ = (  -.0000001 * meanerr * (hebb.array() * (meanerr - meanerrtrace(trialtype)))).transpose().cwiseMin(1e-6).cwiseMax(-1e-6); 
                //dJ = (  -.000005 * meanerr * (hebb.array() * (meanerr - meanerrtrace(trialtype)))).transpose().cwiseMin(5e-5).cwiseMax(-5e-5);
                //dJ = G * (  -  ETA * meanerr * (hebb.array() * (meanerr - meanerrtrace(trialtype)))).transpose().cwiseMin(MAXDW).cwiseMax(-MAXDW);
                
                //dJ = (  -  ETA * meanerrtrace(trialtype) * meanerrtrace(trialtype) * (hebb.array() * (meanerr - meanerrtrace(trialtype)))).transpose().cwiseMin(MAXDW).cwiseMax(-MAXDW);
                dJ = (  -  ETA * meanerrtrace(trialtype) * (hebb.array() * (meanerr - meanerrtrace(trialtype)))).transpose().cwiseMin(MAXDW).cwiseMax(-MAXDW);


                //dJ.rightCols(NBIN).setZero();



                //J /= G;
                J +=  dJ;
                //J *= G;


            }


        meanerrtrace(trialtype) = ALPHATRACE * meanerrtrace(trialtype) + (1.0 - ALPHATRACE) * meanerr; 
        //meanerrtrace(trialtype) = meanerr; 
        meanerrs(numtrial) = meanerr;
       

        if (PHASE == LEARNING)
        {
            if (numtrial % 3000 < 8) 
            {
                //myfile.open("rs"+std::to_string((numtrial/2)%4)+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
                //myfile.open("rs"+std::to_string(trialtype)+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
                
                //myfile.open("rs"+std::to_string(numtrial % 3000)+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
            }
            if ((numtrial % 10000 == 0) || (numtrial == NBTRIALS - 1))
            {
                saveWeights(J, "J.dat");
                saveWeights(win, "win.dat"); // win doesn't change over time.
                myfile.open("J_"+std::to_string(numtrial)+".txt", ios::trunc | ios::out);  myfile << J << endl; myfile.close();
                saveWeights(J, "J_"+std::to_string(numtrial)+".dat");
                myfile.open("J" + SUFFIX + ".txt", ios::trunc | ios::out);  myfile << J << endl; myfile.close();
                myfile.open("win" + SUFFIX + ".txt", ios::trunc | ios::out);  myfile << win << endl; myfile.close();
                saveWeights(J, "J" + SUFFIX + ".dat");
                saveWeights(win, "win" + SUFFIX + ".dat"); // win doesn't change over time.
                
                myfile.open("errs" + SUFFIX + ".txt", ios::trunc | ios::out);  myfile << endl << meanerrs.head(numtrial) << endl; myfile.close();

            }


            if (numtrial % (NBTRIALTYPES * 100) <  4*NBTRIALTYPES)
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
            //myfile.open("rs_long_type"+std::to_string(trialtype)+"_alphabias"+ std::to_string(ALPHAINPUTBIAS)+ "_"+std::to_string(int(numtrial/NBTRIALTYPES))+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
            //myfile.open("lastr_type"+std::to_string(trialtype)+"_alphabias"+ std::to_string(ALPHAINPUTBIAS)+ "_"+std::to_string(int(numtrial/NBTRIALTYPES))+".txt", ios::trunc | ios::out);  myfile << endl << r(0) << endl; myfile.close();

            myfile.open("rs_long_type"+std::to_string(trialtype)+"_bias1_"+ std::to_string(BIAS1)+ "_bias2_"+ std::to_string(BIAS2)+ "_"+std::to_string(int(numtrial/ NBTRIALTYPES))+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
            myfile.open("lastr_type"+std::to_string(trialtype)+"_bias1_"+ std::to_string(BIAS1) + "_bias2_"+ std::to_string(BIAS2)+ "_"+std::to_string(int(numtrial/ NBTRIALTYPES))+".txt", ios::trunc | ios::out);  myfile << endl << r(0) << endl; myfile.close();
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

/*
load rs0.txt; load rs1.txt; load rs2.txt; load rs3.txt;
figure; plot(rs0(1:7,:)'); figure; plot(rs1(1:7,:)'); figure; plot(rs2(1:7,:)'); figure; plot(rs3(1:7,:)'); 

r=load('resp0.txt');
figure; plot(r(2:8:end)); hold on; plot(r(3:8:end), 'r'); plot(r(5:8:end), 'g'); plot(r(7:8:end), 'm'); hold off

*/
