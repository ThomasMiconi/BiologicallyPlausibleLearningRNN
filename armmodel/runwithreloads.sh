# The arm model experiment has a tendency to crash. This script runs the
# experiment as successive short runs, each time reloading the data file from
# the previous run, to approximate one long run in a way that is robust to
# crashes. 

./armmodel NBTRIALS 501 RNGSEED 1 PROBAMODUL .003 ALPHAMODUL 8.0 ETA .1 MAXDW 5e-5
for i in `seq 2 100`;
do
    ./armmodel RELOAD  NBTRIALS 501 RNGSEED $i PROBAMODUL .003 ALPHAMODUL 8.0 ETA .1  MAXDW 5e-5  # AM 4.0
    sleep 10
done

