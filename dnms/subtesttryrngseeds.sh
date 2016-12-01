# NOTE: this is just to test the effects of multiple random seeds, do not use.

i=0
                                    for RNGSEED in 1 2 3 4 5 6 7 8 9 10  11 12 13 14 15 16 17 18 19 20
                                    do
                                        i=$((i+1))
                                        echo $i

                                        #mycmd="bsub -q short -g /net  -W 10:00 ./net test PROBAMODUL 0.001 ALPHATRACE 0.5 ALPHAMODUL 32.0 ETA 0.03 MAXDW 3e-4   RNGSEED $RNGSEED "
                                        mycmd="bsub -q short -g /net  -W 10:00 ./net test MAXDW 0.000300 ETA 0.100000 ALPHAMODUL 16.000000 PROBAMODUL 0.003000 ALPHATRACE 0.750000  RNGSEED $RNGSEED "

                                        echo $mycmd
                                        $mycmd

                                    done

