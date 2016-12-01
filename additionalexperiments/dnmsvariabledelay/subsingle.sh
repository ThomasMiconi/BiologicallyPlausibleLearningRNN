# NOTE: This is to generate the training curve figure - starts 20 learning runs.
# the "single" refers to fact that we're only using a single combination of
# parameters, as opposed to sub.sh which tests many combinations in a grid
# search

i=0
                                    for RNGSEED in 0 1 2 3 4 5 6 7 8 9 10  11 12 13 14 15 16 17 18 19 20
                                    do
                                        i=$((i+1))
                                        echo $i

                                        mycmd="bsub -q short -g /net  -W 10:00 ./net  MAXDW 0.000300 ETA 0.100000 ALPHAMODUL 16.000000 PROBAMODUL 0.003000 ALPHATRACE 0.750000  RNGSEED $RNGSEED "

                                        echo $mycmd
                                        $mycmd

                                    done

