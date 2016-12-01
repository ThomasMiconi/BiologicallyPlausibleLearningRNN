# Grid search for parameters

i=0
#for G in 1.2 1.5 1.8
for G in 1.5 
do
                        for PROBAMODUL in   .003 .001 .1
                        do
                        for MAXDW in     1e-5 3e-5 3e-4 
                        do
                            for ETA in .01 .03 .003   #  3.0 1.5 .5  # 5.0 10.0 20.0 
                            do
                                    for RNGSEED in 1 2 3 4 5 6 7 8 9 10  11 12 13 14 15 # 16 17 18 19 20
                                    do
                                        i=$((i+1))
                                        echo $i

                                        #suffix="-TAU$TAU-ALPHAMODUL$ALPHAMODUL-G$G-ETA$ETA-MAXDW$MAXDW-ATRACE$ALPHATRACE-ATRACEEXC$ALPHATRACEEXC-PROBAMODUL$PROBAMODUL-SQUARING$SQUARING-NDUPL$NDUPL-RNGSEED$RNGSEED"   #PROBAMODUL$PROBAMODUL
                                        suffix="-ETA$ETA-MAXDW$MAXDW-PROBAMODUL$PROBAMODUL-RNGSEED$RNGSEED"

                                        #mycmd="bsub -q short -g /net -oo output$suffix.txt -eo error$suffix.txt  -W 10:00 ./net RNGSEED $RNGSEED ETA $ETA MAXDW $MAXDW G $G ALPHATRACEEXC $ALPHATRACEEXC ALPHAMODUL $ALPHAMODUL PROBAMODUL $PROBAMODUL SQUARING $SQUARING TAU $TAU ALPHATRACE $ALPHATRACE NDUPL $NDUPL"
                                        mycmd="bsub -q short -g /net -oo output$suffix.txt -eo error$suffix.txt  -W 10:00 ./net RNGSEED $RNGSEED ETA $ETA MAXDW $MAXDW PROBAMODUL $PROBAMODUL"

                                        echo $mycmd
                                        $mycmd

            done
            done
        done
    done
done
