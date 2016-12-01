# Grid search for parameters

i=0
#for G in 1.2 1.5 1.8
for G in 1.5 
do
                                    for RNGSEED in 1 2 3 4 5 6 7 8 9 10  11 12 13 14 15  16 17 18 19 20
                                    do
                                        i=$((i+1))
                                        echo $i

                                        #suffix="-TAU$TAU-ALPHAMODUL$ALPHAMODUL-G$G-ETA$ETA-MAXDW$MAXDW-ATRACE$ALPHATRACE-ATRACEEXC$ALPHATRACEEXC-PROBAMODUL$PROBAMODUL-SQUARING$SQUARING-NDUPL$NDUPL-RNGSEED$RNGSEED"   #PROBAMODUL$PROBAMODUL
                                        suffix="-RNGSEED$RNGSEED"

                                        #mycmd="bsub -q short -g /net -oo output$suffix.txt -eo error$suffix.txt  -W 10:00 ./net RNGSEED $RNGSEED ETA $ETA MAXDW $MAXDW G $G ALPHATRACEEXC $ALPHATRACEEXC ALPHAMODUL $ALPHAMODUL PROBAMODUL $PROBAMODUL SQUARING $SQUARING TAU $TAU ALPHATRACE $ALPHATRACE NDUPL $NDUPL"
                                        mycmd="bsub -q short -g /net -oo output$suffix.txt -eo error$suffix.txt  -W 10:00 ./net RNGSEED $RNGSEED" 

                                        echo $mycmd
                                        $mycmd

            done
            done
