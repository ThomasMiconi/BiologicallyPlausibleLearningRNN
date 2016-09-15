# Run the code with 20 different random seeds.

i=0
#for G in 1.2 1.5 1.8
                                    for RNGSEED in 1 2 3 4 5 6 7 8 9 10  11 12 13 14 15 16 17 18 19 20
                                    do
                                        i=$((i+1))
                                        echo $i

                                        suffix="-RNGSEED$RNGSEED"   #PROBAMODUL$PROBAMODUL

                                        mycmd="bsub -q short -g /net -oo output$suffix.txt -eo error$suffix.txt  -W 10:00 ./net RNGSEED $RNGSEED"

                                        echo $mycmd
                                        $mycmd

                                    done
