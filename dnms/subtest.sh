!#/bin/sh

i=0
                                    for RNGSEED in 1 2 3 4 5 6 7 8 9 10  11 12 13 14 15 16 17 18 19 20
                                    do
                                        i=$((i+1))
                                        echo $i

                                        mycmd="bsub -q short -g /net  -W 10:00 ./net test RNGSEED $RNGSEED "

                                        echo $mycmd
                                        $mycmd

                                    done

