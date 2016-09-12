
i=0
for BIAS1 in  -.5 -.4  -.3  -.2   -.1 0.0  .1  .2 .3 .4 .5
do
for BIAS2 in  -.5 -.4  -.3  -.2   -.1 0.0  .1  .2 .3 .4 .5
do
    i=$((i+1))
    echo $i


    #mycmd="bsub -q short -g /subtest -oo output$i.txt -eo error$i.txt  -W 10:00 ./net TEST BIAS1 $BIAS1 BIAS2 $BIAS2 "
    mycmd="bsub -q short -g /subtest -oo output$i.txt -eo error$i.txt  -W 2:00 ./net TEST  ETA .01 ALPHAMODUL 30.0 MAXDW 2e-4 ALPHABIAS .5 BIAS1 $BIAS1 BIAS2 $BIAS2"  # Note that PROBAMODUL is not set to zero - perturbations even in test!

    echo $mycmd
    $mycmd

done
done

