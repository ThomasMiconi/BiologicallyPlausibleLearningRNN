This contains the additional experiments described in the paper. These include:

* `dnmslongdelay`: The delayed non-match to sample experiment with long (1000 ms) delays.

* `dnmsvariabledelay`: The delayed non-match to sample experiment with variable inter-stimulus intervals (300 to 800 ms).

* `gradients`: Using a simple experiment, compute the reward gradient over the weights according to various methods, and compare themn with the backpropagation-computed "true" error gradient.

* `EInet`: The delayed non-match to sample experiment, using a network with
non-negative responses and separate populations of excitatory and inhibitory
neurons (Dale's law). This builds on the work of [Mastrogiuseppe and Ostojic](https://arxiv.org/abs/1605.04221),
who studied the regime in which such networks generate the same ongoing chaotic
activity as the Sompolinksy-type recurent networks.


Note that `EInet` contains an additional Python file (EInet.py) to simulate networks of this type with various settings. 

