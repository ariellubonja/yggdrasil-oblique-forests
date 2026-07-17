GBT Chrono

[X] Correct params to use : 300 trees + depth 6

[ ] e2e

Dynamic is working. Runtime Vectorized Hist + Dynamic

which one did we run?

Reported time is wrong by runtime.sh . rerun

    - [X] Dynamic Random
        
    - [ ] Vectorized Random used? No. fix 
        n_bins is also ignored. No binning on the fly?
    

chrono

    [ ] Extend vectorized to FindSplitRegerssionHistogram
        [X] Find a better code way for it to generalize across all methods that use Random Hist
        ✅ No other methods benefit 
        [ ] test accuracy
            [ ] implement test on _test.csv - test set

    [X] Remove isnan in FindSplitRegerssionHistogram

    [X] Add histogramming cols

    [ ] Proportion of AP-EP by dataset

    [ ] Why is 15k x 400k so slow e2e

