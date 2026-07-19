Symmetric is inefficient:

Why the big time jump depth 1 -> 2?
    d=2 is a lot slower than BFS
            ---HIGGS chrono should look like trunk 1.5m

    [X] run FINE chrono & see which part jumps 1 -> 2
            50% of time is Sort. This is unnecessary - bags are individually sorted

            K-way merge is the normal way. But:
                current sorted bag is last depth's sorted bag.

