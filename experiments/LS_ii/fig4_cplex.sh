#!/bin/bash

for case; do

    # get a few parameters 
    N=$(grep num_sources $case | egrep -o '[[:digit:]]{1,4}')
    rmin=$(grep '"rmin' $case | egrep -o '[[:digit:]]{0,4}\.{0,1}[[:digit:]]{0,4}')
    beta=$(grep '"beta' $case | egrep -o '[[:digit:]]{0,4}\.{0,1}[[:digit:]]{0,4}')
    gamma_m=$(grep gamma_m $case | egrep -o '[[:digit:]]{0,4}\.{0,1}[[:digit:]]{0,4}')
    gamma_v=$(grep gamma_v $case | egrep -o '[[:digit:]]{0,4}\.{0,1}[[:digit:]]{0,4}')
    resultfile=$(grep resultfile $case | cut --delimiter='"' --fields=4)
    outdir=$(grep outdir $case | cut --delimiter='"' --fields=4)

    # state some basic info to std
    echo "the configuration file is $case"
    echo "rmin = $rmin, beta = $beta, gamma_m = $gamma_m and gamma_v = $gamma_v"
    echo "hdf5 result is $resultfile, and the output directory is $outdir/"

    # name log file
    log=$(echo "$case" | sed "s/json/log/g")
    echo "log file is $log"

    # run forward
    mpiexec -np $N python fwd.py -i vp_ls_complex_exact.hdf5 -c $case

    # run inversion
    mpiexec -np $N python fwi_cplex.py -O cplex -c $case > $log

done
