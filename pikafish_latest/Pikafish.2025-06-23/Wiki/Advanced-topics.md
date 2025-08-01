# A note on NNUE evaluation

This approaches assign a value to a position that is used in alpha-beta (PVS) search to find the best move. The NNUE evaluation computes this value with a neural network based on basic inputs (e.g. piece positions only). The network is optimized and trained on the evaluations of millions of positions at moderate search depth.

The NNUE evaluation was first introduced in shogi, and ported to Stockfish afterward. It can be evaluated efficiently on CPUs, and exploits the fact that only parts of the neural network need to be updated after a typical chess move. [The nodchip repository](https://github.com/nodchip/Stockfish) provided the first version of the needed tools to train and develop the NNUE networks. Today, more advanced training tools are available in [the nnue-pytorch repository](https://github.com/official-pikafish/pikafish-nnue-pytorch), while data generation tools are available in [a dedicated branch](https://github.com/official-pikafish/Pikafish/tree/tools).

On CPUs supporting modern vector instructions (avx2 and similar), the NNUE evaluation results in much stronger playing strength, even if the nodes per second computed by the engine is somewhat lower (roughly 50% of nps is typical).

Notes:

1. the NNUE evaluation depends on the Pikafish binary and the network parameter file (see the EvalFile UCI option). Not every parameter file is compatible with a given Pikafish binary, but the default value of the EvalFile UCI option is the name of a network that is guaranteed to be compatible with that binary.

2. to use the NNUE evaluation, the additional data file with neural network parameters needs to be available. The filename for the default (recommended) net can be found as the default value of the `EvalFile` UCI option, with the format `pikafish.nnue`. This file can be downloaded from `http://test.pikafish.org`.

---

## Large Pages

Pikafish supports large pages on Linux and Windows. Large pages make the hash access more efficient, improving the engine speed, especially on large hash sizes.  
The support is automatic, Pikafish attempts to use large pages when available and will fall back to regular memory allocation when this is not the case.  
Typical increases are 5-10% in terms of nodes per second, but speed increases up to 30% have been measured.

### Linux

Large page support on Linux is obtained by the Linux kernel transparent huge pages functionality. Typically, transparent huge pages are already enabled, and no configuration is needed.

### Windows

The use of large pages requires "Lock Pages in Memory" privilege. See [Enable the Lock Pages in Memory Option (Windows)](https://docs.microsoft.com/en-us/sql/database-engine/configure-windows/enable-the-lock-pages-in-memory-option-windows) on how to enable this privilege, then run [RAMMap](https://docs.microsoft.com/en-us/sysinternals/downloads/rammap) to double-check that large pages are used.  
We suggest that you reboot your computer after you have enabled large pages, because long Windows sessions suffer from memory fragmentation, which may prevent Pikafish from getting large pages: a fresh session is better in this regard.

---

## Measure the speed of Pikafish

The "speed of Pikafish" is the number of nodes (positions) Pikafish can search per second. 
Nodes per second (nps) is a useful benchmark number as the same version of Pikafish playing will play stronger with larger nps.
Different versions of Pikafish will play at different nps, for example, if the NNUE network architecture changes, but in this case the nps difference is not related to the strength difference.

Notes:
* Stop all other applications when measuring the speedup of Pikafish
* Run at least 20 default benches (depth 13) for each build of Pikafish to have accurate measures
* A speedup of 0.3% could be meaningless (i.e. within the measurement noise)

To measure the speed of several builds of Pikafish, use one of these applications:
* All OS:
  * [pyshbench](https://github.com/hazzl/pyshbench): Latest release [pyshbench](https://github.com/hazzl/pyshbench/archive/master.zip)
  * bash script `bench_parallel.sh` (run `bash bench_parallel.sh` for the help)  
    it might be that you have to install gawk as well on your system to not get syntax errors.

    <details><summary>Click to view</summary>
    
    ```bash
    #!/bin/bash
    _bench () {
    ${1} << EOF > /dev/null 2>> ${2}
    bench 16 1 ${depth} default depth
    EOF
    }
    # _bench function customization example
    # setoption name SyzygyPath value C:\table_bases\wdl345;C:\table_bases\dtz345
    # bench 128 4 ${depth} default depth
    
    if [[ ${#} -ne 4 ]]; then
    cat << EOF
    usage: ${0} ./pikafish_base ./pikafish_test depth n_runs
    fast bench:
    ${0} ./pikafish_base ./pikafish_test 13 10
    slow bench:
    ${0} ./pikafish_base ./pikafish_test 20 10
    EOF
    exit 1
    fi
    
    pf_base=${1}
    pf_test=${2}
    depth=${3}
    n_runs=${4}
    
    # preload of CPU/cache/memory
    printf "preload CPU"
    (_bench ${pf_base} pf_base0.txt)&
    (_bench ${pf_test} pf_test0.txt)&
    wait
    
    # temporary files initialization
    : > pf_base0.txt
    : > pf_test0.txt
    : > pf_temp0.txt
    
    # bench loop: SMP bench with background subshells
    for ((k=1; k<=${n_runs}; k++)); do
        printf "\rrun %3d /%3d" ${k} ${n_runs}
    
        # swap the execution order to avoid bias
        if [ $((k%2)) -eq 0 ]; then
            (_bench ${sf_base} pf_base0.txt)&
            (_bench ${sf_test} pf_test0.txt)&
            wait
        else
            (_bench ${sf_test} pf_test0.txt)&
            (_bench ${sf_base} pf_base0.txt)&
            wait
        fi
    done
    
    # text processing to extract nps values
    cat pf_base0.txt | grep second | grep -Eo '[0-9]{1,}' > pf_base1.txt
    cat pf_test0.txt | grep second | grep -Eo '[0-9]{1,}' > pf_test1.txt
    
    for ((k=1; k<=${n_runs}; k++)); do
        echo ${k} >> pf_temp0.txt
    done
    
    printf "\rrun   pf_base   pf_test      diff\n"
    paste pf_temp0.txt pf_base1.txt pf_test1.txt | awk '{printf "%3d  %8d  %8d  %8+d\n", $1, $2, $3, $3-$2}'
    #paste pf_temp0.txt pf_base1.txt pf_test1.txt | awk '{printf "%3d\t%8d\t%8d\t%7+d\n", $1, $2, $3, $3-$2}'
    paste pf_base1.txt pf_test1.txt | awk '{printf "%d\t%d\t%d\n", $1, $2, $2-$1}' > pf_temp0.txt
    
    # compute: sample mean, 1.96 * std of sample mean (95% of samples), speedup
    # std of sample mean = sqrt(NR/(NR-1)) * (std population) / sqrt(NR)
    cat pf_temp0.txt | awk '{sum1 += $1 ; sumq1 += $1**2 ;sum2 += $2 ; sumq2 += $2**2 ;sum3 += $3 ; sumq3 += $3**2 } END {printf "\nsf_base = %8d +/- %6d (95%)\nsf_test = %8d +/- %6d (95%)\ndiff    = %8d +/- %6d (95%)\nspeedup = %.5f% +/- %.3f% (95%)\n\n", sum1/NR , 1.96 * sqrt(sumq1/NR - (sum1/NR)**2)/sqrt(NR-1) , sum2/NR , 1.96 * sqrt(sumq2/NR - (sum2/NR)**2)/sqrt(NR-1) , sum3/NR  , 1.96 * sqrt(sumq3/NR - (sum3/NR)**2)/sqrt(NR-1) , 100*(sum2 - sum1)/sum1 , 100 * (1.96 * sqrt(sumq3/NR - (sum3/NR)**2)/sqrt(NR-1)) / (sum1/NR) }'
    
    # remove temporary files
    rm -f pf_base0.txt pf_test0.txt pf_temp0.txt pf_base1.txt pf_test1.txt
    ```
    </details>

    <details><summary>Bench two git branches with bench_parallel</summary>

    ```bash
    #!/bin/bash
    if [ "$#" -ne 5 ]; then
    echo "Usage: $0 branch1 branch2 depth runs compile_flags"
    exit 1
    fi
    BRANCH1=$1
    BRANCH2=$2
    DEPTH=$3
    RUNS=$4
    COMPILE_FLAGS=$5
    set -e
    echo "Switching to $BRANCH1 and building..."
    git switch $BRANCH1
    make clean
    make -j profile-build EXE=pikafish-$BRANCH1 $COMPILE_FLAGS
    echo "Switching to $BRANCH2 and building..."
    git switch $BRANCH2
    make clean
    make -j profile-build EXE=pikafish-$BRANCH2 $COMPILE_FLAGS
    echo "Running bench_parallel.sh with pikafish-$BRANCH1 and pikafish-$BRANCH2..."
    ./bench_parallel.sh ./pikafish-$BRANCH1 ./pikafish-$BRANCH2 $DEPTH $RUNS
    ```
    </details>

* Windows only:
  * [FishBench](https://github.com/zardav/FishBench): Latest release [Fishbench v6.0](https://github.com/zardav/FishBench/releases/download/v6.0/FishBench.zip)
  * [Buildtester](http://software.farseer.org/): Latest release [Buildtester 1.4.7.0](http://software.farseer.org/Software/BuildTester.7z)
