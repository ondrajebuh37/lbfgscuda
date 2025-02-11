# lbfgscuda [![Build Status](https://github.com/ondrajebuh37/lbfgscuda/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/ondrajebuh37/lbfgscuda/actions/workflows/CI.yml?query=branch%3Amaster)



################


The main runnable script is lbfgscuda.jl
The CUDA enhanced version of L-BFGS is l_bfgs_with_cuda.jl
TODO je spravne CUDA_?
TODO proc sou tak random importy
TODO spousta funkci co mam v lbfgswithcuda/utils tak by tam asi nemusela byt a pridal jsem to jen protoze se to nedalo importovat a nemel sem cas -> (na aspon jednom celem pusteni jsem pracoval 3 dny)
nejcastejsi issue : scalar indexing -> fix broadcasting, neexistujici multiple dispatch -> doimplementovat(obcas jsem to nahradil ::Any docasne a pridal TODO), missing importy -> Zjistit z jakyho pkg chybi a doplnit





TODO vsechno nad timhle smazat, real dokumentace je dole:::::::

This package should provide a GPU-friendly L-BFGS implementation.
All of the functional parts of L-BFGS were taken from some already existing Numerical Julia libraries and are referenced to during their definition.

The base L-BFGS was directly stolen from https://github.com/JuliaNLSolvers/Optim.jl/blob/master/src/multivariate/solvers/first_order/l_bfgs.jl
The modified version is in l_bfgs_with_cuda.jl.

In utils.jl there should be the rest of the functions, which are also taken from existing Numerical Julia libraries and are referenced to during their definition.

You can test run your script with either /scripts/run_simple_quadratic.jl or /scripts/run_simple_gaussian.jl

To obtain some basic benchmarking, run /scripts/simple_bench.jl
