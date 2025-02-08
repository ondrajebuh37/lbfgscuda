module lbfgscuda
using Random
using BenchmarkTools
using CUDA
using Optim

include("l_bfgs_with_cuda.jl")


function random_init(seed::Int)
    Random.seed!(seed)
    return [rand()]
end

# CPU Variant
function compute_and_print(f::Function, x0::Array)
    res_cpu = optimize(f, x0, LBFGS())
    minimizer = Optim.minimizer(res_cpu)
    minimum = Optim.minimum(res_cpu)
    println("Optimal x without CUDA: ", minimizer)
    println("Minimum f(x) without CUDA: ", minimum) 
    return minimizer, minimum
end

# GPU Variant
function compute_and_print(f::Function, x0::CuArray)
    res_gpu = optimize(f, x0, LBFGS_CUDA())
    #perhaps needed to convert res_gpu
    # as the output res_gpu minimizer is now 1D vector containing all N*M solutions where N=num_of_points, M=len_of_sol TODO
    println("Optimal x with CUDA: ", Optim.minimizer(res_gpu))
    println("Minimum f(x) with CUDA: ", Optim.minimum(res_gpu)) 
end

#Define a function to optimize

given_height = 9
f(x) = abs((x[1])^2 - given_height)  # Objective function : here quadratic function f(x) = x^2

#Run the base bfgs on this function
x0 = random_init(69420) # Initial guess
compute_and_print(f, x0)
#Run the CUDA bfgs on this function
#TODO in the end there will be preprocessing for x0 needed during some actual testing.
# I randomly generate N initial solutions and append them into 1XN*M array, where M=len of one sol
compute_and_print(f, CuArray(x0))

#Compare the results


#benchmarking BenchmarkTools? TODO

#testcases can be done just by comparing LBFGS and cuda LBFGS results.

#TODO add to tests cca 3 functions and test on known results.

#If enough time, create demo where you do this with inverted gaussian. You first click points to sample quartic function and do MLE function f(x).

#run N times BFGS on random init solutions x0 and draw where they end. It will be feasible to visualise in 2d.
end
