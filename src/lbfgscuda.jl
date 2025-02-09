module lbfgscuda
using Random
using BenchmarkTools
using CUDA
#TODO jak vyresit ty importy? mam vsechno strasne random
using Optim
using Optim: FirstOrderOptimizer, Newton, ZerothOrderOptimizer, SecondOrderOptimizer, AbstractOptimizer, 
AbstractObjective, Options, default_options, OptimizationTrace, InplaceObjective, NotInplaceObjective
using NLSolversBase
using ForwardDiff
using FiniteDiff
using BenchmarkTools
using ProfileView

include("l_bfgs_with_cuda.jl")
using .l_bfgs_with_cuda


function random_init(seed::Int, M::Int)
    Random.seed!(seed)
    return rand(M)
end

# CPU Variant
function compute_and_print(f::Function, x0::Array,verbose::Bool)
    res_cpu = optimize(f, x0, LBFGS())
    minimizer = Optim.minimizer(res_cpu)
    minimum = Optim.minimum(res_cpu)
    if verbose
        # println("Optimal x without CUDA: ", minimizer)
        println("Minimum f(x) without CUDA: ", minimum) 
    end
        return minimizer, minimum
end

# GPU Variant


function compute_and_print(f::Function, x0::CuArray, verbose::Bool)
    d = OnceDifferentiable_CUDA(f, x0)
    method = LBFGS_CUDA()
    # @show typeof(method)
    res_gpu = optimize_CUDA(d, x0, method)
    #perhaps needed to convert res_gpu
    # as the output res_gpu minimizer is now 1D vector containing all N*M solutions where N=num_of_points, M=len_of_sol TODO 
    if verbose
        # println("Optimal x with CUDA: ", Optim.minimizer(res_gpu))
        println("Minimum f(x) with CUDA: ", Optim.minimum(res_gpu)) 
    end
end

compute_and_print(f, x0; verbose=false) = compute_and_print(f, x0, verbose)

#Define a function to optimize
function f(x::CUDA.CuArray{T}) where T
    # Compute the sum of the absolute differences for all elements
    given_height = 9
    return sum(abs.(x .^ 2 .- given_height))
end

function f(x::Array{T}) where T
    # Compute the sum of the absolute differences for all elements
    given_height = 9
    return sum(abs.(x .^ 2 .- given_height))
end

#Run the base bfgs on this function
benchmarking = false
M = 100
x0 = random_init(69420, M) # Initial guess
compute_and_print(f, x0, verbose=true)

x0 = random_init(69420, M) # Initial guess
compute_and_print(f, CuArray(x0), verbose=true)


#Run the CUDA bfgs on this function
#TODO in the end there will be preprocessing for x0 needed during some actual testing.
# I randomly generate N initial solutions and append them into 1XN*M array, where M=len of one sol
#TODO jak pustit pres profview

#Compare the results


#benchmarking BenchmarkTools? TODO proc to nevykresluje? Neni to problem s tim ze se mi to ani nekmpiluje? -> pkg se ukladaj asi do default v1.11 a ne do lbfgscuda env. Do REPLu using lbfgscuda
if benchmarking
    num_of_points = Int64[1e3, 1e4, 1e5, 1e6]
    for m in num_of_points
        print("Testing quadratic SAD loss for: ", m, " variables\n")
        print("Without CUDA")
        x0 = random_init(69420, m) # Initial guess
        # @benchmark compute_and_print(f, x0)
        @btime compute_and_print(f, x0)

        print("With CUDA")

        x0 = random_init(69420, m) # Initial guess
        # @benchmark compute_and_print(f, CuArray(x0))
        @btime compute_and_print(f, CuArray(x0))
    end
end
#testcases can be done just by comparing LBFGS and cuda LBFGS results.

#TODO add to tests cca 3 functions and test on known results.

#If enough time, create demo where you do this with inverted gaussian. You first click points to sample quartic function and do MLE function f(x).

#run N times BFGS on random init solutions x0 and draw where they end. It will be feasible to visualise in 2d.
end





