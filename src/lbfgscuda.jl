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
using DataFrames
using CSV

#TODO rozdelit na dva soubory. Vsechny funkce krome toho co bylo v og lbfgs dat do utils
# include("utils.jl")
# using .utils
include("l_bfgs_with_cuda.jl")
using .l_bfgs_with_cuda


#M is single solution size, N is number of solutions wanted
function random_init(seed::Int, M::Int, min_r::T, max_r::F) where {T<:Number,F<:Number}
    # Random.seed!(seed)
    return rand(M).*(max_r.-min_r) .+ min_r
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
    minimizer = Optim.minimizer(res_gpu)
    minimum = Optim.minimum(res_gpu)
    #perhaps needed to convert res_gpu
    # as the output res_gpu minimizer is now 1D vector containing all N*M solutions where N=num_of_points, M=len_of_sol TODO 
    if verbose
        # println("Optimal x with CUDA: ", Optim.minimizer(res_gpu))
        println("Minimum f(x) with CUDA: ", minimum) 
    end
    return minimizer, minimum

end


# function compute_alt(f::Function, x0::CuArray, verbose::Bool)



function gaussian(x::AbstractVector, mu::Number, std::Number)
    (1/(std*sqrt(2*pi))).*exp.(-((x.-mu).^2)./(2*std^2))
end

compute_and_print(f, x0; verbose=false) = compute_and_print(f, x0, verbose)

#Define a function to optimize
function f(x::CUDA.CuArray{T}, given_height::F, gaus_mu::G, gaus_std::H) where {T<:Number,F<:Number,G<:Number,H<:Number}
    # Compute the sum of the absolute differences for all elements
    return sum((gaussian(x .^ 2, gaus_mu, gaus_std) .- given_height).^2)
end

function f(x::Array{T}, given_height::F, gaus_mu::G, gaus_std::H) where {T<:Number,F<:Number,G<:Number,H<:Number}
    # Compute the sum of the absolute differences for all elements
    return sum((gaussian(x .^ 2, gaus_mu, gaus_std) .- given_height).^2)
end



#quadratic functions -> not working? TODO 
function f_q(x::CUDA.CuArray{T}, given_height::F) where {T<:Number,F<:Number}
    # Compute the sum of the absolute differences for all elements
    return sum((x .^ 2 .- given_height).^2)
end

function f_q(x::Array{T}, given_height::F) where {T<:Number,F<:Number}
    # Compute the sum of the absolute differences for all elements
    return sum((x .^ 2 .- given_height).^2)
end

struct fun_sel
    gauss::Bool
end

function Base.show(io::IO, f_sel::fun_sel)
    if f_sel.gauss
        print(io, "Aeˣ")
    else
        print(io, "x²")
    end
end


########################################################################################################
#Run the base bfgs on this function
benchmarking = true
gauss = false #If gauss = true, obj. function is gauss. If gauss= false : objective func is quadratic.   (sum of squared diff*)
#This is for graphical purposes only
f_sel = fun_sel(gauss)

M = 2000 #solution size TODO FUNGUJE JEN PRO M=1 WTF
min_r, max_r = -10, 10 # min_range, max_range for the random initialization : TODO maybe better to change range on diff axis, now same for all
given_height = 1/2

gaus_mu = 2
gaus_std = 1

#TODO zeptat se -> tenhle postup asi neni uplne idealni a bylo by lepsi mozna pustit to cely M krat a ne to takhle cpat do ty funkce protoze je to moc tezky?
#Creating the actual function with only 1 input

#Sum of Gaussians
g = x -> f(x, given_height, gaus_mu, gaus_std)

#Sum of Quadratics
q = x -> f_q(x, given_height)

# x0 = random_init(69420, M, min_r, max_r) # Initial guess
# @btime compute_and_print(g, x0, verbose=true) #Prop 2000 trva 1 iterace cca minutu? TODO je tohle legit? Co mam odevzdat? Zpracovani vejsledku/grafy/tabulky atd kdyz to nejde pustit xd
#porovnani pro 2k samplu Minimum f(x) without CUDA: 183.8186949024613, Minimum f(x) with CUDA: 438.77129073357696  19.617 ms 
# print(x0)
# # compute_alt(g, x0, verbose=true) #TODO takhle ta loss?
# x0 = random_init(69420, M, min_r, max_r) # Initial guess
# print(CuArray(x0))
# @btime compute_and_print(g, CuArray(x0), verbose=true)

#Run the CUDA bfgs on this function
#TODO in the end there will be preprocessing for x0 needed during some actual testing.
# I randomly generate N initial solutions and append them into 1XN*M array, where M=len of one sol
#TODO jak pustit pres profview

#Compare the results

# Initialize DataFrame to store results
results = DataFrame(
    Num_Variables = Int[],
    CUDA = Bool[],
    Mean_t = Float64[],
    Min_t = Float64[],
    Min_Value = Float64[]
)



#Choose benchmarking function
if f_sel.gauss
    f_b = g
    num_of_points = Int64[1e2, 5e2, 1e3, 1500, 2000]# 1e4]#, 1e5, 1e6]
else
    f_b = q
    num_of_points = Int64[1e2, 1e3, 4e3, 7e3, 1e4]
end
#benchmarking BenchmarkTools? TODO proc to nevykresluje? Neni to problem s tim ze se mi to ani nekmpiluje? -> pkg se ukladaj asi do default v1.11 a ne do lbfgscuda env. Do REPLu using lbfgscuda
if benchmarking
    for m in num_of_points
        print("Testing ",f_sel," loss for: ", m, " variables --------> ")
        print(" Without CUDA ------->")
        x0 = random_init(69420, m, min_r, max_r) # Initial guess
        # @btime compute_and_print(f_b, $x0)
        # Benchmark and extract results
        #Its unreal for 10k+ samples, so here is just min=mean TODO ?
        elapsed_time = @elapsed compute_and_print(f_b, x0) 
        min_time = elapsed_time#minimum(stats.times)  # Minimum cycle time
        mean_time = elapsed_time#mean(stats.times)  # Mean cycle time
        min_sol, min_value = compute_and_print(f_b, x0)  # Compute min value (assuming it returns one)
        push!(results, (m, false, mean_time, min_time, min_value))

        print("With CUDA\n") #TODO bezi to vubec na cude?? :(
        x0_cuda = random_init(69420, m, min_r, max_r) # Initial guess
        # @btime compute_and_print(f_b, CuArray($x0))
        stats_cuda = @elapsed compute_and_print(f_b, x0_cuda)
        min_time_cuda = stats_cuda#minimum(stats_cuda.times)
        mean_time_cuda = stats_cuda#mean(stats_cuda.times)  # Mean cycle time

        min_sol_cuda, min_value_cuda = compute_and_print(f_b, x0_cuda)

        push!(results, (m, true, mean_time_cuda, min_time_cuda, min_value_cuda))
    end
    # Display results
    println(results)
    CSV.write("benchmark_results.csv", results)
end
#testcases can be done just by comparing LBFGS and cuda LBFGS results.

#TODO add to tests cca 3 functions and test on known results.

#If enough time, create demo where you do this with inverted gaussian. You first click points to sample quartic function and do MLE function f(x).

#run N times BFGS on random init solutions x0 and draw where they end. It will be feasible to visualise in 2d.

end




