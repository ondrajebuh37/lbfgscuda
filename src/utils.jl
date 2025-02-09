module utils
#TODO poresit using jak jednodusse rozlisit co je potreba a co ne
using CUDA
using NLSolversBase
using ForwardDiff
using FiniteDiff
using LinearAlgebra
using Optim: ZerothOrderOptimizer, FirstOrderOptimizer, SecondOrderOptimizer, @add_linesearch_fields, AbstractOptimizerState, common_trace!,
 @initial_linesearch, LineSearches, _alphaguess, Manifold, Flat, retract!, project_tangent!, Newton, AbstractOptimizer, Options, InplaceObjective, NotInplaceObjective,
 default_options, OptimizationTrace, ldiv!, rmul!, perform_linesearch!, ManifoldObjective, NewtonTrustRegion, assess_convergence, MultivariateOptimizationResults, x_abschange, x_relchange,
 pick_best_x, pick_best_f, f_abschange, f_relchange, g_residual#, value_gradient!

export  GradientCache_CUDA, value_gradient_CUDA!, OnceDifferentiable_CUDA, is_finitediff, finitediff_fdtype, optimize_CUDA

update_g!(d, state, method) = nothing
function update_g!(d, state, method::M) where M<:Union{FirstOrderOptimizer, Newton}
    # Update the function value and gradient
    value_gradient_CUDA!(d, state.x)
    if M <: FirstOrderOptimizer #only for methods that support manifold optimization
        project_tangent!(method.manifold, gradient(d), state.x)
    end
end
update_fg!(d, state, method) = nothing
update_fg!(d, state, method::ZerothOrderOptimizer) = value!(d, state.x)
function update_fg!(d, state, method::M) where M<:Union{FirstOrderOptimizer, Newton}
    value_gradient_CUDA!(d, state.x)
    if M <: FirstOrderOptimizer #only for methods that support manifold optimization
        project_tangent!(method.manifold, gradient(d), state.x)
    end
end

# Update the Hessian
update_h!(d, state, method) = nothing
update_h!(d, state, method::SecondOrderOptimizer) = hessian!(d, state.x)

after_while!(d, state, method, options) = nothing

function initial_convergence(d, state, method::AbstractOptimizer, initial_x, options)
    gradient!(d, initial_x)
    stopped = !isfinite(value(d)) || any(!isfinite, gradient(d))
    maximum(abs, gradient(d)) <= options.g_abstol, stopped
end
function initial_convergence(d, state, method::ZerothOrderOptimizer, initial_x, options)
    false, false
end

# Used for objectives and solvers where the gradient is available/exists
mutable struct OnceDifferentiable_CUDA{TF, TDF, TX} <: AbstractObjective
    f # objective
    df # (partial) derivative of objective
    fdf # objective and (partial) derivative of objective
    F::TF # cache for f output
    DF::TDF # cache for df output
    x_f::TX # x used to evaluate f (stored in F)
    x_df::TX # x used to evaluate df (stored in DF)
    f_calls::Vector{Int}
    # h_calls::Vector{Int} #TODO tohle jsem sem pridal protoze to chybi ale nikde jsem nenasel co to je? asi hesiany?
    df_calls::Vector{Int}
end

function alloc_DF(x::CUDA.CuArray{T}, F::T) where T
    # Example of allocating a CUDA array for the gradient, sized to match `x`
    # You can use `F` in some way depending on what you want to compute for `DF`
    return CUDA.zeros(T, length(x))
end
### Only the objective
# Ambiguity
OnceDifferentiable_CUDA(f, x::AbstractArray,
                   F::Real = real(zero(eltype(x))),
                   DF::AbstractArray = alloc_DF(x, F); inplace = true, autodiff = :finite,  
                   chunk::ForwardDiff.Chunk = ForwardDiff.Chunk(x)) =
    OnceDifferentiable_CUDA(f, x, F, DF, autodiff, chunk)
#OnceDifferentiable_CUDA(f, x::AbstractArray, F::AbstractArray; autodiff = :finite) =
#    OnceDifferentiable_CUDA(f, x::AbstractArray, F::AbstractArray, alloc_DF(x, F))
function OnceDifferentiable_CUDA(f, x::AbstractArray,
                   F::AbstractArray, DF::AbstractArray = alloc_DF(x, F);
                   inplace = true, autodiff = :finite)
    f! = f!_from_f(f, F, inplace)

    OnceDifferentiable_CUDA(f!, x::AbstractArray, F::AbstractArray, DF, autodiff)
end


function OnceDifferentiable_CUDA(f, x_seed::AbstractArray{T},
                            F::Real,
                            DF::AbstractArray,
                            autodiff, chunk) where T
    # When here, at the constructor with positional autodiff, it should already
    # be the case, that f is inplace.
    if  typeof(f) <: Union{InplaceObjective, NotInplaceObjective}

        fF = make_f(f, x_seed, F)
        dfF = make_df(f, x_seed, F)
        fdfF = make_fdf(f, x_seed, F)

        return OnceDifferentiable_CUDA(fF, dfF, fdfF, x_seed, F, DF)
    else
        if is_finitediff(autodiff)

            # Figure out which Val-type to use for FiniteDiff based on our
            # symbol interface.
            fdtype = finitediff_fdtype(autodiff)
            df_array_spec = DF
            x_array_spec = x_seed
            return_spec = typeof(F)
            gcache = GradientCache_CUDA(df_array_spec, x_array_spec, fdtype, return_spec)

            function g!(storage, x)
                finite_difference_gradient!(storage, f, x, gcache)
                return
            end
            function fg!(storage, x)
                g!(storage, x)
                return f(x)
            end
        elseif is_forwarddiff(autodiff)
            gcfg = ForwardDiff.GradientConfig(f, x_seed, chunk)
            g! = (out, x) -> ForwardDiff.gradient!(out, f, x, gcfg)

            fg! = (out, x) -> begin
                gr_res = DiffResults.DiffResult(zero(T), out)
                ForwardDiff.gradient!(gr_res, f, x, gcfg)
                DiffResults.value(gr_res)
            end
        else
            error("The autodiff value $autodiff is not support. Use :finite or :forward.")
        end

        return OnceDifferentiable_CUDA(f, g!, fg!, x_seed, F, DF)
    end
end

has_not_dep_symbol_in_ad = Ref{Bool}(true)
OnceDifferentiable_CUDA(f, x::AbstractArray, F::AbstractArray, autodiff::Symbol, chunk::ForwardDiff.Chunk = ForwardDiff.Chunk(x)) =
OnceDifferentiable_CUDA(f, x, F, alloc_DF(x, F), autodiff, chunk)
function OnceDifferentiable_CUDA(f, x::AbstractArray, F::AbstractArray,
                            autodiff::Bool, chunk::ForwardDiff.Chunk = ForwardDiff.Chunk(x))
    if autodiff == false
        throw(ErrorException("It is not possible to set the `autodiff` keyword to `false` when constructing a OnceDifferentiable_CUDA instance from only one function. Pass in the (partial) derivative or specify a valid `autodiff` symbol."))
    elseif has_not_dep_symbol_in_ad[]
        @warn("Setting the `autodiff` keyword to `true` is deprecated. Please use a valid symbol instead.")
        has_not_dep_symbol_in_ad[] = false
    end
    OnceDifferentiable_CUDA(f, x, F, alloc_DF(x, F), :forward, chunk)
end
function OnceDifferentiable_CUDA(f, x_seed::AbstractArray, F::AbstractArray, DF::AbstractArray,
    autodiff::Symbol , chunk::ForwardDiff.Chunk = ForwardDiff.Chunk(x_seed))
    if  typeof(f) <: Union{InplaceObjective, NotInplaceObjective}
        fF = make_f(f, x_seed, F)
        dfF = make_df(f, x_seed, F)
        fdfF = make_fdf(f, x_seed, F)
        return OnceDifferentiable_CUDA(fF, dfF, fdfF, x_seed, F, DF)
    else
        if is_finitediff(autodiff)
            # Figure out which Val-type to use for FiniteDiff based on our
            # symbol interface.
            fdtype = finitediff_fdtype(autodiff)
            # Apparently only the third input is aliased.
            j_finitediff_cache = FiniteDiff.JacobianCache(copy(x_seed), copy(F), copy(F), fdtype)
            if autodiff == :finiteforward
                # These copies can be done away with if we add a keyword for
                # reusing arrays instead for overwriting them.
                Fx = copy(F)
                DF = copy(DF)

                x_f, x_df = x_of_nans(x_seed), x_of_nans(x_seed)
                f_calls, j_calls = [0,], [0,]
                function j_finiteforward!(J, x)
                    # Exploit the possibility that it might be that x_f == x
                    # then we don't have to call f again.

                    # if at least one element of x_f is different from x, update
                    if any(x_f .!= x)
                        f(Fx, x)
                        f_calls .+= 1
                    end

                    FiniteDiff.finite_difference_jacobian!(J, f, x, j_finitediff_cache, Fx)
                end
                function fj_finiteforward!(F, J, x)
                    f(F, x)
                    FiniteDiff.finite_difference_jacobian!(J, f, x, j_finitediff_cache, F)
                end


                return OnceDifferentiable_CUDA(f, j_finiteforward!, fj_finiteforward!, Fx, DF, x_f, x_df, f_calls, j_calls)
            end

            function fj_finitediff!(F, J, x)
                f(F, x)
                FiniteDiff.finite_difference_jacobian!(J, f, x, j_finitediff_cache)
                F
            end
            function j_finitediff!(J, x)
                F_cache = copy(F)
                fj_finitediff!(F_cache, J, x)
            end

            return OnceDifferentiable_CUDA(f, j_finitediff!, fj_finitediff!, x_seed, F, DF)

        elseif is_forwarddiff(autodiff)

            jac_cfg = ForwardDiff.JacobianConfig(f, F, x_seed, chunk)
            ForwardDiff.checktag(jac_cfg, f, x_seed)

            F2 = copy(F)
            function j_forwarddiff!(J, x)
                ForwardDiff.jacobian!(J, f, F2, x, jac_cfg, Val{false}())
            end
            function fj_forwarddiff!(F, J, x)
                jac_res = DiffResults.DiffResult(F, J)
                ForwardDiff.jacobian!(jac_res, f, F2, x, jac_cfg, Val{false}())
                DiffResults.value(jac_res)
            end

            return OnceDifferentiable_CUDA(f, j_forwarddiff!, fj_forwarddiff!, x_seed, F, DF)
        else
            error("The autodiff value $(autodiff) is not supported. Use :finite or :forward.")
        end
    end
end

### Objective and derivative
function OnceDifferentiable_CUDA(f, df,
                   x::AbstractArray,
                   F::Real = real(zero(eltype(x))),
                   DF::AbstractArray = alloc_DF(x, F);
                   inplace = true)


    df! = df!_from_df(df, F, inplace)

    fdf! = make_fdf(x, F, f, df!)

    OnceDifferentiable_CUDA(f, df!, fdf!, x, F, DF)
end

function OnceDifferentiable_CUDA(f, j,
                   x::AbstractArray,
                   F::AbstractArray,
                   J::AbstractArray = alloc_DF(x, F);
                   inplace = true)

    f! = f!_from_f(f, F, inplace)
    j! = df!_from_df(j, F, inplace)
    fj! = make_fdf(x, F, f!, j!)

    OnceDifferentiable_CUDA(f!, j!, fj!, x, F, J)
end


### Objective, derivative and combination
function OnceDifferentiable_CUDA(f, df, fdf,
    x::AbstractArray,
    F::Real = real(zero(eltype(x))),
    DF::AbstractArray = alloc_DF(x, F);
    inplace = true)

    # f is never "inplace" since F is scalar
    df! = df!_from_df(df, F, inplace)
    fdf! = fdf!_from_fdf(fdf, F, inplace)

    x_f, x_df = x_of_nans(x), x_of_nans(x)

    OnceDifferentiable_CUDA{typeof(F),typeof(DF),typeof(x)}(f, df!, fdf!,
    copy(F), copy(DF),
    x_f, x_df,
    [0,], [0,])
end

function OnceDifferentiable_CUDA(f, df, fdf,
                            x::AbstractArray,
                            F::AbstractArray,
                            DF::AbstractArray = alloc_DF(x, F);
                            inplace = true)

    f = f!_from_f(f, F, inplace)
    df! = df!_from_df(df, F, inplace)
    fdf! = fdf!_from_fdf(fdf, F, inplace)

    x_f, x_df = x_of_nans(x), x_of_nans(x)

    OnceDifferentiable_CUDA(f, df!, fdf!, copy(F), copy(DF), x_f, x_df, [0,], [0,])
end

function is_finitediff(autodiff)
    return autodiff == :finite
end

function forward_difference(f, x, h=1e-6)
    return (f(x + h) - f(x)) / h
end

function central_difference(f, x, h=1e-6)
    return (f(x + h) - f(x - h)) / (2 * h)
end

# Updated finitediff_fdtype function
function finitediff_fdtype(autodiff::Symbol)
    if autodiff == :finite
        return Val(:central)
    elseif autodiff == :forward
        return Val(:forward)
    else
        error("Unsupported autodiff type: $autodiff")
    end
end


#These were taken from NLSolversBase.jl/src/objective_types/inplace_factory.jl and for some reason simple using didnt work TODO

function fdf!_from_fdf(fg, F::Real, inplace)
    if inplace
        return fg
    else
        return function ffgg!(G, x)
            fx, gx = fg(x)
            copyto!(G, gx)
            fx
        end
    end
end
function fdf!_from_fdf(fj, F::AbstractArray, inplace)
    if inplace
        return fj
    else
        return function ffjj!(F, J, x)
            fx, jx = fj(x)
            copyto!(J, jx)
            copyto!(F, fx)
        end
    end
end

function df!_from_df(g, F::Real, inplace)
    if inplace
        return g
    else
        return function gg!(G, x)
            copyto!(G, g(x))
            G
        end
    end
end
function df!_from_df(j, F::AbstractArray, inplace)
    if inplace
        return j
    else
        return function jj!(J, x)
            copyto!(J, j(x))
            J
        end
    end
end
#same for this except src/NLSolversBase.jl
x_of_nans(x, Tf=eltype(x)) = fill!(Tf.(x), Tf(NaN))


#TODO co to any tady >:(
# function default_relstep(::Any, ::Any)
#     # Handle Float32 cases
#     return 1e-6
# end

struct GradientCache_CUDA{CacheType1,CacheType2,CacheType3,CacheType4,fdtype,returntype,inplace}
    fx::CacheType1
    c1::CacheType2
    c2::CacheType3
    c3::CacheType4
end
function finite_difference_gradient!(
    df::StridedVector{<:Number},
    f,
    x::StridedVector{<:Number},
    cache::GradientCache_CUDA{T1,T2,T3,T4,fdtype,returntype,inplace};
    relstep=1e-6,#default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    dir=true) where {T1,T2,T3,T4,fdtype,returntype,inplace}

    fx, c1, c2, c3 = cache.fx, cache.c1, cache.c2, cache.c3
    if fdtype != Val(:complex) && eltype(df) <: Complex && !(eltype(x) <: Complex)
        copyto!(c1, x)
    end
    copyto!(c3, x)

    if fdtype == Val(:forward)
        epsilons = compute_epsilon.(fdtype, x, relstep, absstep, dir)
        x_old = copy(x)
        c3 .= x .+ epsilons
        dfi = (f(c3) .- fx) ./ epsilons
        df .= real.(dfi)
        c3 .= x_old
        
        if eltype(df) <: Complex
            if eltype(x) <: Complex
                c3 .= x .+ im .* epsilons
                dfi = (f(c1) .- fx) ./ (im .* epsilons)
            else
                c1 .= x .+ im .* epsilons
                dfi = (f(c1) .- fx) ./ (im .* epsilons)
            end
            df .-= im .* imag.(dfi)
        end
    
    elseif fdtype == Val(:central)
        epsilons = compute_epsilon.(fdtype, x, relstep, absstep, dir)
        x_old = copy(x)
        c3 .= x .+ epsilons
        dfi = f(c3)
        c3 .= x .- epsilons
        #funkce je asi (val, grad)?TODO #druhy update - uz jsem to snad opravil ale check
        dfi -= f(c3)
        df .= real.(dfi ./ (2 .* epsilons))
        c3 .= x_old
        
        if eltype(df) <: Complex
            if eltype(x) <: Complex
                c3 .= x .+ im .* epsilons
                dfi = f(c3)
                c3 .= x .- im .* epsilons
                dfi .-= f(c3)
            else
                c1 .= x .+ im .* epsilons
                dfi = f(c1)
                c1 .= x .- im .* epsilons
                dfi .-= f(c1)
            end
            df .-= im .* imag.(dfi ./ (2 .* im .* epsilons))
        end
    
    elseif fdtype == Val(:complex) && returntype <: Real && eltype(df) <: Real && eltype(x) <: Real
        copyto!(c1, x)
        epsilon_complex = eps(real(eltype(x)))
        c1_old = copy(c1)
        c1 .= x .+ im * epsilon_complex
        df .= imag.(f(c1)) ./ epsilon_complex
        c1 .= c1_old
    else
        fdtype_error(returntype)
    end
    
    df
end
# Compute gradient using finite differences on the GPU TODO Any - dal jsem tam D protoze to uz nekde bylo na stejnym miste o funkci vys
function value_gradient_CUDA!(d::D, initial_x::CUDA.CuArray{T}) where {D,T}
    ff = d.f  # The function itself
    epsilon = eps(T) * 1000  # A small perturbation for finite differences
    # dd = CUDA.zeros(T, length(initial_x))  # Gradient output array on the GPU

    # Compute the value of the function at initial_x
    fx = ff(initial_x)

    # Compute the gradient using finite differences
    perturbed_x = initial_x .+ epsilon  # Perturb the array without scalar indexing

    # Calculate the gradient: (perturbed - original) / epsilon
    df = (ff(perturbed_x) .- fx) ./ epsilon  # Element-wise operation

    # Store the gradient result on the GPU
    # CUDA.copyto!(dd, df)  # Ensure this operation happens on the GPU

    return df
end




"""
    FiniteDiff.GradientCache(
        df         :: Union{<:Number,AbstractArray{<:Number}},
        x          :: Union{<:Number, AbstractArray{<:Number}},
        fdtype     :: Type{T1} = Val{:central},
        returntype :: Type{T2} = eltype(df),
        inplace    :: Type{Val{T3}} = Val{true})

Allocating Cache Constructor
"""
function GradientCache_CUDA(
    df,
    x,
    fdtype=Val(:central),
    returntype=eltype(df),
    inplace=Val(true))

    fdtype isa Type && (fdtype = fdtype())
    inplace isa Type && (inplace = inplace())
    if typeof(x) <: AbstractArray # the vector->scalar case
        if fdtype != Val(:complex) # complex-mode FD only needs one cache, for x+eps*im
            if typeof(x) <: StridedVector
                if eltype(df) <: Complex && !(eltype(x) <: Complex)
                    _c1 = zero(Complex{eltype(x)}) .* x
                    _c2 = nothing
                else
                    _c1 = nothing
                    _c2 = nothing
                end
            else
                _c1 = zero(x)
                _c2 = zero(real(eltype(x))) .* x
            end
        else
            if !(returntype <: Real)
                fdtype_error(returntype)
            else
                _c1 = x .+ zero(eltype(x)) .* im
                _c2 = nothing
            end
        end
        _c3 = zero(x)
    else # the scalar->vector case
        # need cache arrays for fx1 and fx2, except in complex mode, which needs one complex array
        if fdtype != Val(:complex)
            _c1 = zero(df)
            _c2 = zero(df)
        else
            _c1 = zero(Complex{eltype(x)}) .* df
            _c2 = nothing
        end
        _c3 = x
    end

    GradientCache_CUDA{Nothing,typeof(_c1),typeof(_c2),typeof(_c3),fdtype,
        returntype,inplace}(nothing, _c1, _c2, _c3)

end

"""
    FiniteDiff.GradientCache(
        fx         :: Union{Nothing,<:Number,AbstractArray{<:Number}},
        c1         :: Union{Nothing,AbstractArray{<:Number}},
        c2         :: Union{Nothing,AbstractArray{<:Number}},
        c3         :: Union{Nothing,AbstractArray{<:Number}},
        fdtype     :: Type{T1} = Val{:central},
        returntype :: Type{T2} = eltype(fx),
        inplace    :: Type{Val{T3}} = Val{true})

Non-Allocating Cache Constructor

# Arguments 
- `fx`: Cached function call.
- `c1`, `c2`, `c3`: (Non-aliased) caches for the input vector.
- `fdtype = Val(:central)`: Method for cmoputing the finite difference.
- `returntype = eltype(fx)`: Element type for the returned function value.
- `inplace = Val(false)`: Whether the function is computed in-place or not.

# Output 
The output is a [`GradientCache`](@ref) struct.

```julia
julia> x = [1.0, 3.0]
2-element Vector{Float64}:
 1.0
 3.0

julia> _f = x -> x[1] + x[2]
#13 (generic function with 1 method)

julia> fx = _f(x)
4.0

julia> gradcache = GradientCache(copy(x), copy(x), copy(x), fx)
GradientCache{Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}, Val{:central}(), Float64, Val{false}()}(4.0, [1.0, 3.0], [1.0, 3.0], [1.0, 3.0])
```
"""
function GradientCache_CUDA(
    fx::Fx,# match order in struct for Setfield
    c1::T,
    c2::T,
    c3::T,
    fdtype=Val(:central),
    returntype=eltype(fx),
    inplace=Val(true)) where {T,Fx} # Val(false) isn't so important for vector -> scalar, it gets ignored in that case anyway.
    GradientCache_CUDA{Fx,T,T,T,fdtype,returntype,inplace}(fx, c1, c2, c3)
end


#From FiniteDiff
@inline function compute_epsilon(::Val{:forward}, x::T, relstep::Real, absstep::Real, dir::Real) where T<:Number
    return max(relstep*abs(x), absstep)*dir
end
@inline function compute_epsilon(::Val{:central}, x::T, relstep::Real, absstep::Real, dir=nothing) where T<:Number
    return max(relstep*abs(x), absstep)
end

end