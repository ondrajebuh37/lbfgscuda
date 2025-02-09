module l_bfgs_with_cuda
#Directly stolen from https://github.com/JuliaNLSolvers/Optim.jl/blob/master/src/multivariate/solvers/first_order/l_bfgs.jl
using CUDA
using NLSolversBase
using ForwardDiff
using FiniteDiff
using LinearAlgebra
using Optim: ZerothOrderOptimizer, FirstOrderOptimizer, SecondOrderOptimizer, @add_linesearch_fields, AbstractOptimizerState, common_trace!,
 @initial_linesearch, LineSearches, _alphaguess, Manifold, Flat, retract!, project_tangent!, Newton, AbstractOptimizer, Options, InplaceObjective, NotInplaceObjective,
 default_options, OptimizationTrace, ldiv!, rmul!, perform_linesearch!, ManifoldObjective, NewtonTrustRegion, assess_convergence, MultivariateOptimizationResults, x_abschange, x_relchange,
 pick_best_x, pick_best_f, f_abschange, f_relchange, g_residual#, value_gradient!


export LBFGS_CUDA, optimize_CUDA, GradientCache_CUDA, initial_state_CUDA, value_gradient_CUDA!, OnceDifferentiable_CUDA, is_finitediff, finitediff_fdtype


# Notational note
# JMW's dx_history <=> NW's S
# JMW's dg_history <=> NW's Y

# Here alpha is a cache that parallels betas
# It is not the step-size
# q is also a cache
function twoloop!(s,
                  gr,
                  rho,
                  dx_history,
                  dg_history,
                  m::Integer,
                  pseudo_iteration::Integer,
                  alpha,
                  q,
                  scaleinvH0::Bool,
                  precon)
    # Count number of parameters
    n = length(s)

    # Determine lower and upper bounds for loops
    lower = pseudo_iteration - m
    upper = pseudo_iteration - 1

    # Copy gr into q for backward pass
    copyto!(q, gr)
    # Backward pass
    for index in upper:-1:lower
        if index < 1
            continue
        end
        i   = mod1(index, m)
        dgi = dg_history[i]
        dxi = dx_history[i]
        @inbounds alpha[i] = rho[i] * real(dot(dxi, q))
        @inbounds q .-= alpha[i] .* dgi
    end

    # Copy q into s for forward pass
    if scaleinvH0 == true && pseudo_iteration > 1
        # Use the initial scaling guess from
        # Nocedal & Wright (2nd ed), Equation (7.20)

        #=
        pseudo_iteration > 1 prevents this scaling from happening
        at the first iteration, but also at the first step after
        a reset due to invH being non-positive definite (pseudo_iteration = 1).
        TODO: Maybe we can still use the scaling as long as iteration > 1?
        =#
        i = mod1(upper, m)
        dxi = dx_history[i]
        dgi = dg_history[i]
        scaling = real(dot(dxi, dgi)) / sum(abs2, dgi)
        @. s = scaling*q
    else
        # apply preconditioner if scaleinvH0 is false as the true setting
        # is essentially its own kind of preconditioning
        # (Note: preconditioner update was done outside of this function)
        ldiv!(s, precon, q)
    end
    # Forward pass
    for index in lower:1:upper
        if index < 1
            continue
        end
        i = mod1(index, m)
        dgi = dg_history[i]
        dxi = dx_history[i]
        @inbounds beta = rho[i] * real(dot(dgi, s))
        @inbounds s .+= dxi .* (alpha[i] - beta)
    end

    # Negate search direction
    rmul!(s, eltype(s)(-1))

    return
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



# Define T as a type parameter
struct LBFGS_CUDA{T, IL, L, Tprep} <: FirstOrderOptimizer
    m::Int
    alphaguess!::IL
    linesearch!::L
    P::T
    precondprep!::Tprep
    manifold::Manifold
    scaleinvH0::Bool
end

# Constructor for LBFGS
function LBFGS_CUDA(; m::Integer = 10,
                 alphaguess = LineSearches.InitialStatic(), # TODO: benchmark defaults
                 linesearch = LineSearches.HagerZhang(),  # TODO: benchmark defaults
                 P=nothing,
                 precondprep = (P, x) -> nothing,
                 manifold::Manifold=Flat(),
                 scaleinvH0::Bool = true && (typeof(P) <: Nothing) )
    LBFGS_CUDA(Int(m), _alphaguess(alphaguess), linesearch, P, precondprep, manifold, scaleinvH0)
end

Base.summary(::LBFGS_CUDA) = "L-BFGS-CUDA"

mutable struct LBFGSState{Tx, Tdx, Tdg, T, G} <: AbstractOptimizerState
    x::Tx
    x_previous::Tx
    g_previous::G
    rho::Vector{T}
    dx_history::Tdx
    dg_history::Tdg
    dx::Tx
    dg::Tx
    u::Tx
    f_x_previous::T
    twoloop_q
    twoloop_alpha
    pseudo_iteration::Int
    s::Tx
    @add_linesearch_fields()
end


# Fix T definition here for initial_state
function initial_state_CUDA(method::LBFGS_CUDA{F, IL, L, Tprep}, options, d, initial_x) where {F, IL, L, Tprep}
    # T is defined from initial_x
    T = real(eltype(initial_x))

    n = length(initial_x)
    initial_x = copy(initial_x)
    retract!(method.manifold, initial_x)
    value_gradient_CUDA!(d, initial_x)

    project_tangent!(method.manifold, gradient(d), initial_x)
    LBFGSState(initial_x, # Maintain current state in state.x
              copy(initial_x), # Maintain previous state in state.x_previous
              copy(gradient(d)), # Store previous gradient in state.g_previous
              fill(T(NaN), method.m), # state.rho
              [similar(initial_x) for i = 1:method.m], # Store changes in position in state.dx_history
              [eltype(gradient(d))(NaN).*gradient(d) for i = 1:method.m], # Store changes in position in state.dg_history
              T(NaN)*initial_x, # Buffer for new entry in state.dx_history
              T(NaN)*initial_x, # Buffer for new entry in state.dg_history
              T(NaN)*initial_x, # Buffer stored in state.u
              real(T)(NaN), # Store previous f in state.f_x_previous
              similar(initial_x), #Buffer for use by twoloop
              Vector{T}(undef, method.m), #Buffer for use by twoloop
              0,
              eltype(gradient(d))(NaN).*gradient(d), # Store current search direction in state.s
              @initial_linesearch()...)
end

function update_state!(d, state::LBFGSState, method::LBFGS_CUDA)
    n = length(state.x)
    # Increment the number of steps we've had to perform
    state.pseudo_iteration += 1

    project_tangent!(method.manifold, gradient(d), state.x)

    # update the preconditioner
    method.precondprep!(method.P, state.x)

    # Determine the L-BFGS search direction # FIXME just pass state and method?
    twoloop!(state.s, gradient(d), state.rho, state.dx_history, state.dg_history,
             method.m, state.pseudo_iteration,
             state.twoloop_alpha, state.twoloop_q, method.scaleinvH0, method.P)
    project_tangent!(method.manifold, state.s, state.x)

    # Save g value to prepare for update_g! call
    copyto!(state.g_previous, gradient(d))

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))

    # Update current position
    state.dx .= state.alpha .* state.s
    state.x .= state.x .+ state.dx
    retract!(method.manifold, state.x)

    lssuccess == false # break on linesearch error
end

function update_h!(d, state, method::LBFGS_CUDA)
    n = length(state.x)
    # Measure the change in the gradient
    state.dg .= gradient(d) .- state.g_previous

    # Update the L-BFGS history of positions and gradients
    rho_iteration = one(eltype(state.dx)) / real(dot(state.dx, state.dg))
    if isinf(rho_iteration)
        # TODO: Introduce a formal error? There was a warning here previously
        state.pseudo_iteration=0
        return true
    end
    idx = mod1(state.pseudo_iteration, method.m)
    state.dx_history[idx] .= state.dx
    state.dg_history[idx] .= state.dg
    state.rho[idx] = rho_iteration
    false
end

function trace!(tr, d, state, iteration, method::LBFGS_CUDA, options, curr_time=time())
  common_trace!(tr, d, state, iteration, method, options, curr_time)
end




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

function optimize_CUDA(d::D, initial_x::Tx, method::M,
                  options::Options{T, TCallback} = Options(;default_options(method)...),
                  state = initial_state_CUDA(method, options, d, initial_x)) where {D<:AbstractObjective, M<:AbstractOptimizer, Tx <: AbstractArray, T, TCallback}

    t0 = time() # Initial time stamp used to control early stopping by options.time_limit
    tr = OptimizationTrace{typeof(value(d)), typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.extended_trace || options.callback !== nothing
    stopped, stopped_by_callback, stopped_by_time_limit = false, false, false
    f_limit_reached, g_limit_reached, h_limit_reached = false, false, false
    x_converged, f_converged, f_increased, counter_f_tol = false, false, false, 0

    f_converged, g_converged = initial_convergence(d, state, method, initial_x, options)
    converged = f_converged || g_converged
    # prepare iteration counter (used to make "initial state" trace entry)
    iteration = 0

    options.show_trace && print_header(method)
    _time = time()
    trace!(tr, d, state, iteration, method, options, _time-t0)
    ls_success::Bool = true
    while !converged && !stopped && iteration < options.iterations
        iteration += 1
        ls_success = !update_state!(d, state, method)
        if !ls_success
            break # it returns true if it's forced by something in update! to stop (eg dx_dg == 0.0 in BFGS, or linesearch errors)
        end
        if !(method isa NewtonTrustRegion)
            update_g!(d, state, method) # TODO: Should this be `update_fg!`?
        end
        x_converged, f_converged,
        g_converged, f_increased = assess_convergence(state, d, options)
        # For some problems it may be useful to require `f_converged` to be hit multiple times
        # TODO: Do the same for x_tol?
        counter_f_tol = f_converged ? counter_f_tol+1 : 0
        converged = x_converged || g_converged || (counter_f_tol > options.successive_f_tol)
        if !(converged && method isa Newton) && !(method isa NewtonTrustRegion)
            update_h!(d, state, method) # only relevant if not converged
        end
        if tracing
            # update trace; callbacks can stop routine early by returning true
            stopped_by_callback = trace!(tr, d, state, iteration, method, options, time()-t0)
        end

        # Check time_limit; if none is provided it is NaN and the comparison
        # will always return false.
        _time = time()
        stopped_by_time_limit = _time-t0 > options.time_limit
        f_limit_reached = options.f_calls_limit > 0 && f_calls(d) >= options.f_calls_limit ? true : false
        g_limit_reached = options.g_calls_limit > 0 && g_calls(d) >= options.g_calls_limit ? true : false
        h_limit_reached = options.h_calls_limit > 0 && h_calls(d) >= options.h_calls_limit ? true : false

        if (f_increased && !options.allow_f_increases) || stopped_by_callback ||
            stopped_by_time_limit || f_limit_reached || g_limit_reached || h_limit_reached
            stopped = true
        end

        if method isa NewtonTrustRegion
            # If the trust region radius keeps on reducing we need to stop
            # because something is wrong. Wrong gradients or a non-differentiability
            # at the solution could be explanations.
            if state.delta ≤ method.delta_min
                stopped = true
            end
        end

        if g_calls(d) > 0 && !all(isfinite, gradient(d))
            options.show_warnings && @warn "Terminated early due to NaN in gradient."
            break
        end
        #TODO OnceDifferentiable_CUDA has no field h_calls. To je sice pravda, ale ani v originalnim OnceDifferentiable nebylo h_calls?
        # if h_calls(d) > 0 && !(d isa TwiceDifferentiableHV) && !all(isfinite, hessian(d))
        #     options.show_warnings && @warn "Terminated early due to NaN in Hessian."
        #     break
        # end
    end # while

    after_while!(d, state, method, options)

    # we can just check minimum, as we've earlier enforced same types/eltypes
    # in variables besides the option settings
    Tf = typeof(value(d))
    f_incr_pick = f_increased && !options.allow_f_increases
    stopped_by =(f_limit_reached=f_limit_reached,
                 g_limit_reached=g_limit_reached,
                 h_limit_reached=h_limit_reached,
                 time_limit=stopped_by_time_limit,
                 callback=stopped_by_callback,
                 f_increased=f_incr_pick)
    return MultivariateOptimizationResults{typeof(method),Tx,typeof(x_abschange(state)),Tf,typeof(tr), Bool, typeof(stopped_by)}(method,
                                        initial_x,
                                        pick_best_x(f_incr_pick, state),
                                        pick_best_f(f_incr_pick, state, d),
                                        iteration,
                                        iteration == options.iterations,
                                        x_converged,
                                        Tf(options.x_abstol),
                                        Tf(options.x_reltol),
                                        x_abschange(state),
                                        x_relchange(state),
                                        f_converged,
                                        Tf(options.f_abstol),
                                        Tf(options.f_reltol),
                                        f_abschange(d, state),
                                        f_relchange(d, state),
                                        g_converged,
                                        Tf(options.g_abstol),
                                        g_residual(d, state),
                                        f_increased,
                                        tr,
                                        f_calls(d),
                                        g_calls(d),
                                        g_calls(d), #Tady ma byt h_calls ale to neexistuje ani v originalni funkci :DDDD TODO
                                        ls_success,
                                        options.time_limit,
                                        _time-t0,
                                        stopped_by,
                                        )
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