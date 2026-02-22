## Wrap solvers so they can all be called with the same inputs
using Base: with_logger
using Logging,LinearAlgebra, Printf, Statistics, Random
using WENDy
using Logging: LogLevel, Info, Warn, ConsoleLogger
using WENDy: OELS, WLS, IRLS, TrustRegion
using WENDy: SecondOrderCostFunction, LeastSquaresCostFunction
import WENDy: WENDyProblem, WENDyParameters
## external deps 
using Distributions: Normal, LogNormal, Distribution, Uniform, rand
using OrdinaryDiffEq: ODEProblem, OrdinaryDiffEqAlgorithm, Rosenbrock23, solve as solve_ode
using JLD2
## define all odes 
includet("wendyData.jl")
includet("odes/logisticGrowth.jl")
includet("odes/lorenz.jl")
includet("odes/hindmarshRose.jl")
includet("odes/goodwin.jl")
includet("odes/sir.jl")
##
function _fmt_allocations(a::Real)
    _exp_dict = Dict(
        0=>"",
        1=>"K",
        2=>"Mp1",
        3=>"B",
        4=>"T"
    )
    if a == 0 
        return "$a allocations"
    end
    e = log10(a)
    k = Int(floor(e / 3))
    suffix = _exp_dict[k]
    str = @sprintf "%.4g" (a * 10.0^-(k*3))
    str*" $suffix allocations"
end
##
@kwdef mutable struct AlgoRes 
    what::AbstractVector{<:Real} = zeros(0)
    cov::AbstractMatrix{<:Real}  = zeros(0,0)
    wits::AbstractMatrix{<:Real} = zeros(0,0)
    cl2::Real                    = 0.0
    mean_fsl2::Real              = 0.0
    final_fsl2::Real             = 0.0
    fsnll::Real                  = 0.0
    wl2::Real                    = 0.0
    wnll::Real                   = 0.0
    dt::Real                     = 0.0
    iters::Int                   = 0
end
##
function computeMetrics!(
    res::AlgoRes, 
    wendyProb::WENDyProblem, 
    ex::SimulatedWENDyData,
    alg_dt::Real, 
    what::AbstractVector{<:Real}, 
    iters::Int, 
    wits::AbstractMatrix{<:Real};
    ll::LogLevel=Info
)   
    computeMetrics!(
        res, wendyProb, 
        ex.f!, ex.tt[], ex.Ustar[], ex.U[], ex.sigTrue[], ex.wTrue, ex.params, 
        alg_dt, what, iters, wits; ll=ll
    ) 
end

function computeMetrics!(
    res::AlgoRes, 
    wendyProb::WENDyProblem, 
    f!::Function, 
    tt::AbstractVector{<:Real}, 
    Ustar::AbstractMatrix{<:Real}, 
    U::AbstractMatrix{<:Real}, 
    sig::AbstractVector{<:Real}, 
    wTrue::AbstractVector{<:Real},
    params::WENDyParameters,
    alg_dt::Real, 
    what::AbstractVector{<:Real}, 
    iters::Int, 
    wits::AbstractMatrix{<:Real};
    ll::LogLevel=Info
)   
    # Extract data from the example
    
    Mp1   = length(tt)
    J     = length(wTrue)
    # transfer metrics to the AlgoRes struct
    res.dt    = alg_dt
    res.iters = iters
    res.what  = what
    res.wits  = wits
    # use a linearization about what to estimate the covariance of the parameters
    res.cov = try
        S = WENDy.Covariance(wendyProb.data, params)
        ∇r = WENDy.JacobianResidual(wendyProb.data, params)
        Ghat = ∇r(what)  
        Shat = S(what; transpose=false, doChol=false)
        (Ghat \ Shat) / (Ghat')
    catch
        NaN*ones(J,J)
    end
    # relative coefficient l2 error
    res.cl2 = try 
        norm(what[1:J] - wTrue) / norm(wTrue)
    catch 
        NaN 
    end
    # weak form relative l2 error
    res.wl2 = try 
        _res = zeros(wendyProb.wlsq.KD)
        wendyProb.wlsq.r!(_res, what[1:J])
        1/2 * norm(_res)^2
    catch 
        NaN 
    end
    # Forward solve metrics
    res.mean_fsl2, res.final_fsl2, res.fsnll = try
        u0star = Ustar[1,:]
        tRng = (tt[1], tt[end])
        dt = tt[2]- tt[1] # mean(diff(tt))
        odeprob = ODEProblem(f!, u0star, tRng, what)
        sol = solve_ode(odeprob, params.fsAlg; 
            reltol=params.fsReltol, abstol=params.fsAbstol, 
            saveat=dt, verbose=false
        )
        Uhat = reduce(vcat, um' for um in sol.u)
        _R = Uhat - Ustar
        mean_fsl2 = norm(_R)/norm(Ustar)
        final_fsl2 = norm(_R[end,:])/norm(Ustar[end,:])
        fsnll = sum(
            1/2*log(2*pi) 
            + J/2 * sum(log.(sig)) 
            + 1/2*dot(Uhat[m,:] - U[m,:], diagm(1 ./ sig), Uhat[m,:] - U[m,:])
            for m in 1:Mp1
        )
        mean_fsl2, final_fsl2, fsnll
    catch 
        NaN, NaN, NaN
    end
    # Weak form negative log-likelihood
    res.wnll  = try
        wendyProb.wnll.f(what[1:J])
    catch 
        NaN 
    end
    with_logger(ConsoleLogger(stdout, ll)) do 
        @info """      dt         = $(@sprintf "%.4g" res.dt)s
                    cl2        = $(@sprintf "%.4g" res.cl2*100)%
                    mean_fsl2  = $(@sprintf "%.4g" res.mean_fsl2*100)%
                    final_fsl2 = $(@sprintf "%.4g" res.final_fsl2*100)%
                    fsnll      = $(@sprintf "%.4g" res.fsnll)
                    wl2        = $(@sprintf "%.4g" res.wl2)
                    wnll       = $(@sprintf "%.4g" res.wnll)
                    iters      = $iters
        """
    end
    nothing
end
##
function runExample(ex, simParams,
    algorithms=(oels=OELS(),wls=WLS(),wendy_irls=IRLS(),wendy_mle=TrustRegion());
    ll::LogLevel=Info, w0=nothing, wendyProb=nothing,
    kwargs...
)
    with_logger(ConsoleLogger(stdout,ll)) do
        @info "========================================================="
        @info "Running $(ex.name)" 
        if isnothing(wendyProb) # Run with a pre build wendyprob (loaded from a mega job probably)
            @info "  Initializing problem by subsampling and adding noise"
            simulate!(ex, simParams; ll=ll)
            @info "  Building wendyProb"
            wendyProb = WENDyProblem(ex; ll=ll);
        end
        ## Pick a initialization point
        algoNames = vcat(:init, :truth, keys(algorithms)...)
        J = wendyProb.J
        if isnothing(w0)
            w0 = zeros(J)
            if !isnothing(simParams.seed)
                Random.seed!(simParams.seed)
            end
            for j in 1:J  
                d = Uniform(ex.wRng[j]...)
                w0[j] = rand(d)
            end
        end
        @info "    w0 = $(w0')"
        ## run cost functions once so that it does not effect timing 
        @info "    Running Cost functions once for JIT"
        K, D = wendyProb.K, wendyProb.D
        wTest = ex.wTrue
        wnllGradient = zeros(J)
        wnllHessian = zeros(J,J)
        @info "     wnll.f ..."
        dt = @elapsed a = @allocations wendyProb.wnll.f(wTest)
        @info "      $(@sprintf "%.4g" dt ) s, $(_fmt_allocations(a))"
        @info "     wnll.∇f! ..."
        dt = @elapsed a = @allocations wendyProb.wnll.∇f!(wnllGradient, wTest)
        @info "      $(@sprintf "%.4g" dt ) s, $(_fmt_allocations(a))"
        @info "     wnll.Hf! ..."
        dt = @elapsed a = @allocations wendyProb.wnll.Hf!(wnllHessian, wTest)
        @info "      $(@sprintf "%.4g" dt ) s, $(_fmt_allocations(a))"
        weakResidual = zeros(K*D)
        weakJacobian = zeros(K*D,J)
        @info "     wlsq.r! ..."
        dt = @elapsed a = @allocations wendyProb.wlsq.r!(weakResidual, wTest)
        @info "      $(@sprintf "%.4g" dt ) s, $(_fmt_allocations(a))"
        @info "     wlsq.∇r! ..."
        dt = @elapsed a = @allocations wendyProb.wlsq.∇r!(weakJacobian, wTest)
        @info "      $(@sprintf "%.4g" dt ) s, $(_fmt_allocations(a))"
        _Mp1,_ = size(ex.U[])
        forwardResidual = zeros(_Mp1*D)
        forwardJacobian = zeros(_Mp1*D,J+D)
        @info "     fslsq.r! ..."
        dt = @elapsed a = @allocations wendyProb.oels.r!(forwardResidual, vcat(wTest, ex.U[][1,:]))
        @info "      $(@sprintf "%.4g" dt ) s, $(_fmt_allocations(a))"
        @info "     fslsq.∇r! ..."
        dt = @elapsed a = @allocations wendyProb.oels.∇r!(forwardJacobian, vcat(wTest, ex.U[][1,:]))
        @info "      $(@sprintf "%.4g" dt ) s, $(_fmt_allocations(a))"
        ## run algorithms
        J = wendyProb.J
        results = NamedTuple(
            algoName=>AlgoRes()
            for algoName in algoNames
        );
        # Loop through algos and run this example
        @info "=================================="
        @info "Starting Optimization Routines"
        for algoName in algoNames 
            @info "  $algoName"
            alg_dt = @elapsed begin   
                what, iters, wits = if algoName == :truth 
                    ex.wTrue, 0, reshape(ex.wTrue,J,1)
                elseif algoName == :init
                    w0, 0, reshape(w0,J,1)
                else
                    try 
                        solve(wendyProb, w0, ex.params; alg=algorithms[algoName], return_wits=true)
                    catch
                        (NaN * ones(J), -1, zeros(J,0)) 
                    end
                end
            end
            computeMetrics!(results[algoName], wendyProb, ex, alg_dt, what, iters, wits;ll=ll)
        end

        return wendyProb, w0, results
    end
end 