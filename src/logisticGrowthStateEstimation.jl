@info "Loading Dependencies"
includet("util.jl")
includet("../plot/plotBase.jl")
using Distributions, Printf
## define priors
α = 3
s = 5 / ((α / (1 + α))^(1 / α ))
priors = [2*(Beta(2,5) + .8), Frechet(α,s)]
p_mode = [mode(prior) for prior in priors]
pstar = p_mode + [.25, 2]
p₀ = [rand(prior) for prior in priors]
@info "argmax(prior) = $(p_mode)"
@info "p* = $pstar"
@info "p₀ = $p₀"
## Build ODE and simulate noise
function LOGISTIC_f!(du, u, w, t)
    du[1] = w[1] * u[1] - w[2] * u[1]^2
    nothing
end
noiseDist = Val(LogNormal)
wendyParams = WENDyParameters(;Kmax=1000)
ex = SimulatedWENDyData(
    "logisticGrowth", 
    LOGISTIC_f!,
    [0.01], # u₀
    (0.0, 10.0), # (t₀, tₘ)
    pstar, # p*
    [
        (0.0,10.0),
        (0.0,10.0)
    ],
    wendyParams;
    linearInParameters=Val(true),
    noiseDist=noiseDist
);
simParams = SimulationParameters(
    seed=1, 
    timeSubsampleRate=4,
    noiseRatio=.5,
    corruptU0=true
) 
simulate!(ex, simParams, ll=Warn)
J = length(p₀)
wendyProb = WENDyProblem(ex; priors=priors, ll=Info);
## extract data from their structures
p1a = plotjs(ex)
##
using ForwardDiff, PreallocationTools
using WENDy: _wnll, _g!, _L₀!, _L!, _R!
Mp1, K, D = wendyProb.Mp1, wendyProb.K, wendyProb.D
KD = K*D
tt = wendyProb.data.tt 
X₀ = wendyProb.data.X 
f! = wendyProb.data.f! 
∇ₓf! = wendyProb.data.∇ₓf! 
sig = wendyProb.data.sig
V = wendyProb.data.V
Vp = wendyProb.data.Vp
diagReg = wendyParams.diagReg
# Cholesky factor of the covariance
function cholFact(x)
    X = reshape(x, Mp1, D)
    __L₀ = get_tmp(dualcache(zeros(K,D,D,Mp1)),x) # buffer
    _L₀ = get_tmp(dualcache(zeros(K,D,Mp1,D)),x) # buffer
    L₀ = get_tmp(dualcache(zeros(K*D,Mp1*D)),x) # output
    _L₀!(L₀,Vp, sig,__L₀,_L₀)

    JuF = get_tmp(dualcache(zeros(D,D,Mp1)),x) # buffer
    _L₁ = get_tmp(dualcache(zeros(K,D,Mp1,D)),x) # buffer
    L = get_tmp(dualcache(zeros(K*D,Mp1*D)),x) # output
    _L!(L,pstar,tt,X,V,L₀,sig,∇ₓf!,JuF, _L₁)

    thisI = Matrix(I, KD, KD)
    S = get_tmp(dualcache(zeros(KD, KD)),x) # buffer
    Sreg = get_tmp(dualcache(zeros(KD, KD)),x) # buffer
    
    R = get_tmp(dualcache(zeros(KD, KD)),x) # output
    _R!(R,L, diagReg, thisI, Sreg, S; doChol=true)
    return R
end
# Unweighted residual
function res!(r, x)
    X = reshape(x, Mp1, D)
    ## Compute the residual
    F = get_tmp(dualcache(zeros(Mp1, D)),x)# buffer
    G = get_tmp(dualcache(zeros(K, D)),x) # buffer
    g = get_tmp(dualcache(zeros(KD)),x) # output
    _g!(g, pstar, tt, X, V, f!, F, G) # wRhs
    b = reshape(-Vp * X, K*D) # wLhs
    r .= g-b
    return r 
end
function res(x)
    r = get_tmp(dualcache(zeros(KD)),x) # output
    return res!(r, x)
end
# Weighted Residual 
function weightedRes!(wr, x, R=nothing; retR=false)
    r = res(x)
    isnothing(R) && (R = cholFact(x))
    wr .= R' \ r
    return retR ? (wr,R) : wr
end
function weightedRes(x, R=nothing; retR=false)
    wr = get_tmp(dualcache(zeros(KD)),x)
    return weightedRes!(wr,x,R; retR=retR)
end
# likelihood
function ℓ(x)
    r, R = weightedRes(x; retR=true)
    logDet = 2*sum(log.(diag(R)))
    constTerm = KD/2*log(2*pi)
    return 1/2*(norm(r)^2 + logDet) + constTerm
end
x₀ =  reshape(X₀,Mp1*D)
@info @sprintf "ℓ(X₀; p*) = %.4g" ℓ(x₀)
@info @sprintf "ℓ(p*; x₀) = %.4g" wendyProb.wnll.f(pstar)
@info @sprintf "relErr = %.4g" norm(ℓ(x₀)-wendyProb.wnll.f(pstar)) / norm(wendyProb.wnll.f(pstar))
##
∇ₓℓ!(g,x) = ForwardDiff.gradient!(g, ℓ, x)
∇ₓres!(J,x) = ForwardDiff.jacobian!(J, res, x)
∇ₓweightedRes!(J,x) = ForwardDiff.jacobian!(J, weightedRes, x)
# ∇ₓ²ℓ(H,x) = ForwardDiff.hessian!(H, ℓ, x)
g₀ = zeros(Mp1*D)
J₀ = zeros(KD,Mp1*D)
# H₀ = zeros(Mp1*D,Mp1*D)
@info "Negative Log-likelihood Gradient"
@time ∇ₓℓ!(g₀,x₀)
@time ∇ₓℓ!(g₀,x₀)
@info "Residual Jacobian"
@time ∇ₓres!(J₀,x₀) 
@time ∇ₓres!(J₀,x₀) 
@info "Weighted Residual Jacobian"
@time ∇ₓweightedRes!(J₀,x₀) 
@time ∇ₓweightedRes!(J₀,x₀) 
# @info "Hessian"
# @time ∇ₓ²ℓ(H₀,x₀)
# @time ∇ₓ²ℓ(H₀,x₀)
nothing
## Perform Optimization 
using WENDy: LeastSquaresCostFunction, FirstOrderCostFunction
using WENDy: irls, nonlinearLeastSquares, bfgs
function irls_iter(xnm1;ll=Warn)
    R = cholFact(xnm1)
    _weightedRes(x) = weightedRes(x,R)
    _weightedRes!(wr,x) = weightedRes!(wr,x,R)
    _∇ₓweightedRes!(J,x) = ForwardDiff.jacobian!(J, _weightedRes, x)
    xn = nonlinearLeastSquares(LeastSquaresCostFunction(_weightedRes!, _∇ₓweightedRes!,KD), xnm1, wendyParams)
    return xn 
end
xhat_ols = nonlinearLeastSquares(LeastSquaresCostFunction(res!, ∇ₓres!,KD), x₀,wendyParams)
xhat_irls,_ = irls(irls_iter, x₀, wendyParams)
xhat_nls = nonlinearLeastSquares(LeastSquaresCostFunction(weightedRes!, ∇ₓweightedRes!,KD), x₀,wendyParams)
# Accuracy checks
xstar = reshape(log.(ex.Ustar[]),Mp1*D)
@info @sprintf "ols relErr = %.4g" norm(xhat_ols-xstar) / norm(xstar)
@info @sprintf "irls relErr = %.4g" norm(xhat_irls-xstar) / norm(xstar)
@info @sprintf "nls relErr = %.4g" norm(xhat_nls-xstar) / norm(xstar)
##
# g = zeros(Mp1*D)
# ∇ₓℓ!(g, xhat_irls)
# @info @sprintf "‖∇ₓℓ‖ = %.4g" norm(g)
# xhat_mle = bfgs(FirstOrderCostFunction(ℓ, ∇ₓℓ!), xhat_irls, wendyParams)
## Plot the results
Uhat_ols = reshape(exp.(xhat_ols), Mp1, D)
Uhat_irls = reshape(exp.(xhat_irls), Mp1, D)
Uhat_nls = reshape(exp.(xhat_nls), Mp1, D)
# Uhat_mle = reshape(exp.(xhat_mle), Mp1, D)
U₀ = ex.U[]
Ustar = ex.Ustar[]
pp = make_subplots(cols=3)
# data
add_trace!(pp,
    scatter( 
        x=tt, y=U₀[:], name=L"\{\mathbf{u}_m\}_{m=0}^M", mode="markers", marker_color="blue", marker_symbol="square", marker_size=5, marker_opacity=1
    ),row=1, col=1)
add_trace!(pp,
    scatter( 
        x=tt, y=U₀[:], mode="markers", marker_color="blue", marker_symbol="square", marker_size=5, marker_opacity=1, showlegend=false
    ),row=1, col=2)
add_trace!(pp,
    scatter( 
        x=tt, y=U₀[:], mode="markers", marker_color="blue", marker_symbol="square", marker_size=5, marker_opacity=1, showlegend=false
    ),row=1, col=3)
# Truth 
add_trace!(pp,
    scatter( 
        x=tt, y=Ustar[:], name=L"\{\mathbf{u}^*_m\}_{m=0}^M", line_width=5, line_color="black"
    ),row=1, col=1)
add_trace!(pp,
    scatter( 
        x=tt, y=Ustar[:], line_width=5, line_color="black", showlegend=false
    ),row=1, col=2)
add_trace!(pp,
    scatter( 
        x=tt, y=Ustar[:], line_width=5, line_color="black", showlegend=false
    ),row=1, col=3)
# estimates
add_trace!(pp,
    scatter( 
        x=tt, y=Uhat_ols[:], name=L"\text{Ordinary Least Squares } \hat{\mathbf{U}}_\mathrm{ols}", mode="markers", marker_color="purple", marker_size=4, marker_opacity=0.8
    ),row=1, col=1)
add_trace!(pp,
    scatter( 
        x=tt, y=Uhat_irls[:], name=L"\text{Iterative Reweighted Least Squares } \hat{\mathbf{U}}_\mathrm{irls}", mode="markers", marker_color="red", legendgrouptitle=attr(text="Estimates", font_size=15), marker_size=4, marker_opacity=0.8
    ),row=1, col=2)
add_trace!(pp,
    scatter( 
        x=tt, y=Uhat_nls[:], name=L"\text{Nonlinear Least Squares } \hat{\mathbf{U}}_\mathrm{nls}", mode="markers", marker_color="green", marker_size=4, marker_opacity=0.8
    ),row=1, col=3)

relayout!(pp,
    title=attr(text="Comparing State Estimates",font_size=20),
    yaxis=attr(range=[-1,2]),
    xaxis=title=attr(text="time", font_size=15),
    yaxis2=attr(range=[-1,2]),
    xaxis2=title=attr(text="time", font_size=15),
    yaxis3=attr(range=[-1,2]),
    xaxis3=title=attr(text="time", font_size=15),
)
savefig(pp, joinpath(@__DIR__, "../../docs/fig/LogisticGrowth_stateEstimation.pdf"),height=400, width=1200)
pp