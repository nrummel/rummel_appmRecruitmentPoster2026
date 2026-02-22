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
