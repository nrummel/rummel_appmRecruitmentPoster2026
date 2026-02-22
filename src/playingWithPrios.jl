using Distributions
import Distributions: logpdf
using PlotlyJS
using Symbolics
## The idea is to get the logpdf, and its derivatives from symbolic expressions
function _logPrior(prior::Distribution)
    @variables p
    π_sym = try
        -logpdf(prior,p)
    catch e
        @error "This function logpdf function probably is not symbolics friendly, please extend the logpdf function of interest"
        throw(e)
    end
    ∇ₚπ_sym = Symbolics.derivative(π_sym, p)
    ∇ₚ²π_sym = Symbolics.derivative(∇ₚπ_sym, p)
    
    _π = build_function(π_sym, p; expression=false)
    _∂π = build_function(∇ₚπ_sym, p; expression=false)
    _∂²π = build_function(∇ₚ²π_sym, p; expression=false)
    int = support(prior)
    function π(p)
        if p < int.lb || p > int.ub
            return 1e6
        end
        return _π(p)
    end
    function ∂π(p)
        if p < int.lb || p > int.ub
            return 1e6
        end
        return _∂π(p)
    end
    function ∂²π(p)
        if p < int.lb || p > int.ub
            return 1e6
        end
        return _∂²π(p)
    end
    return π, ∂π, ∂²π
end
## Beta 
function logpdf(dist::Beta, x::T) where {T <: Real}
    return Distributions.xlogy(dist.α - 1, x) + Distributions.xlog1py(dist.β - 1, -x) - Distributions.logbeta(dist.α, dist.β)
end
xx = 0.0:0.01:1.0
trs = AbstractTrace[]
for (a,b,c) = [(.5,.5, "red"), (5,1, "blue"), (1,3, "green"), (2,2, "purple"), (2,5, "orange")]
    π, ∂π, ∂²π = _logPrior(Beta(a,b));
    push!(trs,scatter(x=xx, legendgroup="$a,$b", y=exp.(-π.(xx)), name="$a, $b", marker_color=c))
    push!(trs,scatter(x=xx, legendgroup="$a,$b", y=∂π.(xx), name="$a, $b", showlegend=false, marker_color=c,yaxis="y2"))
    push!(trs,scatter(x=xx, legendgroup="$a,$b", y=∂²π.(xx), name="$a, $b", showlegend=false, marker_color=c,yaxis="y3"))
end
plot(trs, Layout(
    title="Beta",
    yaxis=attr(domain=[0,.3],title="exp(Π)"),
    yaxis2=attr(domain=[.45, .65],title="∂π"),
    yaxis3=attr(domain=[.7,1], title="∂²π"),
))

## Gamma
function logpdf(dist::Gamma, x::T) where {T <: Real}
    α,θ = dist.α, dist.θ
    - x / θ +  (α -1) * log(x) - α *log(θ) - Distributions.loggamma(α)
end 
xx = 0.0:0.1:20.0
trs = AbstractTrace[]
for (a,t,c) = [(1,2, "red"), (2,2, "orange"), (3,2, "yellow"), (5,1, "green"), (9,0.5, "black"), (7.5,1, "blue"), (0.5,1, "purple")]
    π, ∂π, ∂²π = _logPrior(Gamma(a,t));
    push!(trs,scatter(x=xx, legendgroup="$a,$t", y=exp.(-π.(xx)), name="$a, $t", marker_color=c))
    push!(trs,scatter(x=xx, legendgroup="$a,$t", y=∂π.(xx), name="$a, $t", showlegend=false, marker_color=c,yaxis="y2"))
    push!(trs,scatter(x=xx, legendgroup="$a,$t", y=∂²π.(xx), name="$a, $t", showlegend=false, marker_color=c,yaxis="y3"))
end
plot(trs, Layout(
    title="Gamma",
    yaxis=attr(domain=[0,.3],range=[0,0.7], title="exp(Π)"),
    yaxis2=attr(domain=[.45, .65],range=[-10,10], title="∂π"),
    yaxis3=attr(domain=[.7,1], range=[-10,10], title="∂²π"),
))
## Frechet https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution
function logpdf(dist::Frechet, x::T) where {T <: Real}
    α, θ = dist.α, dist.θ
    return log(α) - log(θ) + (-1-α) * (log(x) - log(θ)) - (x / θ) ^ (-α)
end
xx = 0.0:0.01:5.0
trs = AbstractTrace[]
for (a,s,c) = [(1,1, "blue"), (2,1, "green"), (3,1, "red"), (1,2, "teal"), (2,2, "purple"), (3,2, "yellow")]
    π, ∂π, ∂²π = _logPrior(Frechet(a,s));
    push!(trs,scatter(x=xx, legendgroup="$a,$s", y=exp.(-π.(xx)), name="$a, $s", marker_color=c))
    push!(trs,scatter(x=xx, legendgroup="$a,$s", y=∂π.(xx), name="$a, $s", showlegend=false, marker_color=c,yaxis="y2"))
    push!(trs,scatter(x=xx, legendgroup="$a,$s", y=∂²π.(xx), name="$a, $s", showlegend=false, marker_color=c,yaxis="y3"))
end
plot(trs, Layout(
    title="Frechet",
    yaxis=attr(domain=[0,.3], range=[-.1,1.5],title="exp(Π)"),
    yaxis2=attr(domain=[.45, .65], range=[-10,10],title="∂π"),
    yaxis3=attr(domain=[.7,1], range=[-10,10], title="∂²π"),
))
## Gaussian
xx = -5.0:0.01:5.0
trs = AbstractTrace[]
for (a,s,c) = [(0,1, "blue"),(0,2, "red"),(0,0.5, "green") ]
    π, ∂π, ∂²π = _logPrior(Normal(a,s));
    push!(trs,scatter(x=xx, legendgroup="$a,$s", y=exp.(-π.(xx)), name="$a, $s", marker_color=c))
    push!(trs,scatter(x=xx, legendgroup="$a,$s", y=∂π.(xx), name="$a, $s", showlegend=false, marker_color=c,yaxis="y2"))
    push!(trs,scatter(x=xx, legendgroup="$a,$s", y=∂²π.(xx), name="$a, $s", showlegend=false, marker_color=c,yaxis="y3"))
end
plot(trs, Layout(
    title="Normal",
    yaxis=attr(domain=[0,.3], range=[-.1,1],title="exp(Π)"),
    yaxis2=attr(domain=[.45, .65], range=[-10,10],title="∂π"),
    yaxis3=attr(domain=[.7,1], range=[-10,10], title="∂²π"),
))
##
using SpecialFunctions: gamma
α = 3
s = 5 / ((α / (1 + α))^(1 / α ))
mode(Frechet(α,s))
##
priors = [2*(Beta(2,5)+.8), 2.5*Gamma(2,2), Frechet(α,s), Normal(0,1)]
p₀ = [rand(π) for π in priors]
constraints = [support(prior) for prior in priors]
priorFunList = [_logPrior(prior) for prior in priors]
function Π(p) 
    v = sum(f[1](p[i]) for (i,f) in enumerate(priorFunList))
    # println("Π($p)=$v")
    return v
end
function ∇ₚΠ!(g, p) 
    for (i,f) in enumerate(priorFunList)
        g[i] = f[2](p[i]) 
    end
    #  println("∇ₚΠ($p) = $g")
    return g 
end
function ∇ₚ²Π!(H, p) 
    H .= 0
    for (i,f) in enumerate(priorFunList)
        H[i,i] = f[3](p[i]) 
    end
    # println("∇ₚ²Π($p) = $H")
    return H
end
## 
using Optim, NLSolversBase
df = TwiceDifferentiable(Π, ∇ₚΠ!, ∇ₚ²Π!, p₀)
dfc = TwiceDifferentiableConstraints([int.lb for int in constraints], [int.ub for int in constraints])
res = optimize(df, dfc, p₀, IPNewton())
pstar = res.minimizer
##
# g₀ = zeros(4)
# ∇ₚΠ!(g₀, p₀)
# H₀ = zeros(4,4)
# ∇ₚ²Π!(H₀, p₀)
# t = 0:0.01:1
# φ =  [Π( p₀ - ti * (H₀ \ g₀)) for ti in t]
# plot(scatter(x=t, y=φ), Layout(title="Linsearch for the first gradient descent step", yaxis_range=[0,30]))
##
##
p = make_subplots(rows=length(priors), cols=1)
for (j,prior) in enumerate(priors)
    int = support(prior)
    l = clamp(int.lb, -10,15)
    u = clamp(int.ub, -10,15)
    xx = l:0.01:u
    π, _, _ = _logPrior(prior);
    add_trace!(p,scatter(x=xx, y=π.(xx), legendgroup="priors", legendgrouptitle_text="Priors", name="Π_$j", yaxis="y$j", line_color="blue"), row=j, col=1)
    add_trace!(p,scatter(x=[p₀[j]], y=[π(p₀[j])], showlegend=j==1,name="p₀", yaxis="y$j", marker=attr(symbol="square", color="black", size=10)), row=j, col=1)
    add_trace!(p,scatter(x=[pstar[j]], y=[π(pstar[j])], showlegend=j==1, name="p*", yaxis="y$j", marker=attr(symbol="star", color="green", size=10)), row=j, col=1)
end
relayout!(p, title="Negative Log Pdf of the Priors",
    yaxis_range=[-2,10],
    yaxis2_range=[-2,10],
    yaxis3_range=[-2,10],
    yaxis4_range=[-2,10],
)
p