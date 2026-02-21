@info "Loading Dependencies"
includet("util.jl")
includet("../plot/plotBase.jl")
using Distributions, Tullio, ProgressMeter, Random
using Base.Meta
using Symbolics: @variables, jacobian, build_function
##
FIG_DIR = "/Users/niru8088/scratch/bayesWENDy/recruitmentPoster2026/figures"
# define priors
Random.seed!(1)
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
    WENDyParameters(;Kmax=1000);
    linearInParameters=Val(true),
    noiseDist=noiseDist,
    Mp1=501
);
simParams = SimulationParameters(
    seed=1, 
    timeSubsampleRate=1,
    noiseRatio=.3,
    corruptU0=true
) 
simulate!(ex, simParams, ll=Warn)
J = length(p₀)
tt=ex.tt[]
U=ex.U[]
U_exact= ex.Ustar[] 
dt = diff(tt)
dt = dt[1];
Mp1 = size(U, 1)
D = size(U, 2)
# Get FS Error 
function fsErr(phat)
    u0star = ex.Ustar[][1,:]
    tRng = (tt[1], tt[end])
    odeprob = ODEProblem(LOGISTIC_f!, u0star, tRng, phat)
    sol = solve_ode(odeprob; 
        reltol=ex.params.optimReltol, abstol=ex.params.optimAbstol, 
        saveat=dt, verbose=false
    )
    Uhat = reduce(vcat, um' for um in sol.u)[:]
    fsRelErr = norm(Uhat - U_exact) / norm(U_exact)
    return fsRelErr, Uhat
end
@info "========================================================================="
# Build and solve wendy problem 
wendyProb = WENDyProblem(ex; priors=priors, ll=Info)
phat_MLE, iters, P = solve(wendyProb, p₀, ex.params; alg=WENDy.IP(), costFun=:wnll,return_wits=true);
# relative error 
relErr_MLE = norm(phat_MLE - pstar) / norm(pstar)
@info "WENDy MLE"
@info @sprintf "  Relation Coefficient error %.2g" relErr_MLE
fsRelErr_MLE, Uhat_MLE = fsErr(phat_MLE)
@info @sprintf "  Average Relative Forward Solver Error %.2g" fsRelErr_MLE
## Plot the Trajectory with the estimates
p1a = plotjs(
    AbstractTrace[
        scatter(
            x=tt,
            y=U[:,1],
            name="\$\\{(\\mathrm{t}_m, \\mathrm{u}_m)\\}\$", 
            mode="markers" ,
            marker_color=PLOTLYJS_COLORS[1], 
            marker_opacity=0.5, 
            marker_size=10, 
        ),
        scatter(
            x=tt,
            y=Uhat_MLE,
            name="\$u(\\hat{\\mathbf{p}})\$",
            mode="lines",
            line=attr(
                color=CU_BOULDER_COLORS[2],
                width=5,
            )
        ),
        scatter(
            x=tt,
            y=U_exact[:,1],
            name="\$u(\\mathbf{p}^*)\$", 
            mode="lines" ,
            line=attr(
                color=CU_BOULDER_COLORS[1],
                dash="dash",
                width=5,
                opacity=0.3
            ),
        )
    ], 
    Layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=attr(
            text="Logistic Growth", 
            x=0.5,
            xanchor="center",
            font_size=20
        ),
        # yaxis_type=yaxis_type,
        showlegend=true, 
        xaxis=attr(
            title=attr(
                text="\$t\$",
                font_size=25
            ),
            tick_font_size=20,
            showgrid=true, 
            zeroline=true,
            # domain=[0.1,0.9]
        ),
        yaxis=attr(
            title=attr(
                text="\$u\$",
                font_size=25
            ),
            tick_font_size=20,
            showgrid=true, 
            zeroline=true
        ),
        legend=attr(
            # x=1,
            # y=1,
            # xref="paper", 
            # yref="paper",
            # xanchor="right",
            # yanchor="top",
            font=(
                family="sans-serif",
                size=20,
                color="#000"
            ),
            bgcolor="#E2E2E2",
            bordercolor= "#636363",
            # entrywidth= 0,        # Manually set this based on your longest equation
            # entrywidthmode= "pixels", 
            # borderpad= 20,          # Give extra "buffer" for tall fractions or exponents
            borderwidth= 1
        ),
        # margin_r= 150,
        hovermode="x unified"
    )
)
# save figure
savefig(
    p1a, 
    joinpath(FIG_DIR,"LogisticGrowth_TrajectoryWithData.pdf"),
    height=400, 
    width=600
)
# display figure
p1a
## Plot the priors
"""
    p1b = make_subplots(rows=1, cols=2)
    # first prior
    xx = 0:0.01:4
    π1(x) = pdf(priors[1],x)
    add_trace!(p1b,scatter(x=xx, y=π1.(xx), legendgroup="priors", legendgrouptitle_text="Priors", name="\$\\pi_1\$", yaxis="y1", line_color="blue"), row=1, col=1)
    add_trace!(p1b,scatter(x=[p₀[1]], y=[π1(p₀[1])],name="\$\\mathbf{p}^{(0)}\$", yaxis="y1", mode="markers", marker=attr(symbol="square", color="black", size=10)), row=1, col=1)
    add_trace!(p1b,scatter(x=[mode(priors[1])], y=[π1(mode(priors[1]))], name="\$\\operatorname{argmax}\\bigl(\\Pi(\\mathbf{p}) \\bigr) \$", yaxis="y1", mode="markers", marker=attr(symbol="square", color="red", size=10)), row=1, col=1)
    add_trace!(p1b,scatter(x=[pstar[1]], y=[π1(pstar[1])], name="\$\\mathbf{p}^*\$", yaxis="y1", mode="markers", marker=attr(symbol="star", color="green", opacity=0.8, size=10)), row=1, col=1)
    # second prior
    xx = 0:0.01:20
    π2(x) = pdf(priors[2],x)
    add_trace!(p1b,scatter(x=xx, y=π2.(xx), legendgroup="priors", legendgrouptitle_text="Priors", yaxis="y1", line_color="blue", showlegend=false), row=1, col=2)
    add_trace!(p1b,scatter(x=[p₀[2]], y=[π2(p₀[2])], yaxis="y1",mode="markers", marker=attr(symbol="square", color="black", size=10), showlegend=false), row=1, col=2)
    add_trace!(p1b,scatter(x=[mode(priors[2])], y=[π2(mode(priors[2]))], yaxis="y1",mode="markers", marker=attr(symbol="square", color="red", opacity=0.8, size=10), showlegend=false), row=1, col=2)
    add_trace!(p1b,scatter(x=[pstar[2]], y=[π2(pstar[2])], yaxis="y1",mode="markers", marker=attr(symbol="star", color="green", size=10), showlegend=false), row=1, col=2)
    # format
    relayout!(p1b, title="PDFs of the Priors", xaxis=attr(title=attr(text="Shifted and Scaled Beta")), xaxis2=attr(title=attr(text="Frechet"))
    )
    p1b
    savefig(p1b, joinpath(FIG_DIR, "LogistiGrowth_oneDimPriors.pdf"),height=400, width=600)
"""
## Build and solve wendy problem 
wendyProb = WENDyProblem(ex; priors=priors, ll=Info)
phat_prior, iters, P = solve(wendyProb, p₀, ex.params; alg=WENDy.IP(), costFun=:priorLoss,return_wits=true);
phat_MLE, iters, P = solve(wendyProb, p₀, ex.params; alg=WENDy.IP(), costFun=:wnll,return_wits=true);
phat_MAP, iters, P = solve(wendyProb, p₀, ex.params; alg=WENDy.IP(), costFun=:wnlp,return_wits=true);
#
relErr_prior = norm(phat_prior - pstar) / norm(pstar)
relErr_MLE = norm(phat_MLE - pstar) / norm(pstar)
relErr_MAP = norm(phat_MAP - pstar) / norm(pstar)
@info "Relation coefficient error with prior $(relErr_prior)"
@info "Relation coefficient error with MLE $(relErr_MLE)"
@info "Relation coefficient error with MAP $(relErr_MAP)"
## Let's see if we can expand the weak resual with symbolic programming
# Get all the data we need in local vars
phat = pstar # use this approximation of p
X = noiseDist === Val(LogNormal) ? log.(U) : U # transformed state
Xstar = noiseDist === Val(LogNormal) ? log.(U_exact) : U_exact # transformed TRUE state
E = X - Xstar # noise (Mp1 x D)
Φ = wendyProb.data.V # test fun matrix (K x Mp1)
Φ_dot = wendyProb.data.Vp # der test fun matrix (K x Mp1)
K = size(Φ,1)
B = - Φ_dot * X # weak lhs (K x D)
B_star = - Φ_dot * Xstar
B_eps = - Φ_dot * E # weak lhs (K x D)
# Transform the rhs fun so that the lognormal noise to becomes additive gaussian
f! = WENDy._getf(LOGISTIC_f!, D, J, noiseDist) 
# The jacobian wrt P
∇ₚf! = WENDy._get∇ₚf(f!, D, J) 
@variables p[1:J] x[1:D] t
dx = Matrix(undef, D, J)
∇ₚf!(dx, x, p, t)
∇ₚ²f_sym = jacobian(dx, x)
∇ₚ²f = build_function(∇ₚ²f_sym, x, p, t; expression=false)[end]
# Get the true weak residual R = Φ*F - B
res = wendyProb.wnll.f.r!(phat) # from tested WENDy.jl code
R = reshape(res, K, D); # weak residual (K x D)
## Solve a nonlinear system to debias
function r_p!(r,p)
    F = zeros(Mp1,D)
    ∇ₚF = zeros(Mp1,D,J)
    dx0 = zeros(D)
    dx1 = zeros(D, J)
    for m = 1:Mp1
        @views x = X[m,:]
        t = tt[m]
        f!(dx0, x, p, t)
        @views F[m,:] .= dx0
        ∇ₚf!(dx1, x, p, t)
        @views ∇ₚF[m,:,:] .= dx1
    end
    B = - Φ_dot * X
    Φ_F = Φ * F 
    Φ_F = reshape(Φ_F, K*D)
    Φ_∇ₚF = zeros(K, D, J)
    @tullio Φ_∇ₚF[k,d,j] = Φ[k,m] * ∇ₚF[m,d,j] 
    Φ_∇ₚF = reshape(Φ_∇ₚF, K*D, J)
    weakResidual = reshape(Φ_F-B, K*D)
    r .= Φ_∇ₚF \ weakResidual
    return r
end
function ∇ₚr_p!(∇ₚr, p)
    F = zeros(Mp1,D)
    ∇ₚF = zeros(Mp1,D,J)
    ∇ₚ²F = zeros(Mp1,D,J,J)
    dx0 = zeros(D)
    dx1 = zeros(D,J)
    dx2 = zeros(D,J,J)
    for m = 1:Mp1
        @views x = X[m,:]
        t = tt[m]
        f!(dx0, x, p, t)
        @views F[m,:] .= dx0
        ∇ₚf!(dx1, x, p, t)
        @views ∇ₚF[m,:,:] .= dx1
        ∇ₚ²f(dx2, x, p, t)
        @views ∇ₚ²F[m,:,:,:] .= dx2
    end
    # 0th der
    weakResidual = reshape(Φ*F+Φ_dot*X, K*D)
    # 1st der
    Φ_∇ₚF = zeros(K, D, J)
    @tullio Φ_∇ₚF[k,d,j] = Φ[k,m] * ∇ₚF[m,d,j] 
    Φ_∇ₚF = reshape(Φ_∇ₚF, K*D, J)
    # 2nd der
    Φ_∇ₚ²F = zeros(K, D, J, J)
    @tullio Φ_∇ₚ²F[k,d,j1,j2] = Φ[k,m] * ∇ₚ²F[m,d,j1,j2] 
    Φ_∇ₚ²F = reshape(Φ_∇ₚ²F, K*D, J, J)
    # product and chain rule 
    tmp1 = Φ_∇ₚF \ weakResidual # J
    @tullio tmp2[k,j1] := Φ_∇ₚ²F[k,j1,j2] * tmp1[j2]
    # @show size(tmp2)
    # @show size(Φ_∇ₚF)
    fact = svd(Φ_∇ₚF*Φ_∇ₚF')
    U,S,Vt = fact.U, fact.S, fact.Vt
    # @show size(S)
    Sinv = vcat(
        hcat(
            diagm([1/s for s in S[1:J]]),
            zeros(J, K*D-J)
        ),
        zeros(K*D-J, K*D)
    )
    tmp3 = Φ_∇ₚF'*(U*Sinv*Vt)' * tmp2
    # @show size(tmp3)
    ∇ₚr .= Matrix{Float64}(I, J, J) - tmp3
    return ∇ₚr
end
∇ₚr = zeros(J,J) 
∇ₚr_p!(∇ₚr, pstar)
∇ₚr
##
phat_cor = WENDy.nonlinearLeastSquares(
    LeastSquaresCostFunction(r_p!,∇ₚr_p!,J),
    phat_MAP,
    WENDyParameters()
)
relErr_prior = norm(phat_prior - pstar) / norm(pstar)
relErr_MLE = norm(phat_MLE - pstar) / norm(pstar)
relErr_MAP = norm(phat_MAP - pstar) / norm(pstar)
relErr_cor = norm(phat_cor - pstar) / norm(pstar)
@info @sprintf "Relation coefficient error with prior      %.4g" relErr_prior
@info @sprintf "Relation coefficient error with MLE        %.4g" relErr_MLE
@info @sprintf "Relation coefficient error with MAP        %.4g" relErr_MAP
@info @sprintf "Relation coefficient error with correction %.4g" relErr_cor
## sanity check
r = similar(pstar)
offset =20
pp1 = [[p1, pstar[2]] for p1 in pstar[1]-offset:0.1:pstar[1]+offset]
pp2 = [[pstar[1], p2] for p2 in pstar[2]-offset:0.1:pstar[2]+offset]
p1 = plot(
    [
        scatter(
            x = pstar[1]-offset:0.1:pstar[1]+offset,
            y = [norm(r_p!(r,p1))^2 for p1 in pp1],
            name="p₁",
            legendgroup="p"
        ),
        scatter(
            x = [pstar[1]],
            y = [norm(r_p!(r,pstar))^2],
            name="p⋆₁",
            legendgroup="p⋆",
            mode="markers",
            marker_color="black",
            marker_symbol="star",
            marker_size=10,
        ),
        scatter(
            x = [phat_MAP[1]],
            y = [norm(r_p!(r,phat_MAP))^2],
            name="p̂₁",
            legendgroup="p̂",
            mode="markers",
            marker_color="green",
            marker_symbol="square",
            marker_size=10,
        ),
        scatter(
            x = [phat_cor[1]],
            y = [norm(r_p!(r,phat_cor))^2],
            name="p̃₁",
            legendgroup="p̃",
            mode="markers",
            marker_color="red",
            marker_symbol="circle",
            marker_size=10,
        )
    ],
    Layout(
        yaxis_title_text="loss", xaxis_title_text="p₁",
    )
)
p2 = plot(
    [
        scatter(
            x = pstar[2]-offset:0.1:pstar[2]+offset,
            y = [norm(r_p!(r, p2))^2 for p2 in pp2],
            name="p₂",
            legendgroup="p"
        ),
        scatter(
            x = [pstar[2]],
            y = [norm(r_p!(r,pstar))^2],
            name="p⋆",
            legendgroup="p⋆",
            mode="markers",
            marker_color="black",
            marker_symbol="star",
            marker_size=10,
        ),
        scatter(
            x = [phat_MAP[2]],
            y = [norm(r_p!(r,phat_MAP))^2],
            name="p̂₂",
            legendgroup="p̂",
            mode="markers",
            marker_color="green",
            marker_symbol="square",
            marker_size=10,
        ),
        scatter(
            x = [phat_cor[2]],
            y = [norm(r_p!(r,phat_cor))^2],
            name="p̃₂",
            legendgroup="p̃",
            mode="markers",
            marker_color="red",
            marker_symbol="circle",
            marker_size=10,
        )
    ],
    Layout(
        yaxis_title_text="loss", 
        xaxis_title_text="p₂"
    )
)
display([p1; p2])
## Get the residual only due to integration error 
Fstar = zeros(Mp1, D) 
for m = 1:Mp1 
    dx = zeros(D) 
    xstar = Xstar[m,:]
    t = tt[m] 
    f!(dx, xstar, phat, t)
    Fstar[m,:] .= dx
end
R_int = Φ*Fstar - B_star
@show total_error = norm(R)
int_error = norm(R_int)
@info (@sprintf "Percent error due to integration : %.4g" (int_error/total_error*100))*"%"
## Get all the functions with symbolic programming
α_max = 10
# the derivatives with respect to x need to be computed here
α_list = 0:α_max
∇ₓᵅf_sym_list = Vector(undef, α_max+1) # symbolic expr
∇ₓᵅf_list = Vector(undef, α_max+1) # functions
# store the 0th derivative (the function itself) in the first entry of the list
@variables p[1:J] x[1:D] t
dx = Vector(undef, D)
f!(dx, x, p, t)
∇ₓᵅf_sym_list[1] = dx 
∇ₓᵅf_list[1] = f!
for i = 2:α_max+1
    ∇ₓᵅ⁻¹f_sym = ∇ₓᵅf_sym_list[i-1]
    # take derivative
    ∇ₓᵅf_sym = jacobian(∇ₓᵅ⁻¹f_sym, x)
    # store in list
    ∇ₓᵅf_sym_list[i] = ∇ₓᵅf_sym
    ∇ₓᵅf_list[i] = build_function(∇ₓᵅf_sym, x, p, t; expression=false)[end]
end
## Now use the asymptotic expansion 
# first compute the linear part R_lin = Φ*(∇ᵤF*E) - B
∇ᵤF = zeros(Mp1,D,D)
∇ᵤf! = ∇ₓᵅf_list[2]
∇ᵤF_E = zeros(Mp1,D)
dx = zeros(D, D)
this = zeros(D)
for m = 1:Mp1
    x = X[m,:]
    t = tt[m]
    ∇ᵤf!(dx, x, phat, t)
    @views ∇ᵤF[m,:,:] .= dx
    ϵ = E[m,:]
    this = dx * ϵ
    ∇ᵤF_E[m,:] .= this 
end
Φ_∇ᵤF_E = Φ * ∇ᵤF_E 
Φ_∇ᵤF_E = reshape(Φ_∇ᵤF_E, K*D, D)
R_lin = Φ_∇ᵤF_E - B_eps
lin_error = norm(R - R_int - R_lin)
@info (@sprintf "Percent error due to truncation after linearization : %.4g" (lin_error/total_error*100))*"%"
##
function remainder(p; debug = false)
    rem = zeros(α_max-1, Mp1,D)
    # need these names in the global namespace ...
    this = zeros(D); dx = zeros(D); ϵ = zeros(D) 
    for i = 3:α_max 
        debug && (@info "i,α = $i,$α")
        α = α_list[i]
        ∇ₓᵅf_sym = ∇ₓᵅf_sym_list[i]
        # @info "∇ₓᵅf = $∇ₓᵅf_sym"
        ∇ₓᵅf = ∇ₓᵅf_list[i]
        local dx = zeros(Tuple(D for _ = 1:α+1))
        scale = 1 / factorial(α)
        for m = 1:Mp1
            debug && (@info "- m = $m")
            @views x = X[m,:]
            @views t = tt[m]
            debug && (@info "∇ₓᵅf(dx, x, p, t)")
            # @time 
            ∇ₓᵅf(dx, x, p, t)
            debug && (@info "-- dx = $dx")
            @views ϵ = E[m,:]
            # @tullio this[d,d1,...,dα] = dx[d,d1,...,dα]*ϵ[d1]*...*ϵ[dα]
            debug && (@info "-- @tullio this[d,d1,...,dα] = dx[d,d1,...,dα]*ϵ[d1]*...*ϵ[dα]")
            # @time 
            if α == 2 
                @tullio this[d] = dx[d,d1,d2]*ϵ[d1]*ϵ[d2]
            elseif α == 3 
                @tullio this[d] = (dx[d,d1,d2,d3]
                    *ϵ[d1]*ϵ[d2]*ϵ[d3])
            elseif α == 4 
                @tullio this[d] = (dx[d,d1,d2,d3,d4]
                    *ϵ[d1]*ϵ[d2]*ϵ[d3]*ϵ[d4])
            elseif α == 5 
                @tullio this[d] = (dx[d,d1,d2,d3,d4,d5]
                    *ϵ[d1]*ϵ[d2]*ϵ[d3]*ϵ[d4]*ϵ[d5])
            elseif α == 6 
                @tullio this[d] = (dx[d,d1,d2,d3,d4,d5,d6]
                    *ϵ[d1]*ϵ[d2]*ϵ[d3]*ϵ[d4]*ϵ[d5]*ϵ[d6])
            elseif α == 7 
                @tullio this[d] = (dx[d,d1,d2,d3,d4,d5,d6,d7]
                    *ϵ[d1]*ϵ[d2]*ϵ[d3]*ϵ[d4]*ϵ[d5]*ϵ[d6]*ϵ[d7])
            elseif α == 8 
                @tullio this[d] = (dx[d,d1,d2,d3,d4,d5,d6,d7,d8]
                    *ϵ[d1]*ϵ[d2]*ϵ[d3]*ϵ[d4]*ϵ[d5]*ϵ[d6]*ϵ[d7]*ϵ[d8])
            elseif α == 9 
                @tullio this[d] = (dx[d,d1,d2,d3,d4,d5,d6,d7,d8,d9]
                    *ϵ[d1]*ϵ[d2]*ϵ[d3]*ϵ[d4]*ϵ[d5]*ϵ[d6]*ϵ[d7]*ϵ[d8]*ϵ[d9])
            elseif α == 10 
                @tullio this[d] = (dx[d,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10]
                    *ϵ[d1]*ϵ[d2]*ϵ[d3]*ϵ[d4]*ϵ[d5]*ϵ[d6]*ϵ[d7]*ϵ[d8]*ϵ[d9]*ϵ[d10])
            else 
                @error "not implemented"
            end
            debug && (@info "-- this=$this")
            rem[i-2,m,:] .+= scale * this
            # if m > 2
            #     break
            # end
        end
        # break
    end 
    if debug 
        for (i,α) = zip(1:size(rem,1), 2:α_max)
            @info "α = $(α)"
            @info "- norm(rem) = $(norm(rem[i,m,:]))"
        end
    end
    Rem = reshape(sum(rem,dims=1), Mp1, D)
    return Rem 
end
Rem = remainder(phat);
##
lin_error = norm(R - R_int - R_lin )
@info (@sprintf "Percent error due to truncation after linearization : %.4g" (lin_error/total_error*100))*"%"
trun_error = norm(R - R_int - R_lin + Φ*Rem)
@info (@sprintf "Percent error due to truncation after %s terms : %.4g" α_max (trun_error/total_error*100))*"%"
## visualize how the error falls off 
[
    plot(
        [
            # scatter(
            #     x=1:K,
            #     y=R[:],
            #     name="R - weak residual"
            # ),
            scatter(
                x=1:K,
                y=R[:] - R_int[:],
                name="R_int - integration error"
            ),
            scatter(
                x=1:K,
                y=R[:] - (R_int[:] + R_lin[:]),
                name="R_lin - linearized residual"
            ),
            scatter(
                x=1:K,
                y=R[:] - (R_int[:] + R_lin[:] - (Φ*Rem)[:]),
                name="R_$α_max - terms after linearization"
            ),
        ],
        Layout(
            xaxis_title_text="Test Function Index",
            yaxis_title_text="Error",
            hovermode="x unified"
        )
    );
]
## Simple corrector
max_iter = 1
pp = Vector(undef, max_iter+1)
pp[1] = phat_MAP
for i = 2:max_iter+1
    @info "i = $i"
    pᵢ₋₁ = pp[i-1] 
    ∇ₚF = zeros(Mp1,D,J)
    dx1 = zeros(D, J)
    for m = 1:Mp1
        @views x = X[m,:]
        t = tt[m]
        ∇ₚf!(dx1, x, pᵢ₋₁, t)
        @views ∇ₚF[m,:,:] .= dx1
    end
    Φ_∇ₚF = zeros(K, D, J)
    @tullio Φ_∇ₚF[k,d,j] = Φ[k,m] * ∇ₚF[m,d,j] 
    Φ_∇ₚF = reshape(Φ_∇ₚF, K*D, J)
    Rem = remainder(pᵢ₋₁)
    Φ_Rem = reshape(Φ*Rem, K*D)
    c = Φ_∇ₚF \ Φ_Rem
    @info "- c = $(c')"
    pp[i] = pᵢ₋₁ - c
    # pp[i] = phat_MAP - c
    relErr_cor = norm(pp[i] - pstar) / norm(pstar)
    @info @sprintf "- relErr %.4g" relErr_cor
end
phat_cor = pp[end]
relErr_prior = norm(phat_prior - pstar) / norm(pstar)
relErr_MLE = norm(phat_MLE - pstar) / norm(pstar)
relErr_MAP = norm(phat_MAP - pstar) / norm(pstar)
relErr_cor = norm(phat_cor - pstar) / norm(pstar)
@info @sprintf "Relation coefficient error with prior      %.4g" relErr_prior
@info @sprintf "Relation coefficient error with MLE        %.4g" relErr_MLE
@info @sprintf "Relation coefficient error with MAP        %.4g" relErr_MAP
@info @sprintf "Relation coefficient error with correction %.4g" relErr_cor
r = similar(pstar)
offset =20
pp1 = [[p1, pstar[2]] for p1 in pstar[1]-offset:0.1:pstar[1]+offset]
pp2 = [[pstar[1], p2] for p2 in pstar[2]-offset:0.1:pstar[2]+offset]
p1 = plot(
    [
        scatter(
            x = pstar[1]-offset:0.1:pstar[1]+offset,
            y = [norm(r_p!(r,p1))^2 for p1 in pp1],
            name="p₁",
            legendgroup="p"
        ),
        scatter(
            x = [pstar[1]],
            y = [norm(r_p!(r,pstar))^2],
            name="p⋆₁",
            legendgroup="p⋆",
            mode="markers",
            marker_color="black",
            marker_symbol="star",
            marker_size=10,
        ),
        scatter(
            x = [phat_MAP[1]],
            y = [norm(r_p!(r,phat_MAP))^2],
            name="p̂₁",
            legendgroup="p̂",
            mode="markers",
            marker_color="green",
            marker_symbol="square",
            marker_size=10,
        ),
        scatter(
            x = [phat_cor[1]],
            y = [norm(r_p!(r,phat_cor))^2],
            name="p̃₁",
            legendgroup="p̃",
            mode="markers",
            marker_color="red",
            marker_symbol="circle",
            marker_size=10,
        )
    ],
    Layout(
        yaxis_title_text="loss", xaxis_title_text="p₁",
    )
)
p2 = plot(
    [
        scatter(
            x = pstar[2]-offset:0.1:pstar[2]+offset,
            y = [norm(r_p!(r, p2))^2 for p2 in pp2],
            name="p₂",
            legendgroup="p"
        ),
        scatter(
            x = [pstar[2]],
            y = [norm(r_p!(r,pstar))^2],
            name="p⋆",
            legendgroup="p⋆",
            mode="markers",
            marker_color="black",
            marker_symbol="star",
            marker_size=10,
        ),
        scatter(
            x = [phat_MAP[2]],
            y = [norm(r_p!(r,phat_MAP))^2],
            name="p̂₂",
            legendgroup="p̂",
            mode="markers",
            marker_color="green",
            marker_symbol="square",
            marker_size=10,
        ),
        scatter(
            x = [phat_cor[2]],
            y = [norm(r_p!(r,phat_cor))^2],
            name="p̃₂",
            legendgroup="p̃",
            mode="markers",
            marker_color="red",
            marker_symbol="circle",
            marker_size=10,
        )
    ],
    Layout(
        yaxis_title_text="loss", 
        xaxis_title_text="p₂"
    )
)
display([p1; p2])
##
##
function compareOptimizationLandscapes(xx,yy)
    pp = make_subplots(cols=3)
    for (i, costFun) in enumerate([wendyProb.priorLoss, wendyProb.wnll, wendyProb.wnlp])
        f = costFun.f;
        global ff = zeros(length(yy), length(xx))
        for (n, p1) in enumerate(xx), (m,p2) in enumerate(yy)
            ff[m,n] = try 
                log10(f([p1,p2]) + 100)
            catch 
                1e6 
            end
        end
        ixNaN = findall(isnan.(ff) .|| isinf.(ff))
        ff[ixNaN] .= 1e6
        legendgrouptitle, phat,c_x,c_y,c_len,visible,zmin,zmax = if i == 1
            "Prior", phat_prior, .25, 0.5,1, true,1.98,2.05
        elseif i==2
            "Likelihood", phat_MLE, .6, 0.5,1, false,2.5,4.2
        else
            "Posterior", phat_MAP, 1, 0.3,.6, true,2.5,4.2
        end
        add_trace!(pp,heatmap(x=xx, y=yy, z=ff,zmin=zmin,zmax=zmax, legendgroup=legendgrouptitle, legendgrouptitle_text=legendgrouptitle, showlegend=false, colorbar=attr(x=c_x,y=c_y,len=c_len), showscale=visible), row=1, col=i)
        add_trace!(pp,scatter(mode="markers", marker_opacity=1, x=[p₀[1]], y=[p₀[2]], showlegend=i==1,name="\$\\mathbf{p}^{(0)}\$", marker=attr(symbol="square", color="black", size=20)), row=1, col=i)
        add_trace!(pp,scatter(mode="markers", marker_opacity=0.8, x=[phat[1]], y=[phat[2]], showlegend=i==1, name="\$\\hat{\\mathbf{p}}\$", marker=attr(symbol="diamond", color="red", size=20)), row=1, col=i)
        add_trace!(pp,scatter(mode="markers", marker_opacity=.8, x=[pstar[1]], y=[pstar[2]], showlegend=i==1, name="\$\\mathbf{p}^*\$", marker=attr(symbol="star", color="green", size=20)), row=1, col=i)
    end
    pp
end
xx = 1:0.05:4
yy = 1:0.05:12
p2 = compareOptimizationLandscapes(xx,yy)
add_trace!(p2,scatter(mode="markers", marker_opacity=.8, x=[phat_cor[1]], y=[phat_cor[2]], showlegend=true, name="\$\\hat{\\mathbf{p}}_{\\mathrm{cor}}\$", marker=attr(symbol="circle", color="blue", size=20)), row=1, col=3)
##
relayout!(p2, 
    title=attr(
        text="Comparing Optimization Landscapes",
        font_size=40,
        xanchor="center",
        x=0.5
    ),
    yaxis=attr(title_font_size=30, range=[minimum(yy), maximum(yy)],title_text="p₁"),
    yaxis2=attr(title_font_size=30, range=[minimum(yy), maximum(yy)],),
    yaxis3=attr(title_font_size=30, range=[minimum(yy), maximum(yy)],),
    xaxis=attr(title_font_size=30, domain=[0,.30], range=[minimum(xx),maximum(xx)],title_text="p₂<br>Prio}"),
    xaxis2=attr(title_font_size=30, domain=[.35,.65], range=[minimum(xx),maximum(xx)],title_text="p₂<br>Likelihood" ),
    xaxis3=attr(title_font_size=30, domain=[.7,1], range=[minimum(xx),maximum(xx)],title_text="p₂<br>Posterior"),
    legend=attr(
        x=1.05,
        y=.95,
        xref="paper", 
        yref="paper",
        xanchor="right",
        yanchor="top",
        font=(
            family="sans-serif",
            size=35,
            color="#000"
        ),
        bgcolor="#E2E2E2",
        bordercolor= "#636363",
        borderwidth=1
    ),
    coloraxis=attr(colorbar=attr(y=0.3, len=.7))
)
savefig( p2, joinpath(FIG_DIR,"LogisticGrowth_OptimizationLandscape.pdf"),height=400, width=1200)
p2
##
# global this, dx, ϵ
# macro remainder_m(this, dx, ϵ, α)
#     return quote 
#         local val_α = $α
#         # @info "val_α = $val_α"
#         local args1 = [Symbol("d$i") for i = 1:val_α]
#         @info "args1 = $args1"
#         local args2 = [(:ref, esc(ϵ), Symbol("d$i")) for i = 1:val_α]
#         @info "args2 = $args2"
#         expr = esc(:(@tullio esc(this)[d] = prod(esc(dx)[d, $(args1...)], $(args2...))))
#         Meta.show_sexpr(expr)
#         println()
#         expr
#     end
# end
#
# i = 3
# α = 2
# m = 1
# ∇ₓᵅf = ∇ₓᵅf_list[i]
# dx = zeros(Tuple(D for _ = 1:α+1))
# @views x = X[m,:]   
# @views t = tt[m]
# ∇ₓᵅf(dx, x, phat, t)
# @views ϵ = E[m,:]
# #
# this = zeros(D)
# @info "Autogen macro"
# @time @remainder_m this dx ϵ α
# @info "- this = $this"
# ##
# this = zeros(D)
# @info "Hand written macro"
# @time @tullio this[d] = dx[d,d1,d2]*ϵ[d1]*ϵ[d2]
# @info "- this = $this"
