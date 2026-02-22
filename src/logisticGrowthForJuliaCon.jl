@info "Loading Dependencies"
includet("util.jl")
includet("../plot/plotBase.jl")
using Distributions, Random
## define priors
Random.seed!(1)
T = 10.0
α = 3
s = 5 / ((α / (1 + α))^(1 / α ))
priors = [2*(Beta(2,5) + .8), Frechet(α,s)]
p_mode = [mode(prior) for prior in priors]
pstar = p_mode + [.25, 2]
p₀ = [rand(prior) for prior in priors]
# @info "argmax(prior) = $(p_mode)"
@info "p* = $pstar"
@info "p₀ = $p₀"
## Build ODE and simulate noise
function LOGISTIC_f!(du, u, w, t)
    du[1] = w[1] * u[1] - w[2] * u[1]^2
    nothing
end
noiseDist = Val(LogNormal)
params = WENDyParameters(;Kmax=1000)
ex = SimulatedWENDyData(
    "logisticGrowth", 
    LOGISTIC_f!,
    [0.01], # u₀
    (0.0, T), # (t₀, tₘ)
    pstar, # p*
    [
        (0.0,10.0),
        (0.0,10.0)
    ],
    params;
    linearInParameters=Val(true),
    noiseDist=noiseDist,
    Mp1=101
);
simParams = SimulationParameters(
    seed=1, 
    timeSubsampleRate=1,
    noiseRatio=.2,
    corruptU0=true
) 
simulate!(ex, simParams, ll=Warn)
tt=ex.tt[]
U=ex.U[]
J = length(p₀)
U_exact= ex.Ustar[] 
dt = diff(tt)
dt = dt[1];
## Get FS Error 
function fsErr(phat)
    u0star = ex.Ustar[][1,:]
    tRng = (tt[1], tt[end])
    odeprob = ODEProblem(LOGISTIC_f!, u0star, tRng, phat)
    sol = solve_ode(odeprob; 
        reltol=params.optimReltol, abstol=params.optimAbstol, 
        saveat=dt, verbose=false
    )
    Uhat = reduce(vcat, um' for um in sol.u)[:]
    fsRelErr = norm(Uhat - U_exact) / norm(U_exact)
    return fsRelErr, Uhat
end
@info "========================================================================="
## Output Error for comparison point
oeProb = WENDy.OutputErrorProblem(tt,U,LOGISTIC_f!, J)
phat_OE = WENDy.solve(oeProb, p₀)
u0hat_OE = phat_OE[J+1:end]
phat_OE = phat_OE[1:J]
@info "Output Error"
relErr_OE = norm(phat_OE - pstar) / norm(pstar)
@info @sprintf "  Relation Coefficient error %.2g" relErr_OE
fsRelErr_OE, Uhat_OE = fsErr(phat_OE)
@info @sprintf "  Average Relative Forward Solver Error %.2g" fsRelErr_OE
## Build and solve wendy problem 
wendyProb = WENDyProblem(ex; priors=priors, ll=Info)
phat_MLE, iters, P = solve(wendyProb, p₀, ex.params; alg=WENDy.IP(), costFun=:wnll,return_wits=true);
# relative error 
relErr_MLE = norm(phat_MLE - pstar) / norm(pstar)
@info "WENDy MLE"
@info @sprintf "  Relation Coefficient error %.2g" relErr_MLE
fsRelErr_MLE, Uhat_MLE = fsErr(phat_MLE)
@info @sprintf "  Average Relative Forward Solver Error %.2g" fsRelErr_MLE
## covariance
S = WENDy.Covariance(wendyProb.data, params)
∇r = WENDy.JacobianResidual(wendyProb.data, params)
Ghat = ∇r(phat_MLE)  
Shat = S(phat_MLE; transpose=false, doChol=false)
cov_phat = (Ghat \ Shat) / (Ghat')
std_phat = sqrt.(diag(cov_phat))
@info "2σ confidence interval"
@info " p̂₁ : $(phat_MLE[1]) ± $(2*std_phat[1]) "
@info " p̂₂ : $(phat_MLE[2]) ± $(2*std_phat[2]) "
## Visual Demonstration
trs = AbstractTrace[]
push!( 
    trs,
    scatter(
        x=tt,
        y=U[:,1],
        name="\$\\{(\\mathrm{t}_m, \\mathrm{u}_m)\\}\$", 
        mode="markers" ,
        marker_color=PLOTLYJS_COLORS[1], 
        marker_opacity=0.5, 
        marker_size=10, 
        # legendgroup=d,
        # legendgrouptitle_text="State $d"
    )
)

push!(
    trs, 
    scatter(
        x=tt,
        y=Uhat_MLE,
        name="\$u(\\hat{\\mathbf{p}})\$",
        mode="lines",
        line=attr(
            color=CU_BOULDER_COLORS[2],
            width=5,
        )
    )
)

# push!(
#     trs, 
#     scatter(
#         x=tt,
#         y=Uhat_OE,
#         name="\$u_{\\mathrm{OE-LS}}(\\hat{\\mathbf{p}})\$",
#         mode="lines",
#         line=attr(
#             color=CU_BOULDER_COLORS[3],
#             width=5,
#         )
#     )
# )

push!( 
    trs,
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
        # legendgroup=d,
    )
)
p1a = plotjs(
    trs, 
    Layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=attr(
            text="Logistic Growth", 
            x=0.5,
            xanchor="center",
            font_size=30
        ),
        # yaxis_type=yaxis_type,
        showlegend=true, 
        xaxis=attr(
            title=attr(
                text="\$t\$",
                font_size=20
            ),
            tick_font_size=15,
            showgrid=true, 
            zeroline=true
        ),
        yaxis=attr(
            title=attr(
                text="\$u\$",
                font_size=20
            ),
            tick_font_size=15,
            showgrid=true, 
            zeroline=true
        ),
        legend=attr(
            # x=.925,
            y=0.5,
            yanchor="center",
            font=(
                family="sans-serif",
                size=20,
                color="#000"
            ),
            bgcolor="#E2E2E2",
            bordercolor= "#636363",
            entrywidth= 150,        # Manually set this based on your longest equation
            entrywidthmode= "pixels", 
            borderpad= 20,          # Give extra "buffer" for tall fractions or exponents
            borderwidth= 1
        ),
        # margin_r= 150,
        hovermode="x unified"
    )
)



display(p1a)
# Save fig to file

savefig(
    p1a, 
    joinpath("/Users/niru8088/scratch/bayesWENDy/WENDy.jl/paper","LogisticGrowth_TrajectoryWithData.pdf"),
    height=400, 
    width=600
)