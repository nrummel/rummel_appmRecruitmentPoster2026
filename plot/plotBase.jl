using JLD2, Latexify, ColorSchemes, Colors, LaTeXStrings # for plotting and saving captions to tex
using PlotlyJS, Statistics, Printf
using PlotlyKaleido: restart as restart_kaleido
restart_kaleido(plotly_version = "2.35.2", mathjax = true) 
using PlotlyJS: savefig, scatter, Layout, AbstractTrace, attr
import PlotlyJS.plot as plotjs
Latexify.set_default(fmt = "%.4g")
##
FIG_DIR = joinpath(@__DIR__, "../../NonLinearWENDyPaper/figs")
DATA_DIR = joinpath(@__DIR__, "../data")
PLOTLYJS_COLORS = [
    colorant"#1f77b4",  # muted blue
    colorant"#ff7f0e",  # safety orange
    colorant"#2ca02c",  # cooked asparagus green
    colorant"#d62728",  # brick red
    colorant"#9467bd",  # muted purple
    colorant"#8c564b",  # chestnut brown
    colorant"#e377c2",  # raspberry yogurt pink
    colorant"#7f7f7f",  # middle gray
    colorant"#bcbd22",  # curry yellow-green
    colorant"#17becf"   # blue-teal
]
CU_BOULDER_COLORS = [
    colorant"#000000", # black 
    colorant"#CFB87C", # gold 
    colorant"#565A5C", # dark grey 
    colorant"#A2A4A3", # light grey 
]
algo2Color = (
    init=CU_BOULDER_COLORS[1],
    truth=CU_BOULDER_COLORS[3],
    oels=PLOTLYJS_COLORS[1],
    wls=PLOTLYJS_COLORS[2],
    wendy_irls=PLOTLYJS_COLORS[3],
    wendy_mle=CU_BOULDER_COLORS[2],
    hybrid=PLOTLYJS_COLORS[6],
)
algo2Disp = (
    init="p<sub>0</sub> Solution",
    truth="p<sup>*</sup> Solution",
    oels="OE-LS",#"Forward Solve Nonlinear Least Squares",
    wls="WLS", 
    wendy_irls="WENDy-IRLS",
    wendy_mle="WENDy-MLE",
    hybrid="Hybrid",
    # hybrid_wlsq_trustRegion="Hybrid WLS-WENDy"
)
ex2Disp = Dict(
    "sir"=>"SIR-TDI",
    "goodwin"=>"Goodwin",
    "robertson"=>"Robertson",
    "hindmarshRose"=>"Hindmarsh Rose", 
    "logisticGrowth"=>"Logistic Growth", 
    "multimodal"=>"Goodwin 2D", 
    "lorenz"=>"Lorenz System", 
);

function plotjs(prob::WENDyProblem, title::String="", file::Union{Nothing, String}=nothing, yaxis_type::String="linear", showNoiseData::Bool=true)
    plotjs(prob.U, prob.U_exact, prob.tt, title, file, yaxis_type, showNoiseData)
end
function plotjs(ex::SimulatedWENDyData, title::String="", file::Union{Nothing, String}=nothing, yaxis_type::String="linear", showNoiseData::Bool=true)
    plotjs(ex.U[], ex.Ustar[], ex.tt[], title, file, yaxis_type, showNoiseData)
end
function plotjs(U, U_exact, tt, title, file, yaxis_type, showNoiseData)
    _, D = size(U)
    trs = AbstractTrace[]
    for d in 1:D 
        showNoiseData && push!( 
            trs,
            scatter(
                x=tt,
                y=U[:,d],
                name="\$\\mathrm{u}_$d\$", 
                mode="markers" ,
                marker_color=PLOTLYJS_COLORS[d], 
                marker_opacity=0.8, 
                marker_size=5, 
                legendgroup=d,
                legendgrouptitle_text="State $d"
            )
        )
        push!( 
            trs,
            scatter(
                x=tt,
                y=U_exact[:,d],
                name="\$u_$d(t)\$", 
                mode="lines" ,
                line_dash="dash",
                line_width=5,
                line_color=CU_BOULDER_COLORS[d],
                legendgroup=d,
            )
        )
    end
    p = plotjs(
       trs, 
       Layout(
            title_text=title, 
            title_x=0.5,
            title_xanchor="center",
            yaxis_type=yaxis_type,
            showlegend=true, 
            xaxis_title="time (s)",
            legend=attr(
                # x=.925,
                y=0.5,
                yanchor="center",
                font=(
                    family="sans-serif",
                    # size=20,
                    color="#000"
                ),
                bgcolor="#E2E2E2",
                bordercolor= "#636363",
                borderwidth= 2,
            ),
            hovermode="x unified"
        )
    )
    !isnothing(file) && PlotlyJS.savefig(
        p,
        file;
        height=600,
        width=800
    )
    p
end