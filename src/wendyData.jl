##
abstract type WENDyData{lip,DistType<:Distribution} end 
## Struct to store simulated data and all params, IC and the like
struct SimulatedWENDyData{lip,DistType}<:WENDyData{lip,DistType}
    name::String
    f!::Function
    initCond::AbstractVector{<:Real}
    tRng::NTuple{2,<:Real}
    wTrue::AbstractVector{<:Real}
    wRng::AbstractVector{<:Tuple}
    ## specific parameters for this example
    params::WENDyParameters
    Mp1::Int
    file::String 
    tt_full::AbstractVector{<:Real}
    U_full::AbstractMatrix{<:Real}
    ## subsampled
    tt::Ref{Union{Nothing,AbstractVector{<:Real}}}
    U::Ref{Union{Nothing,AbstractMatrix{<:Real}}}
    Ustar::Ref{Union{Nothing,AbstractMatrix{<:Real}}}
    sigTrue::Ref{Union{Nothing,AbstractVector{<:Real}}}
    noise::Ref{Union{Nothing,AbstractMatrix{<:Real}}}
end
## Constructor
function SimulatedWENDyData( 
    name::String,
    f!::Function,
    initCond::AbstractVector{<:Real},
    tRng::NTuple{2,<:Real},
    wTrue::AbstractVector{<:Real},
    wRng::AbstractVector{<:Tuple}, 
    params::WENDyParameters;
    Mp1::Union{Nothing, Int}=1025,
    dt::Union{Nothing, Real}=nothing,
    linearInParameters::Val{lip}=Val(false), #linearInParameters
    noiseDist::Val{DistType}=Val(Normal), #distributionType
    file::Union{String, Nothing}=nothing,
    forceOdeSolve::Bool=true,
    ll::LogLevel=Warn
) where {lip, DistType<:Distribution}
    with_logger(ConsoleLogger(stderr, ll)) do 
        @assert DistType == Normal || DistType == LogNormal "Only LogNormal and Normal Noise distributions are supported"
        @assert !(isnothing(Mp1) && isnothing(dt)) "One must either set the dt or number of time points"
        isnothing(file) && (file = joinpath(@__DIR__, "../data/$name.bson"))
        if !isnothing(dt) 
            Mp1 = Int(floor((tRng[end]- tRng[1]) / dt)) + 1
        end
        @info "  Generating data for ode"
        odeprob = ODEProblem(f!, initCond, tRng, wTrue)
        dt = (tRng[end]-tRng[1]) / (Mp1-1)
        sol = solve_ode(odeprob, Rosenbrock23(); 
            # reltol=1e-12, abstol=1e-12, 
            saveat=dt, verbose=false
        )
        tt_full = sol.t
        U_full = reduce(vcat, sol.u')
        @assert length(tt_full) == Mp1
        @assert !any(isnan.(U_full)) 
        # this may be modified by the log normal handeling or from dt being used instead of Mp1
        tRng = (tt_full[1], tt_full[end])
        return SimulatedWENDyData{lip,DistType}(
            name, f!, initCond, tRng, wTrue, wRng, params, 
            Mp1, file, tt_full, U_full,
            # subsampled and noisey data needs to be simulated with the simulate! function
            nothing, nothing, nothing, nothing, nothing 
        )
    end
end
## Change data's lip or dist type
function SimulatedWENDyData(data::SimulatedWENDyData{old_lip, old_DistType}, ::Val{new_lip}=Val(nothing), ::Val{new_DistType}=Val(nothing)) where {old_lip, old_DistType, new_DistType, new_lip}
    lip = isnothing(new_lip) ? old_lip : new_lip 
    DistType = isnothing(new_DistType) ? old_DistType : new_DistType 
    SimulatedWENDyData(
        data.name, data.odeprob, data.f!, data.initCond, data.tRng, data.wTrue; 
        Mp1=data.Mp1, file=data.file, linearInParameters=Val(lip), noiseDist=Val(DistType)
    )
end
## Simulated data for testing
@kwdef struct SimulationParameters
    timeSubsampleRate::Int = 2
    seed::Union{Int,Nothing} = nothing
    noiseRatio::AbstractFloat = 0.01
    isotropic::Bool = true
    corruptU0::Bool = true
end
## Seed for reproducibility if asked 
function _seed(seed::Union{Nothing,Int})
    if (!isnothing(seed) && seed > 0) 
        Random.seed!(seed)
        @info "  Seeeding the noise with value $(seed)"
    else 
        @info "  no seeding"
    end
end
## Normal
function generateNoise(U_exact::AbstractMatrix{<:Real}, noiseRatio::Real, seed::Union{Nothing, Int}, isotropic::Bool, ::Val{Normal}) 
    @info "     Using Normal Noise"
    @assert noiseRatio >= 0 "Noise ratio must be possitive"
    Mp1, D = size(U_exact)
    if noiseRatio == 0 
        return U_exact, zeros(Mp1, D), zeros(D)
    end
    # seed if asked
    _seed(seed)
    # corrupt data with noise
    U = similar(U_exact)
    noise = similar(U_exact)
    σ = zeros(D)
    mean_signals = isotropic ? mean(U_exact .^2) .* ones(D) : mean(U_exact .^2, dims=1) 
    for (d,signal) in enumerate(mean_signals)
        σ[d] = noiseRatio*sqrt(signal) 
        dist = Normal(0, σ[d])
        noise[:,d] .= rand(dist, Mp1)
        U[:,d] .= U_exact[:,d] + noise[:,d]
    end
    return U,noise,σ
end
## Log normal 
function generateNoise(U_exact::AbstractMatrix{<:Real}, noiseRatio::Real, seed::Union{Nothing, Int}, ::Bool, ::Val{LogNormal})
    @info "     Using LogNormal Noise"
    @assert noiseRatio > 0 "Noise ratio must be possitive"
    Mp1,D = size(U_exact)
    if noiseRatio == 0 
        return U_exact, Matrix{Float64}(I,Mp1,Mp1), zeros(D)
    end
    # seed if asked
    _seed(seed)
    # corrupt data with noise
    U = similar(U_exact)
    noise = similar(U_exact)
    σ = zeros(D)
    σ .= noiseRatio
    for d in 1:D
        dist = LogNormal(0,σ[d]) # lognormal with logmean of 0, and variance of σ_d
        noise[:,d] .= rand(dist,Mp1)
        U[:,d] .= U_exact[:,d] .* noise[:,d]
    end
    return U,noise,σ
end
## Convience function for calling with SimParams struct
function generateNoise(U_exact::AbstractMatrix{<:Real}, simParams::SimulationParameters, ::Val{DistType}) where DistType<:Distribution 
    generateNoise(U_exact, simParams.noiseRatio, simParams.seed, simParams.isotropic, Val(DistType))
end
## add noise and subsample data
function simulate!(data::SimulatedWENDyData{lip,DistType}, params::SimulationParameters; ll::LogLevel=Debug) where {lip, DistType<:Distribution}
    with_logger(ConsoleLogger(stdout,ll)) do 
        @info "Simulating ..."
        @info "  Subsample data with rate $(params.timeSubsampleRate)"
        tt = data.tt_full[1:params.timeSubsampleRate:end]
        Ustar = data.U_full[1:params.timeSubsampleRate:end,:]
        @info "  Mp1_full,D = $(size(data.U_full))"
        @info "  Mp1,D = $(size(Ustar))"
        @info "  Corrupting with noise from Distribution $DistType and noise ratio $(params.noiseRatio)"
        U, noise, sigTrue = generateNoise(Ustar, params, Val(DistType))
        @info "   sigTrue = $sigTrue"
        if !params.corruptU0
            @info "   Reseting the initial condition to be uncorrupted"
            U[1,:] .= Ustar[1,:]
        else 
            @info "   Note: the initial condition has been corrupted" 
        end
        @assert !any(isnan.(U)) "No data should be corrupted with NaN values"
        data.tt[]      = tt
        data.U[]       = U
        data.Ustar[]   = Ustar
        data.sigTrue[] = sigTrue
        data.noise[]   = noise
        nothing
    end
end 
## Extend constructors to play nice with simulated data 
import WENDy.WENDyProblem
## Convience Wrapper
function WENDyProblem(ex::WENDyData{lip, DistType}; ll::LogLevel=Warn, kwargs...) where {lip,DistType<:Distribution}
    @assert !isnothing(ex.tt[]) "tt is nothing, please call the simulate! function to generate noise"
    @assert !isnothing(ex.U[]) "U is nothing, please call the simulate! function to generate noise"
    WENDyProblem(ex.tt[], ex.U[], ex.f!, length(ex.wTrue); linearInParameters=Val(lip), noiseDist=Val(DistType), params=ex.params, constraints=nothing, ll=ll, kwargs...)
end
