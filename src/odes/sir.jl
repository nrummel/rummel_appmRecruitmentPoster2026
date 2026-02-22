# From Mathematical Biology: An Introduction with Maple and Matlab on page 318
# const SIR_H = 1.99
function SIR_f!(du, u, w, t)
    β = (w[1] * exp(-w[1] * w[2])) / (1 - exp(-w[1] * w[2]))
    du[1] = -w[1] * u[1] + w[3] * u[2] + β * u[3] 
    du[2] = w[1] * u[1] - w[3] * u[2] - w[4] * (1 - exp(-w[5]  * t^2)) * u[2] 
    du[3] = w[4] * (1 - exp(-w[5]  * t^2)) * u[2] - β * u[3]
end
h = 0.2
τ = 1.5
ρ = 0.074
R = 0.113
v = 0.0024
SIR_WSTAR = [h,τ,ρ,R,v]
SIR_WRNG = [
    (1e-4,1.0), 
    (1e-4,2.0),
    (1e-4,1.0),
    (1e-4,1.0),
    (1e-4,1.0),
]
SIR_INIT_COND = [1,0,0]
SIR_TRNG = (0.0, 50.0)
SIR_PARAMS = WENDyParameters(
    radiusMinTime  = 0.1,
    radiusMaxTime  = 25.0,
    Kmax           = 300
)
SIR = SimulatedWENDyData(
    "sir", 
    SIR_f!,
    SIR_INIT_COND,
    SIR_TRNG,
    SIR_WSTAR,
    SIR_WRNG,
    SIR_PARAMS;
    linearInParameters=Val(false),
    noiseDist=Val(LogNormal),
    forceOdeSolve=true
);