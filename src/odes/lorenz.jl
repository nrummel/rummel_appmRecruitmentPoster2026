## https://en.wikipedia.org/wiki/Lorenz_system#Julia_simulation
function LORENZ_f!(du, u, w, t)
    du[1] = w[1] * (u[2] - u[1])
    du[2] = u[1] * (w[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - w[3] * u[3]
    nothing
end
LORENZ_INIT_COND = [2; 1; 1]
LORENZ_TRNG = (0.0, 10.0)
# [(0.0, 3.0), (0.0, 5.0), (0.0, 10.0)] # orbited fixed points once 
# 1000 / 1024
LORENZ_WSTAR = [10; 28; 8/3]
LORENZ_WRNG = [
    (0.0,20.0),
    (0.0,35.0),
    (0.0, 5.0)
]
LORENZ_PARAMS = WENDyParameters(
    optimMaxiters  = 200,
    optimTimelimit = 200.0,
    radiusMinTime  = 0.02,
    radiusMaxTime  = 5.,
    Kᵣ             = 200,
    Kmax           = 500,
)

LORENZ = SimulatedWENDyData(
    "lorenz", 
    LORENZ_f!,
    LORENZ_INIT_COND,
    LORENZ_TRNG,
    LORENZ_WSTAR,
    LORENZ_WRNG,
    LORENZ_PARAMS;
    linearInParameters=Val(false),
    noiseDist=Val(Normal),
    forceOdeSolve=true,
    dt=0.01
);