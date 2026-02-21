## See Wendy paper

function LOGISTIC_f!(du, u, w, t)
    du[1] = w[1] * u[1] - w[2] * u[1]^2
    nothing
end
LOGISTIC_TRNG = (0.0, 10.0)
LOGISTIC_INIT_COND = [0.01]
LOGISTIC_WSTAR = [1.0, 1.0]
LOGISTIC_WRNG = [
    (0.0,10.0),
    (0.0,10.0)
]
LOGISTIC_PARAMS = WENDyParameters(;Kmax=1000)
LOGISTIC_GROWTH = SimulatedWENDyData(
    "logisticGrowth", 
    LOGISTIC_f!,
    LOGISTIC_INIT_COND,
    LOGISTIC_TRNG,
    LOGISTIC_WSTAR,
    LOGISTIC_WRNG,
    LOGISTIC_PARAMS;
    linearInParameters=Val(true),
    noiseDist=Val(Normal)
);
