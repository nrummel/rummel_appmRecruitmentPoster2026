## See https://tspace.libraru.utoronto.ca/bitstream/1807/95761/3/Calver_Jonathan_J_201906_PhD_thesis.pdf#page=48
function GOODWIN_f!(du,u,w,t)
    #= Fix 2.15 (dissociation constant) =#
    du[1] = w[1] / (2.15 + w[3]*u[3]^w[4]) - w[2] * u[1]
    du[2] = w[5]*u[1] - w[6]*u[2]
    du[3] = w[7]*u[2] - w[8]*u[3]
    nothing
end
GOODWIN_TRNG = (0.0, 80.0)            
GOODWIN_INIT_COND = [0.3617, 0.9137, 1.3934]
GOODWIN_WSTAR = [3.4884, 0.0969, 1.0, 10.0, 0.0969, 0.0581, 0.0969, 0.0775]
GOODWIN_WRNG = [
    (1.0,5.0), # w1 ligand concentration
    (0.0,0.2), # w2
    (0.0,2.0),  # w3 
    (5.0, 15.0), # w4 hill coefficient
    (0.0,0.2),  # w5
    (0.0,0.2),  # w6
    (0.0,0.2),  # w7
    (0.0,0.2),  # w8
]
GOODWIN_PARAMS = WENDyParameters( 
    radiusMinTime  = 0.1,
    radiusMaxTime  = 20.0,
    Kmax           = 300,
    optimTimelimit = 500.0
)

GOODWIN = SimulatedWENDyData(
    "goodwin", 
    GOODWIN_f!,
    GOODWIN_INIT_COND,
    GOODWIN_TRNG,
    GOODWIN_WSTAR,
    GOODWIN_WRNG,
    GOODWIN_PARAMS;
    linearInParameters=Val(false), # NONLINEAR!
    noiseDist=Val(LogNormal),
    forceOdeSolve=true
);
