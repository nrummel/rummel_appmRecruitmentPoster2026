## See Wendy paper
function HINDMARSH_f!(du, u, w, t)
    du[1] = w[1] * u[2] - w[2] * u[1]^3 + w[3] * u[1]^2 - w[4] * u[3] 
    du[2] = w[5] - w[6] * u[1]^2 - w[7] * u[2] 
    du[3] = w[8] * u[1] + w[9] - w[10] * u[3]
    nothing
end
HINDMARSH_INIT_COND = [-1.31; -7.6; -0.2]
HINDMARSH_TRNG = (0.0, 10.0)
HINDMARSH_WSTAR = [
    10,         # w1            
    10,         # w2 = w7 ?       
    30,         # w3           
    10,         # w4            
    10,         # w5            
    50,         # w6            
    10,         # w7      
    0.04,       # w8      
    0.0319,     # w9      
    0.01        # w10      
]

HINDMARSH_WRNG = [
    (0.0,20.0),    # w1            
    (0.0,20.0),    # w2         
    (0.0,60.0),    # w3           
    (0.0,20.0),    # w4            
    (0.0,20.0),    # w5            
    (0.0,100.0),   # w6            
    (0.0,20.0),    # w7      
    (0.0,1.0),     # w8      
    (0.0,1.0),     # w9      
    (0.0,1.0)      # w10      
]
HINDMARSH_PARAMS = WENDyParameters(
    radiiParams = [4,8,16],
    optimTimelimit = 500.0
)
HINDMARSH_ROSE = SimulatedWENDyData(
    "hindmarshRose", 
    HINDMARSH_f!,
    HINDMARSH_INIT_COND,
    HINDMARSH_TRNG,
    HINDMARSH_WSTAR,
    HINDMARSH_WRNG,
    HINDMARSH_PARAMS;
    linearInParameters=Val(true),
    noiseDist=Val(Normal)
);