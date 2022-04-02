using PyPlot
include("MCMC.jl")

function double_well(sparam, θ::Array{Float64,1})
    return [θ[1]^2]
end

function mixture(sparam, θ::Array{Float64,1})
    if cond == 1
        return [0.9 * θ[1]]
    elseif cond == 2
        return [0.1 * θ[1]^3]
    else
        error("cond : ", cond, " is not defined")
    end
end

struct Setup_Param
    Ny::Int64
    Nθ::Int64
end
s_param = Setup_Param(1, 1)

y = [1.0]  
Σ_η = 0.01*ones(1,1)
μ0 = [0.0] 
Σ0 = ones(1,1)
posterior(θ) = log_bayesian_posterior(s_param, θ, double_well, 
    y,  Σ_η, μ0, Σ0)

θ0 = [-0.3]
step_length, n_ite = 0.2, 1000000
θs = RWMCMC_Run(posterior, θ0, step_length, n_ite) 

PyPlot.hist(θs[:, 1], bins=100)