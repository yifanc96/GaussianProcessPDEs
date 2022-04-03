using PyPlot
include("MCMC.jl")

function double_well(sparam, θ::Array{Float64,1})
    return [θ[1]^2]
end

function d_double_well(sparam, θ::Array{Float64,1})
    return [θ[1]^2], reshape([2θ[1]],1,1)
end

# function mixture(sparam, θ::Array{Float64,1})
#     if sparam.cond == 1
#         return [0.9 * θ[1]]
#     elseif sparam.cond == 2
#         return [0.1 * θ[1]^3]
#     else
#         error("cond : ", cond, " is not defined")
#     end
# end

struct Setup_Param
    Ny::Int64
    Nθ::Int64
end
s_param = Setup_Param(1, 1)

y = [1.0]  
Σ_η = reshape([0.1^2], 1, 1)
μ0 = [2.0] 
Σ0 = reshape([5.0^2], 1, 1)
ln_posterior(θ) = log_bayesian_posterior(s_param, θ, double_well, 
    y,  Σ_η, μ0, Σ0)
d_ln_posterior(θ) = d_log_bayesian_posterior(s_param, θ, d_double_well, 
y,  Σ_η, μ0, Σ0)

# "MALA"  "RWMCMC"  "SVGD"
method = "SVGD" 
if method == "RWMCMC"
    θ0 = [-0.3]
    step_length, n_ite = 0.2, 10^6
    θs = RWMCMC_Run(ln_posterior, θ0, step_length, n_ite) 
elseif method == "MALA"
    θ0 = [-0.3]
    step_length, n_ite = 0.002, 10^6
    θs = MALA_Run(d_ln_posterior, θ0, step_length, n_ite) 
elseif method == "SVGD"
    J, n_ite = 100, 1000
    θs0 = randn(J,1)
    θs = SVGD_Run(θs0, d_ln_posterior, n_ite)
end

# Plot
PyPlot.figure()
PyPlot.hist(θs[:, 1], bins=div(size(θs[:, 1], 1),10), density = true, histtype = "step", label = method)
θθ = reshape(Array(LinRange(-2, 2, 1000)), (1000, 1))
ρρ = zeros(size(θθ, 1)) 
for i =1:size(θθ, 1)
    ρρ[i]= ln_posterior(θθ[i,:])
end
ρρ.= exp.(ρρ)
ρρ /= (sum(ρρ)*(θθ[2,1] - θθ[1,1]))
PyPlot.plot(θθ, ρρ, "--", label="Reference")
PyPlot.legend()