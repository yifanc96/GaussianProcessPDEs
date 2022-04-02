using Statistics
using LinearAlgebra
using Distributions
using Random

function log_bayesian_posterior(s_param, θ::Array{Float64,1}, forward::Function, 
    y::Array{Float64,1},  Σ_η::Array{Float64,2}, 
    μ0::Array{Float64,1}, Σ0::Array{Float64,2})

    Gu = forward(s_param, θ)
    Φ = - 0.5*(y - Gu)'/Σ_η*(y - Gu) - 0.5*(θ - μ0)'/Σ0*(θ - μ0)
    return Φ

end


function d_log_bayesian_posterior(s_param, θ::Array{Float64,1}, forward::Function, 
    y::Array{Float64,1},  Σ_η::Array{Float64,2}, 
    μ0::Array{Float64,1}, Σ0::Array{Float64,2})

    Gu, dGu = forward(s_param, θ)
    dΦ = dGu'* (Σ_η\(y - Gu)) - Σ0\(θ - μ0)
    return dΦ

end

function log_likelihood(s_param, θ::Array{Float64,1}, forward::Function, 
    y::Array{Float64,1},  Σ_η::Array{Float64,2})

    Gu = forward(s_param, θ)
    Φ = - 0.5*(y - Gu)'/Σ_η*(y - Gu)
    return Φ

end


"""
When the density function is Φ/Z, 
The f_density function return log(Φ) instead of Φ
"""
function RWMCMC_Run(log_bayesian_posterior::Function, θ0::Array{FT,1}, step_length::FT, n_ite::IT; seed::IT=11) where {FT<:AbstractFloat, IT<:Int}
    
    N_θ = length(θ0)
    θs = zeros(Float64, n_ite, N_θ)
    fs = zeros(Float64, n_ite)
    
    θs[1, :] .= θ0
    fs[1] = log_bayesian_posterior(θ0)
    
    Random.seed!(seed)
    for i = 2:n_ite
        θ_p = θs[i-1, :] 
        θ = θ_p + step_length * rand(Normal(0, 1), N_θ)
        
        
        fs[i] = log_bayesian_posterior(θ)
        α = min(1.0, exp(fs[i] - fs[i-1]))
        if α > rand(Uniform(0, 1))
            # accept
            θs[i, :] = θ
            fs[i] = fs[i]
        else
            # reject
            θs[i, :] = θ_p
            fs[i] = fs[i-1]
        end

    end
    
    return θs 
end



"""
When the density function is Φ/Z, 
The f_density function return log(Φ) instead of Φ
"""
function PCN_Run(log_likelihood::Function, θ0::Array{FT,1}, θθ0_cov::Array{FT,2}, β::FT, n_ite::IT; seed::IT=11) where {FT<:AbstractFloat, IT<:Int}
    
    N_θ = length(θ0)
    θs = zeros(Float64, n_ite, N_θ)
    fs = zeros(Float64, n_ite)
    
    θs[1, :] .= θ0
    fs[1] = log_likelihood(θ0)
    
    Random.seed!(seed)
    for i = 2:n_ite
        θ_p = θs[i-1, :] 
        θ = sqrt(1 - β^2)*θ_p + β * rand(MvNormal(zeros(size(θ0)), θθ0_cov))
        
        
        fs[i] = log_likelihood(θ)
        α = min(1.0, exp(fs[i] - fs[i-1]))
        if α > rand(Uniform(0, 1))
            # accept
            θs[i, :] = θ
            fs[i] = fs[i]
            @info "accept i = ", i
        else
            # reject
            θs[i, :] = θ_p
            fs[i] = fs[i-1]
        end

    end
    
    return θs 
end






# Stein variational gradient descent 
function SVGD_Kernel(θ; h = -1)
    J, nd = size(θ)

    XY = θ*θ';
    x2= sum(θ.^2, dims=2);
    X2e = repeat(x2, 1, J);
    pairwise_dists = X2e + X2e' - 2*XY

    if h < 0
        h = median(pairwise_dists)  
        h = sqrt(0.5 * h / log(J+1))
    end
    # compute the rbf kernel
    Kxy = exp.( -pairwise_dists / h^2 / 2)

    dxkxy = -Kxy * θ
    sumkxy = sum(Kxy, dims=2)
    for i = 1:nd
        dxkxy[:, i] = dxkxy[:,i] + θ[:,i].*sumkxy
    end

    dxkxy = dxkxy / (h^2)
    return Kxy, dxkxy
end

# lnprob(θ) = ∇ log p(θ)
# θs0 is a J by n_dim matrix
function SVGD_Run(θs0, lnprob, n_ite::IT = 1000, stepsize::FT = 1e-3, alpha = 0.9) where {FT<:AbstractFloat, IT<:Int}
        # Check input
    
    θs = copy(θs0) 
    lnpgrad = similar(θs0)
    J, nd = size(θs0)
    # adagrad with momentum
    fudge_factor = 1e-6
    historical_grad = 0
    
    for iter = 1:n_ite
        for i = 1:J
            lnpgrad[i, :] = lnprob(θs[i, :])
        end
        # calculating the kernel matrix
        kxy, dxkxy = SVGD_Kernel(θs; h = -1)  
        grad_theta = (kxy * lnpgrad + dxkxy) / J  
        
        # adagrad 
        if iter == 1
            historical_grad = historical_grad .+ grad_theta .^ 2
        else
            historical_grad .= alpha * historical_grad .+ (1 - alpha) * (grad_theta .^ 2)
        end

        adj_grad = grad_theta ./ (fudge_factor .+ sqrt.(historical_grad))
        θs = θs + stepsize * adj_grad 
    end     

    return θs
end
   