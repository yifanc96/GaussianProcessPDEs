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