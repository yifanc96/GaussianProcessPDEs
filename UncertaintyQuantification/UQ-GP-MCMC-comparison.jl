include("../CovarianceFunctions/CovarianceFunctions.jl")

using LinearAlgebra
using Logging
using PyPlot
using Distributions
using KernelDensity
using ReverseDiff
using StatsBase
using Statistics
using Random

abstract type AbstractPDEs end
struct NonlinElliptic2d{Tα,Tm,TΩ} <: AbstractPDEs
    # eqn: -Δu + α*u^m = f in [Ω[1,1],Ω[2,1]]*[Ω[1,2],Ω[2,2]]
    α::Tα
    m::Tm
    Ω::TΩ
    bdy::Function
    rhs::Function
end

## sample points
function sample_points_rdm(eqn::NonlinElliptic2d, N_domain, N_boundary)
    Ω = eqn.Ω
    x1l = Ω[1,1]
    x1r = Ω[2,1]
    x2l = Ω[1,2]
    x2r = Ω[2,2]   

    X_domain = hcat(rand(Float64,(N_domain, 1))*(x1r-x1l).+x1l,rand(Float64,(N_domain, 1))*(x2r-x2l).+x2l)

    N_bd_each=convert(Int64, N_boundary/4)
    if N_boundary != 4 * N_bd_each
        println("[sample points] N_boundary not divided by 4, replaced by ", 4 * N_bd_each)
        N_boundary = 4 * N_bd_each
    end

    X_boundary = zeros((N_boundary, 2))
    # bottom face
    X_boundary[1:N_bd_each, :] = hcat((x1r-x1l)*rand(Float64,(N_bd_each,1)).+x1l, x2l*ones(N_bd_each))
    # right face
    X_boundary[N_bd_each+1:2*N_bd_each, :] = hcat(x1r*ones(N_bd_each),(x2r-x2l)*rand(Float64,(N_bd_each,1)).+x2l)
    # top face
    X_boundary[2*N_bd_each+1:3*N_bd_each, :] = hcat((x1r-x1l)*rand(Float64,(N_bd_each,1)).+x1l, x2r*ones(N_bd_each))

    # left face
    X_boundary[3*N_bd_each+1:N_boundary, :] = hcat(x1l*ones(N_bd_each), (x2r-x2l)*rand(Float64,(N_bd_each,1)).+x2l)
    return X_domain', X_boundary'
end

function sample_points_grid(eqn::NonlinElliptic2d, h_in, h_bd)
    Ω = eqn.Ω
    x1l = Ω[1,1]
    x1r = Ω[2,1]
    x2l = Ω[1,2]
    x2r = Ω[2,2]
    x = x1l + h_in:h_in:x1r-h_in
    y = x2l + h_in:h_in:x2r-h_in
    X_domain = reduce(hcat,[[x[i], y[j]] for i in 1:length(x) for j in 1:length(x)])

    l = length(x1l:h_bd:x1r-h_bd)
    X_boundary = vcat([x1l:h_bd:x1r-h_bd x2l*ones(l)], [x1r*ones(l) x2l:h_bd:x2r-h_bd], [x1r:-h_bd:x1l+h_bd x2r*ones(l)], [x1l*ones(l) x2r:-h_bd:x1l+h_bd])
    return X_domain, X_boundary'
end


# assemby Gram matrices
function get_Gram_matrices(eqn::NonlinElliptic2d, cov::AbstractCovarianceFunction, X_domain, X_boundary)
    
    d = 2
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)

    Δδ_coefs = -1.0

    meas_δ_bd = [PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_δ_int = [PointMeasurement{d}(SVector{d,Float64}(X_domain[:,i])) for i = 1:N_domain]
    meas_Δδ = [ΔδPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), Δδ_coefs, 0.0) for i = 1:N_domain]

    measurements = Vector{Vector{<:AbstractPointMeasurement}}(undef,3)
    measurements[1] = meas_δ_bd; measurements[2] = meas_δ_int; measurements[3] = meas_Δδ

    Theta_big = zeros(2*N_domain+N_boundary,2*N_domain+N_boundary)
    cov(Theta_big, reduce(vcat,measurements))
    return Theta_big
end
            
function get_Gram_matrices(eqn::NonlinElliptic2d, cov::AbstractCovarianceFunction, X_domain, X_boundary, sol_now)
    d = 2
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    Δδ_coefs = -1.0
    δ_coefs_int = eqn.α*eqn.m*(sol_now.^(eqn.m-1)) 

    # get linearized PDEs correponding measurements
    meas_δ = [PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_Δδ = [ΔδPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), Δδ_coefs, δ_coefs_int[i]) for i = 1:N_domain]
    meas_test_int = [PointMeasurement{d}(SVector{d,Float64}(X_domain[:,i])) for i = 1:N_domain]

    Theta_train = zeros(N_domain+N_boundary,N_domain+N_boundary)

    measurements = Vector{Vector{<:AbstractPointMeasurement}}(undef,2)
    measurements[1] = meas_δ; measurements[2] = meas_Δδ
    cov(Theta_train, reduce(vcat,measurements))
    
    Theta_test = zeros(N_domain,N_domain+N_boundary)

    cov(Theta_test, meas_test_int, reduce(vcat,measurements))
    return Theta_train, Theta_test
end

function  get_initial_covariance(cov::AbstractCovarianceFunction, X_domain, X_boundary)
    d = 2
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    meas_δ_bd = [PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_bd_int = [PointMeasurement{d}(SVector{d,Float64}(X_domain[:,i])) for i = 1:N_domain]
    Theta_initial = zeros(N_domain+N_boundary,N_domain+N_boundary)
    measurements = Vector{Vector{<:AbstractPointMeasurement}}(undef,2)
    measurements[1] = meas_δ_bd; measurements[2] = meas_bd_int
    cov(Theta_initial, reduce(vcat,measurements))
    return Theta_initial
end

function  get_initial_covariance(cov::AbstractCovarianceFunction, X_domain)
    d = 2
    N_domain = size(X_domain,2)

    meas_bd_int = [PointMeasurement{d}(SVector{d,Float64}(X_domain[:,i])) for i = 1:N_domain]

    Theta_initial = zeros(N_domain,N_domain)
    cov(Theta_initial, meas_bd_int)
    return Theta_initial
end
                                
# Laplace Approximation
function KalmanLaplaceApprox(eqn, cov, X_domain, X_boundary, sol_init, noise_var_int, noise_var_bd, GNsteps; nugget = 1e-12)
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)

    # get the rhs and bdy data
    rhs = [eqn.rhs(X_domain[:,i]) for i in 1:N_domain] .+ sqrt(noise_var_int) * randn(N_domain)
    bdy = [eqn.bdy(X_boundary[:,i]) for i in 1:N_boundary] .+ sqrt(noise_var_bd) * randn(N_boundary)
    Theta_train = zeros(N_domain+N_boundary, N_domain+N_boundary) 
    Theta_test = zeros(N_domain,N_domain+N_boundary)

    v = sol_init
    rhs_now = vcat(bdy, rhs.+eqn.α*(eqn.m-1)*v.^eqn.m)
    noise_cov = diagm(vcat([noise_var_bd for _ in 1:N_boundary], [noise_var_int for _ in 1:N_domain]))
    for _ in 1:GNsteps
        Theta_train, Theta_test = get_Gram_matrices(eqn, cov, X_domain, X_boundary, v)
        rhs_now = vcat(bdy, rhs.+eqn.α*(eqn.m-1)*v.^eqn.m)
        v = Theta_test*(((Theta_train+noise_cov+nugget*diagm(diag(Theta_train))))\rhs_now)
    end

    MAP = v
    Cov_init = get_initial_covariance(cov, X_domain) # only consider UQ of domain pts
    Cov_posterior = Cov_init .- Theta_test*(((Theta_train+noise_cov+nugget*diagm(diag(Theta_train))))\Theta_test')
    approx_rkhs_norm2 = rhs_now'* (((Theta_train+noise_cov+nugget*diagm(diag(Theta_train))))\rhs_now)
    return MAP, Cov_posterior, approx_rkhs_norm2
end 


function log_post_GP_PDE(eqn, cov, X_domain, X_boundary; nugget = 1e-14)

    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)

    # get the rhs and bdy data
    rhs = [eqn.rhs(X_domain[:,i]) for i in 1:N_domain] .+ sqrt(noise_var_int) * randn(N_domain)
    bdy = [eqn.bdy(X_boundary[:,i]) for i in 1:N_boundary] .+ sqrt(noise_var_bd) * randn(N_boundary)
    Theta_big = zeros(2*N_domain+N_boundary, 2*N_domain+N_boundary)
    Theta_big = get_Gram_matrices(eqn, cov, X_domain, X_boundary)
    U = cholesky(Theta_big+nugget*diagm(diag(Theta_big))).U  #Theta = U'*U

    function d_log_post(z)
        Z = vcat(bdy,z,rhs.-eqn.α*z.^eqn.m)
        tmp = U\(U'\Z)
        return -0.5*Z'*tmp, -tmp[N_boundary+1:N_boundary+N_domain] + eqn.α*eqn.m*(z.^(eqn.m-1)).*tmp[N_boundary+N_domain+1:end]
    end
    return d_log_post
end


function hessian_log_post_GP_PDE(eqn, cov, X_domain, X_boundary; nugget = 1e-9)
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)

    # get the rhs and bdy data
    rhs = [eqn.rhs(X_domain[:,i]) for i in 1:N_domain] .+ sqrt(noise_var_int) * randn(N_domain)
    bdy = [eqn.bdy(X_boundary[:,i]) for i in 1:N_boundary] .+ sqrt(noise_var_bd) * randn(N_boundary)
    Theta_big = zeros(2*N_domain+N_boundary, 2*N_domain+N_boundary)
    Theta_big = get_Gram_matrices(eqn, cov, X_domain, X_boundary)
    inv_Theta_big = inv(Theta_big + nugget*diagm(diag(Theta_big)))
    function log_post(z)
        Z = vcat(bdy,z,rhs.-eqn.α*z.^eqn.m)
        return -0.5*Z'*inv_Theta_big*Z
    end
                                                                
    function hessian_log_post(z)
        Z = vcat(bdy,z,rhs.-eqn.α*z.^eqn.m)
        grad_Z = vcat(zeros(N_boundary,N_domain), Matrix(1.0*I,N_domain,N_domain), -eqn.α*eqn.m*diagm(z.^(eqn.m-1)))
        hess = grad_Z'*(Theta_big\grad_Z)
        for i = 1:N_domain
            for j = 1:2*N_domain+N_boundary
                hess[i,i] += inv_Theta_big[N_domain+N_boundary+i, j]*Z[j]*(-eqn.α)*eqn.m*(eqn.m-1)*z[i]^(eqn.m-2)
            end
        end    
        return -hess
    end
#     function hessian_log_post(z)
#         return ReverseDiff.hessian(log_post,z)
#     end
    return log_post, hessian_log_post
end
                                                                       
                                                                

function LaplaceApprox(eqn, cov, X_domain, X_boundary, sol_init, noise_var_int, noise_var_bd, GNsteps; nugget = 1e-14)
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)

    rhs = [eqn.rhs(X_domain[:,i]) for i in 1:N_domain] .+ sqrt(noise_var_int) * randn(N_domain)
    bdy = [eqn.bdy(X_boundary[:,i]) for i in 1:N_boundary] .+ sqrt(noise_var_bd) * randn(N_boundary)

    Theta_train = zeros(N_domain+N_boundary, N_domain+N_boundary) 
    Theta_test = zeros(N_domain,N_domain+N_boundary)

    v = sol_init
    rhs_now = vcat(bdy, rhs.+eqn.α*(eqn.m-1)*v.^eqn.m)
    noise_cov = diagm(vcat([noise_var_bd for _ in 1:N_boundary], [noise_var_int for _ in 1:N_domain]))
    for _ in 1:GNsteps
        Theta_train, Theta_test = get_Gram_matrices(eqn, cov, X_domain, X_boundary, v)
        rhs_now = vcat(bdy, rhs.+eqn.α*(eqn.m-1)*v.^eqn.m)
        v = Theta_test*(((Theta_train+noise_cov+nugget*diagm(diag(Theta_train))))\rhs_now)
    end

    MAP = v
    _, hessian_log_post = hessian_log_post_GP_PDE(eqn, cov, X_domain, X_boundary)
    hessian_mtx = -hessian_log_post(MAP)
    Cov_posterior = inv(hessian_mtx)
    return MAP, Cov_posterior
end 


α = 10.0
m = 3
Ω = [[0,1] [0,1]]
h_in = 0.1
h_bd = 0.1
lengthscale = 0.3
kernel = "Matern7alf"
cov = MaternCovariance7_2(lengthscale)
noise_var_int = 0.0
noise_var_bd = 0.0
GNsteps = 10

function fun_u(x)
    return sin(pi*x[1])*sin(pi*x[2]) + sin(3*pi*x[1])*sin(3*pi*x[2])
end

function fun_rhs(x)
    ans = 2*pi^2*sin(pi*x[1])*sin(pi*x[2]) + 2*(3*pi)^2*sin(3*pi*x[1])*sin(3*pi*x[2])
    return ans + α*fun_u(x)^m 
end

# boundary value
function fun_bdy(x)
    return fun_u(x)
end

eqn = NonlinElliptic2d(α,m,Ω,fun_bdy,fun_rhs)


X_domain, X_boundary = sample_points_grid(eqn, h_in, h_bd)
N_domain = size(X_domain,2)
N_boundary = size(X_boundary,2)

sol_init = zeros(N_domain) # initial solution
truth = [fun_u(X_domain[:,i]) for i in 1:N_domain]

@time MAP, KalmanLaplace_postvar, rkhs_norm2 = KalmanLaplaceApprox(eqn, cov, X_domain, X_boundary, sol_init, noise_var_int, noise_var_bd, GNsteps)
pts_accuracy = sqrt(sum((truth-MAP).^2)/sum(truth.^2))
@info "[L2 accuracy of MAP to true sol] $pts_accuracy"
pts_max_accuracy = maximum(abs.(truth-MAP))/maximum(abs.(truth))
@info "[Linf accuracy of MAP to true sol] $pts_max_accuracy"
sol_std_Kalman = [sqrt(abs(KalmanLaplace_postvar[i,i])) for i in 1:N_domain]

    
@time MAP_Laplace, Laplace_postvar = LaplaceApprox(eqn, cov, X_domain, X_boundary, sol_init, noise_var_int, noise_var_bd, GNsteps)
sol_std = [sqrt(abs(Laplace_postvar[i,i])) for i in 1:N_domain]


# θ_s: N_ens/2 by N_θ matrix, the set
# θ_c: N_ens/2 by N_θ matrix, the complementary set
function emcee_Propose(θ_s::Array{FT,2}, θ_c::Array{FT,2}; a::FT = 2.0) where {FT<:AbstractFloat, IT<:Int}
    Ns, N_θ = size(θ_s)
        
    zz = ((a - 1.0) * rand(Uniform(0, 1), Ns) .+ 1).^2.0 / a
    factors = (N_θ - 1.0) * log.(zz)

    rint = rand(1:Ns, Ns,)
    return θ_c[rint, :] - (θ_c[rint, :] - θ_s) .* zz, factors

end

"""
When the density function is Φ/Z, 
The log_bayesian_posterior function return log(Φ) instead of Φ
θ0 : initial ensemble of size N_ens by N_θ
"""
function emcee_Run(log_bayesian_posterior::Function, θ0::Array{FT,2}, n_ite::IT; thin::IT = 1, random_split::Bool = true, a::FT = 2.0, seed::IT=11) where {FT<:AbstractFloat, IT<:Int}
    Random.seed!(seed)

    N_ens, N_θ = size(θ0)
    @assert(N_ens >= 2N_θ)
    @assert(N_ens % 2 == 0)
#     print(n_ite÷thin+1,"\n")
    θs = zeros(Float64, n_ite÷thin+1, N_ens, N_θ)
    fs = zeros(Float64, n_ite÷thin+1, N_ens)
    store_idx = 1
    θs[1, :, :] .= θ0
    for k = 1:N_ens
        fs[1, k] = log_bayesian_posterior(θ0[k, :])
    end
    
    last_step_θ0 = θs[1, :, :]
    last_step_fs = fs[1, :]
    
    nsplit = 2
    N_s = div(N_ens, 2)
    
    all_inds = Array(1:N_ens)
    inds = all_inds .% nsplit # 2 group
    log_probs = zeros(Float64, N_s)
    
    for i_t = 2:n_ite
        
        if random_split
            shuffle!(inds)
        end
        for split = 0:1
            # boolean array of the current set
            s_inds = (inds .== split)
            # boolean array for the complementary set
            c_inds = (inds .!= split)
#             s, c = θs[i_t - 1, s_inds, :], θs[i_t - 1, c_inds, :]
            s, c = last_step_θ0[s_inds,:], last_step_θ0[c_inds,:]
           
            q, factors = emcee_Propose(s, c; a = a)
            for i = 1:N_s
                log_probs[i] = log_bayesian_posterior(q[i, :])
            end

            # Loop over the walkers and update them accordingly.
            for i = 1:N_s
                j = all_inds[s_inds][i]

                # j is the index
                # @info i, factors[i] , log_probs[i] , fs[i_t - 1, j]
                
                α = min(1.0, exp(factors[i] + log_probs[i] - last_step_fs[j]))
                if α > rand(Uniform(0, 1))
                    # accept
#                     θs[i_t, j, :] = q[i, :]
#                     fs[i_t, j] = log_probs[i]
                    last_step_θ0[j,:] = q[i,:]
                    last_step_fs[j] = log_probs[i]
#                 else
                    # reject
#                     θs[i_t, j, :] = θs[i_t - 1, j, :]
#                     fs[i_t, j] = fs[i_t - 1, j]
                end
            end
        end
        if mod(i_t,thin) == 0
            store_idx += 1
#             print(store_idx, " ", i_t, "\n")
            θs[store_idx, :, :] = last_step_θ0
            fs[store_idx, :] = last_step_fs
        end
    end
    
    return θs
end


emcee_ite = 10^7
log_post, hessian_log_post = hessian_log_post_GP_PDE(eqn, cov, X_domain, X_boundary; nugget = 1e-14)
ensemble_size = 200
θ0 = zeros(ensemble_size, N_domain)
for i in 1:ensemble_size
    θ0[i,:] = MAP + rand(Normal(0,1), N_domain)
end
emcee_samples = emcee_Run(log_post, θ0, emcee_ite; thin=100)
print("")

using JLD2
save("MCMC_1e7steps.jld2", "sample", emcee_samples)

mean_emcmc = mean(emcee_samples[10:end,:,:], dims=[1,2])
std_emcmc = std(emcee_samples[10:end,:,:], dims=[1,2])
print("Laplace mean err ", mean(abs.(mean_emcmc[:] - MAP).^2) ./ mean(abs.(mean_emcmc[:]).^2))
print("\n Laplace std err ", mean(abs.(std_emcmc[:] - sol_std).^2) ./ mean(std_emcmc[:].^2))
print("\n Kalman std err ", mean(abs.(std_emcmc[:] - sol_std_Kalman).^2) ./ mean(std_emcmc[:].^2))
