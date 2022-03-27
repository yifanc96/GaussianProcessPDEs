include("../CovarianceFunctions/CovarianceFunctions.jl")
include("ExtendKalmanFilter.jl")
# For linear algebra
using LinearAlgebra
# logging
using Logging
using PyPlot

## PDEs type
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

# get prior Gram matrices for Θ
function get_Σ0_init(eqn::NonlinElliptic2d, cov::AbstractCovarianceFunction, X_domain, X_boundary)
    d = 2
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    Δδ_coefs = -1.0

    # get linearized PDEs correponding measurements
    meas_bdyδ = [PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_intΔδ = [ΔδPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), Δδ_coefs, 0.0) for i = 1:N_domain]
    meas_intδ = [PointMeasurement{d}(SVector{d,Float64}(X_domain[:,i])) for i = 1:N_domain]

    Σ0_init = zeros(2*N_domain+N_boundary,2*N_domain+N_boundary)

    measurements = Vector{Vector{<:AbstractPointMeasurement}}(undef,3)
    measurements[1] = meas_bdyδ; measurements[2] = meas_intδ
    measurements[3] = meas_intΔδ
    cov(Σ0_init, reduce(vcat,measurements))
    
    return Σ0_init
end

# get_G
function get_G(z,N_boundary,N_domain,α,m,rhs,bdy)
    y = zeros(N_boundary+N_domain)
    y[1:N_boundary] = z[1:N_boundary]-bdy
    y[N_boundary+1:end] = z[N_boundary+N_domain+1:end] + α*z[N_boundary+1:N_boundary+N_domain].^m-rhs
    return y
end

# get dG
function get_dG(z,N_boundary,N_domain,α,m)
    return [LinearAlgebra.I zeros(N_boundary,2*N_domain); zeros(N_domain,N_boundary)  α*m*Diagonal(z[N_boundary+1:N_boundary+N_domain].^(m-1)) LinearAlgebra.I]
end

function get_Σ_η(eqn::NonlinElliptic2d, cov_rhs::AbstractCovarianceFunction, cov_bdy::AbstractCovarianceFunction, X_domain, X_boundary)
    d = 2
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    meas_bdδ = [PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_intδ = [PointMeasurement{d}(SVector{d,Float64}(X_domain[:,i])) for i = 1:N_domain]

    Σ_η_rhs = zeros(N_domain, N_domain)
    Σ_η_bd = zeros(N_boundary, N_boundary)

    cov_rhs(Σ_η_rhs, meas_intδ)
    cov_bdy(Σ_η_bd, meas_bdδ)
    
    Σ_η = zeros(N_boundary+N_domain, N_boundary+N_domain)
    @show size(Σ_η), N_boundary, N_domain
    Σ_η[1:N_boundary,1:N_boundary] = Σ_η_bd
    Σ_η[N_boundary+1:end,N_boundary+1:end] = Σ_η_rhs
    return Σ_η
end




α = 100.0
m = 3
Ω = [[0,1] [0,1]]
h_in = 0.02
h_bd = 0.02
lengthscale_u = 0.3
kernel_u = "Matern7half"
cov_u = MaternCovariance7_2(lengthscale_u)
kernel_bdy = "Matern5half"
lengthscale_bdy = 0.3
cov_bdy = MaternCovariance5_2(lengthscale_bdy)
kernel_rhs = "Matern5half"
lengthscale_rhs = 0.3
cov_rhs = MaternCovariance5_2(lengthscale_rhs)

# ground truth solution
freq = 600
s = 4
function fun_u(x)
    ans = 0
    @inbounds for k = 1:freq
        ans += sin(pi*k*x[1])*sin(pi*k*x[2])/k^s 
        # H^t norm squared is sum 1/k^{2s-2t}, so in H^{s-1/2}
    end
    return ans
end

# right hand side
function fun_rhs(x)
    ans = 0
    @inbounds for k = 1:freq
        ans += (2*k^2*pi^2)*sin(pi*k*x[1])*sin(pi*k*x[2])/k^s 
    end
    return ans + α*fun_u(x)^m
end

# boundary value
function fun_bdy(x)
    return fun_u(x)
end

@info "[solver started] NonlinElliptic2d"
@info "[equation] -Δu + $α u^$m = f"
eqn = NonlinElliptic2d(α,m,Ω,fun_bdy,fun_rhs)
X_domain, X_boundary = sample_points_grid(eqn, h_in, h_bd)
N_domain = size(X_domain,2)
N_boundary = size(X_boundary,2)
@info "[sample points] grid size $h_in"
@info "[sample points] N_domain is $N_domain, N_boundary is $N_boundary"  
@info "[prior on u] choose $kernel_u, lengthscale $lengthscale_u\n"  
@info "[prior on rhs] choose $kernel_rhs, lengthscale $lengthscale_rhs\n"  
@info "[prior on bdy] choose $kernel_bdy, lengthscale $lengthscale_bdy\n"  


Σ0_init = get_Σ0_init(eqn, cov_u, X_domain, X_boundary)
Σ_η_init = get_Σ_η(eqn, cov_rhs, cov_bdy, X_domain, X_boundary)
z_init = zeros(2*N_domain+N_boundary) # initial solution
rhs = [eqn.rhs(X_domain[:,i]) for i in 1:N_domain]
bdy = [eqn.bdy(X_boundary[:,i]) for i in 1:N_boundary]

truth = [fun_u(X_domain[:,i]) for i in 1:N_domain]




N_iter = 20
N_θ = 2*N_domain+N_boundary

r_0, Σ_0 = zeros(Float64, N_θ), Σ0_init
θ0_mean, θθ0_cov = r_0, Σ_0


N_y = N_boundary+N_domain
Σ_η = zeros(N_boundary+N_domain, N_boundary+N_domain)
Σ_η[1:N_boundary,1:N_boundary] = 1e-2*diagm(diag(Σ_0[1:N_boundary,1:N_boundary]))
Σ_η[N_boundary+1:end,N_boundary+1:end] = 1e-2*diagm(diag(Σ_0[N_boundary+N_domain+1:end,N_boundary+N_domain+1:end]))
y = zeros(Float64, N_y)

struct Setup_Param{IT<:Int, FT<:AbstractFloat}
    N_boundary::IT
    N_domain::IT
    α::FT
    m::IT
    rhs::Array{FT,1}
    bdy::Array{FT,1}
    N_θ::IT
    N_y::IT
end



function forward(s_param, θ)
    N_boundary,N_domain,α,m,rhs,bdy = s_param.N_boundary, s_param.N_domain, s_param.α, s_param.m, s_param.rhs, s_param.bdy
    G = get_G(θ,N_boundary,N_domain,α,m,rhs,bdy)
    dG = get_dG(θ,N_boundary,N_domain,α,m)
    return G, dG
end

s_param = Setup_Param(N_boundary, N_domain, α, m, rhs, bdy, N_θ, N_y)

method = "ExKI"
# UKI-1
exki_obj = ExKI_Run(s_param, forward, 
method,
y, Σ_η,
r_0,
Σ_0,
θ0_mean, θθ0_cov,
N_iter,)

# @info "ERRORs are :" , norm(exki_obj.θ_mean[end] - θ_post), norm(exki_obj.θθ_cov[end] - Σ_post)
@info norm(truth - exki_obj.θ_mean[end][N_boundary+1:N_boundary+N_domain])/sqrt(N_domain)
# figure()
# plot(truth, "-o", markersize=2, label="Reference")
# sol_mean = exki_obj.θ_mean[end][N_boundary+1:N_boundary+N_domain]
# sol_std = sqrt.(diag(exki_obj.θθ_cov[end][N_boundary+1:N_boundary+N_domain, N_boundary+1:N_boundary+N_domain]))
# plot(sol_mean, label="Prediction", color="C1")
# plot(sol_mean - sol_std, "--", color="C1")
# plot(sol_mean + sol_std, "--", color="C1")

# legend()


Nh = convert(Int,sqrt(N_domain))
figure()
plot_surface(reshape(X_domain[1,:],Nh,Nh), reshape(X_domain[2,:],Nh,Nh), reshape(truth,Nh,Nh), label = "Reference")
plot_surface(reshape(X_domain[1,:],Nh,Nh), reshape(X_domain[2,:],Nh,Nh), reshape(sol_mean - sol_std,Nh,Nh), color="C1", label = "Lower")
plot_surface(reshape(X_domain[1,:],Nh,Nh), reshape(X_domain[2,:],Nh,Nh), reshape(sol_mean + sol_std,Nh,Nh), color="C1", label = "Upper")
display(gcf())


figure()
contourf(reshape(X_domain[1,:],Nh,Nh), reshape(X_domain[2,:],Nh,Nh), reshape(abs.(truth-sol_mean),Nh,Nh))
colorbar()
display(gcf())

figure()
contourf(reshape(X_domain[1,:],Nh,Nh), reshape(X_domain[2,:],Nh,Nh), reshape(sol_std,Nh,Nh))
colorbar()
display(gcf())