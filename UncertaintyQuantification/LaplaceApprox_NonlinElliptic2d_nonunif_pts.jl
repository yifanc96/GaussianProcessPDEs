include("../CovarianceFunctions/CovarianceFunctions.jl")

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

## exact algorithms
# assemby Gram matrices
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

# iterative GPR
function iterGPR_exact(eqn, cov, X_domain, X_boundary, sol_init, noise, GNsteps)
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    # get the rhs and bdy data
    rhs = [eqn.rhs(X_domain[:,i]) for i in 1:N_domain] .+ sqrt(noise)*randn(N_domain)
    bdy = [eqn.bdy(X_boundary[:,i]) for i in 1:N_boundary]
    v = sol_init

    Theta_train = zeros(N_domain+N_boundary, N_domain+N_boundary) 
    Theta_test = zeros(N_domain,N_domain+N_boundary)

    rhs_now = vcat(bdy, rhs.+eqn.α*(eqn.m-1)*v.^eqn.m)
    for _ in 1:GNsteps
        Theta_train, Theta_test = get_Gram_matrices(eqn, cov, X_domain, X_boundary, v)
        rhs_now = vcat(bdy, rhs.+eqn.α*(eqn.m-1)*v.^eqn.m)
    
        v = Theta_test*(((Theta_train+noise*LinearAlgebra.I+1e-10*diagm(diag(Theta_train))))\rhs_now)
    end

    Cov = get_initial_covariance(cov, X_domain)
    Cov_posterior = Cov .- Theta_test*(((Theta_train+noise*LinearAlgebra.I))\Theta_test')

    # might be wrong when there is noise on the boundary
    rkhs_norm2 = rhs_now'* (((Theta_train+noise*LinearAlgebra.I+1e-10*diagm(diag(Theta_train))))\rhs_now)
    return v, Cov_posterior, rkhs_norm2
end 


α = 1.0
m = 3
Ω = [[0,1] [0,1]]
h_in = 0.02
h_bd = 0.02
lengthscale = 0.3
kernel = "Matern5half"
cov = MaternCovariance5_2(lengthscale)
# noise = 0.0
noise = 1.0

GNsteps = 4

# ground truth solution
# freq = 1000
# s = 4
# function fun_u(x)
#     ans = 0
#     @inbounds for k = 1:freq
#         ans += sin(pi*k*x[1])*sin(pi*k*x[2])/k^s 
#         # H^t norm squared is sum 1/k^{2s-2t}, so in H^{s-1/2}
#     end
#     return ans
# end

function fun_u(x)
    return sin(pi*x[1])*sin(pi*x[2]) + sin(3*pi*x[1])*sin(3*pi*x[2])
end

function fun_rhs(x)
    ans = 2*pi^2*sin(pi*x[1])*sin(pi*x[2]) + 2*(3*pi)^2*sin(3*pi*x[1])*sin(3*pi*x[2])
    return ans + α*fun_u(x)^m 
end

# right hand side
# function fun_rhs(x)
#     ans = 0
#     @inbounds for k = 1:freq
#         ans += (2*k^2*pi^2)*sin(pi*k*x[1])*sin(pi*k*x[2])/k^s 
#     end
#     return ans + α*fun_u(x)^m 
# end

# boundary value
function fun_bdy(x)
    return fun_u(x)
end

@info "[solver started] NonlinElliptic2d"
@info "[equation] -Δu + $α u^$m = f"
eqn = NonlinElliptic2d(α,m,Ω,fun_bdy,fun_rhs)

# non-uniform sampling
Ω = eqn.Ω
x1l = Ω[1,1]
x1r = Ω[2,1]
x2l = Ω[1,2]
x2r = Ω[2,2]
l = length(x1l:h_bd:x1r-h_bd)
x = sort(rand(Float64,l-1)*(x1r-x1l).+x1l)
y = x2l + h_in:h_in:x2r-h_in
X_domain = reduce(hcat,[[x[i], y[j]] for i in 1:length(x) for j in 1:length(x)])


X_boundary = vcat([x1l:h_bd:x1r-h_bd x2l*ones(l)], [x1r*ones(l) x2l:h_bd:x2r-h_bd], [x1r:-h_bd:x1l+h_bd x2r*ones(l)], [x1l*ones(l) x2r:-h_bd:x1l+h_bd])
X_boundary = X_boundary'

# X_domain, X_boundary = sample_points_grid(eqn, h_in, h_bd)
N_domain = size(X_domain,2)
N_boundary = size(X_boundary,2)
@info "[sample points] grid size $h_in"
@info "[sample points] N_domain is $N_domain, N_boundary is $N_boundary"  
@info "[kernel] choose $kernel, lengthscale $lengthscale\n"  
@info "[noise] $noise" 
@info "[GNsteps] $GNsteps" 

sol_init = zeros(N_domain) # initial solution
truth = [fun_u(X_domain[:,i]) for i in 1:N_domain]
@time sol_exact, sol_postvar, rkhs_norm2 = iterGPR_exact(eqn, cov, X_domain, X_boundary, sol_init, noise, GNsteps)
pts_accuracy_exact = sqrt(sum((truth-sol_exact).^2)/sum(truth.^2))
@info "[L2 accuracy: exact method] $pts_accuracy_exact"
pts_max_accuracy_exact = maximum(abs.(truth-sol_exact))/maximum(abs.(truth))
@info "[Linf accuracy: exact method] $pts_max_accuracy_exact"

sol_std = [sqrt(abs(sol_postvar[i,i])) for i in 1:N_domain]

Nh = convert(Int,sqrt(N_domain))


# figure()
# plot_surface(reshape(X_domain[1,:],Nh,Nh), reshape(X_domain[2,:],Nh,Nh), reshape(truth,Nh,Nh), label = "Reference")
# plot_surface(reshape(X_domain[1,:],Nh,Nh), reshape(X_domain[2,:],Nh,Nh), reshape(sol_exact - sqrt(rkhs_norm2)*sol_std,Nh,Nh), color="C1", label = "Lower RKHS bound")
# plot_surface(reshape(X_domain[1,:],Nh,Nh), reshape(X_domain[2,:],Nh,Nh), reshape(sol_exact + sqrt(rkhs_norm2)*sol_std,Nh,Nh), color="C1", label = "Upper RKHS bound")
# display(gcf())


# figure()
# contourf(reshape(X_domain[1,:],Nh,Nh), reshape(X_domain[2,:],Nh,Nh), reshape(truth - sol_exact + sqrt(rkhs_norm2)*sol_std,Nh,Nh))
# colorbar()
# display(gcf())

# figure()
# contourf(reshape(X_domain[1,:],Nh,Nh), reshape(X_domain[2,:],Nh,Nh), reshape(sol_exact + sqrt(rkhs_norm2)*sol_std - truth,Nh,Nh))
# colorbar()
# display(gcf())



figure()
idx = Nh÷2
plot(x, reshape(truth,Nh,Nh)[idx,:], marker=:d, label = "truth")
plot(x, reshape(sol_exact,Nh,Nh)[idx,:], label = "MAP")
plot(x, reshape(sol_exact + 3.0 * sol_std,Nh,Nh)[idx,:], linestyle="dashed", label = "Upper 3 sigma CI")
plot(x, reshape(sol_exact - 3.0 * sol_std,Nh,Nh)[idx,:], linestyle="dashed", label = "Lower 3 sigma CI")

plot(x, reshape(sol_exact + sqrt(rkhs_norm2) * sol_std,Nh,Nh)[idx,:], linestyle="dashed", label = "Upper RKHS bound")
plot(x, reshape(sol_exact - sqrt(rkhs_norm2) * sol_std,Nh,Nh)[idx,:], linestyle="dashed", label = "Lower RKHS bound")
legend()
display(gcf())


### coutour of the confidence band
# figure()
# contourf(reshape(X_domain[1,:],Nh,Nh), reshape(X_domain[2,:],Nh,Nh), reshape(sol_exact  - truth + sol_std,Nh,Nh))
# colorbar()
# display(gcf())

# figure()
# contourf(reshape(X_domain[1,:],Nh,Nh), reshape(X_domain[2,:],Nh,Nh), reshape(sol_exact  - truth - sol_std,Nh,Nh))
# colorbar()
# display(gcf())

#### mcmc






