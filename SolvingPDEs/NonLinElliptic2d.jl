include("../CovarianceFunctions/CovarianceFunctions.jl")

# For linear algebra
using LinearAlgebra
# logging
using Logging

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
    cov(view(Theta_test,1:N_domain,1:N_boundary), meas_test_int, meas_δ)
    cov(view(Theta_test,1:N_domain,N_boundary+1:N_domain+N_boundary), meas_test_int, meas_Δδ)
    return Theta_train, Theta_test

end
# iterative GPR
function iterGPR_exact(eqn, cov, X_domain, X_boundary, sol_init, nugget, GNsteps)
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    # get the rhs and bdy data
    rhs = [eqn.rhs(X_domain[:,i]) for i in 1:N_domain]
    bdy = [eqn.bdy(X_boundary[:,i]) for i in 1:N_boundary]
    v = sol_init

    for _ in 1:GNsteps
        Theta_train, Theta_test = get_Gram_matrices(eqn, cov, X_domain, X_boundary, v)
        rhs_now = vcat(bdy, rhs+eqn.α*(eqn.m-1)*v.^eqn.m)
    
        v = Theta_test*(((Theta_train+nugget*diagm(diag(Theta_train))))\rhs_now)
    end
    return v
end 


α = 1.0
m = 3
Ω = [[0,1] [0,1]]
h_in = 0.02
h_bd = 0.02
lengthscale = 0.3
kernel = "Matern9half"
cov = MaternCovariance9_2(lengthscale)
nugget = 1e-10
GNsteps = 2

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
@info "[kernel] choose $kernel, lengthscale $lengthscale\n"  
@info "[nugget] $nugget" 
@info "[GNsteps] $GNsteps" 

sol_init = zeros(N_domain) # initial solution
truth = [fun_u(X_domain[:,i]) for i in 1:N_domain]
@time sol_exact = iterGPR_exact(eqn, cov, X_domain, X_boundary, sol_init, nugget, GNsteps)
pts_accuracy_exact = sqrt(sum((truth-sol_exact).^2)/N_domain)
@info "[L2 accuracy: exact method] $pts_accuracy_exact"
pts_max_accuracy_exact = maximum(abs.(truth-sol_exact))
@info "[Linf accuracy: exact method] $pts_max_accuracy_exact"



