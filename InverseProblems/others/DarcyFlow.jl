# For linear algebra
using StaticArrays: SVector 
using LinearAlgebra
using SparseArrays: SparseMatrixCSC, sparse
using Interpolations
using Random, Distributions
using PyPlot

include("DarcyFlow_ForwardSolve.jl")
include("../CovarianceFunctions/CovarianceFunctions.jl")

## PDEs type
abstract type InverseProblems end
struct DarcyFlow{TΩ} <: InverseProblems
   # eqn: -∇̇⋅(a∇u) = f in [Ω[1,1],Ω[2,1]]*[Ω[1,2],Ω[2,2]]
   # don't know a and u
   Ω::TΩ
   bdy::Function
   rhs::Function
end

## sample points
function sample_points_rdm(eqn::DarcyFlow, N_domain, N_boundary, N_data)
    Ω = eqn.Ω
    x1l = Ω[1,1]
    x1r = Ω[2,1]
    x2l = Ω[1,2]
    x2r = Ω[2,2]   

    X_domain = hcat(rand(Float64,(N_domain, 1))*(x1r-x1l).+x1l,rand(Float64,(N_domain, 1))*(x2r-x2l).+x2l)

    X_data = hcat(rand(Float64,(N_data, 1))*(x1r-x1l).+x1l,rand(Float64,(N_data, 1))*(x2r-x2l).+x2l)

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
    return X_domain', X_boundary', X_data'
end

# assemby Gram matrices
function get_Gram_matrices(eqn::DarcyFlow, cov, X_domain, X_boundary, X_data, a_old, ∇a_old, ∇u_old, rhs)
    d = 2
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    N_data = size(X_data,2)

    # for u
    Δδ_coefs_u = -1
    ∇δ_coefs_u = -∇a_old
    meas_δbd = [PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_data = [PointMeasurement{d}(SVector{d,Float64}(X_data[:,i])) for i = 1:N_data]
    meas_Δ∇δ = [Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), Δδ_coefs_u, ∇δ_coefs_u[i], 0.0) for i = 1:N_domain]
    
    Theta_train_u = zeros(N_domain+N_boundary+N_data,N_domain+N_boundary+N_data)
    measurements = Vector{Vector{<:AbstractPointMeasurement}}(undef,3)
    measurements[1] = meas_δbd; measurements[2] = meas_data; measurements[3] = meas_Δ∇δ
    cov(Theta_train_u, reduce(vcat,measurements))
    
    Theta_test_u = zeros(3*N_domain,N_domain+N_boundary+N_data)
    meas_δint = [PointMeasurement{d}(SVector{d,Float64}(X_domain[:,i])) for i = 1:N_domain]
    meas_∇1 = [Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), 0.0, [1.0,0.0], 0.0) for i = 1:N_domain]
    meas_∇2 = [Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), 0.0, [0.0,1.0], 0.0) for i = 1:N_domain]
    measurements_test = Vector{Vector{<:AbstractPointMeasurement}}(undef,3)
    measurements_test[1] = meas_δint; measurements_test[2] = meas_∇1; measurements_test[3] = meas_∇2; 
    cov(Theta_test_u, reduce(vcat,measurements_test), reduce(vcat,measurements))

    ## for a
    δ_coefs_a = exp.(-a_old).*rhs
    ∇δ_coefs_a = -∇u_old
    meas_a = [Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), 0.0, ∇δ_coefs_a[i], δ_coefs_a[i]) for i = 1:N_domain]

    Theta_train_a = zeros(N_domain,N_domain)
    cov(Theta_train_a, meas_a)

    Theta_test_a = zeros(3*N_domain,N_domain)
    cov(Theta_test_a, reduce(vcat,measurements_test), meas_a)


    return Theta_train_u, Theta_test_u,Theta_train_a, Theta_test_a 

end

function get_Θ_predict(eqn::DarcyFlow, cov, X_grid, X_domain, X_boundary, X_data, a_old, ∇a_old, ∇u_old)
    d = 2
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    N_data = size(X_data,2)

    # for u
    Δδ_coefs_u = -1
    ∇δ_coefs_u = -∇a_old
    meas_δbd = [PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_data = [PointMeasurement{d}(SVector{d,Float64}(X_data[:,i])) for i = 1:N_data]
    meas_Δ∇δ = [Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), Δδ_coefs_u, ∇δ_coefs_u[i], 0.0) for i = 1:N_domain]
    
    measurements = Vector{Vector{<:AbstractPointMeasurement}}(undef,3)
    measurements[1] = meas_δbd; measurements[2] = meas_data; measurements[3] = meas_Δ∇δ

    ## for a
    δ_coefs_a = exp.(-a_old).*rhs
    ∇δ_coefs_a = -∇u_old
    meas_a = [Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), 0.0, ∇δ_coefs_a[i], δ_coefs_a[i]) for i = 1:N_domain]

    N_grid = length(X_grid)
    Theta_test_a_grid = zeros(N_grid,N_domain)
    Theta_test_u_grid = zeros(N_grid,N_domain+N_boundary+N_data)
    meas_grid = [PointMeasurement{d}(SVector{d,Float64}(X_grid[i])) for i = 1:N_grid]
    cov(Theta_test_a_grid, meas_grid, meas_a)
    cov(Theta_test_u_grid, meas_grid, reduce(vcat,measurements))

    return Theta_test_u_grid, Theta_test_a_grid
end

# iterative GPR
function iterGPR_exact(eqn, cov, X_grid, X_domain, X_boundary, X_data, u_data, rhs_data, bdy_data, a_init, ∇a_init, u_init, ∇u_init, nugget, noise, GNsteps)
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    N_data = size(X_data,2)

    rhs = rhs_data
    bdy = bdy_data

    u_old  = u_init
    a_old = a_init
    ∇a_old = ∇a_init
    ∇u_old = ∇u_init


    for iter_i in 1:GNsteps
        Theta_train_u, Theta_test_u,Theta_train_a, Theta_test_a = get_Gram_matrices(eqn, cov, X_domain, X_boundary, X_data, a_old, ∇a_old, ∇u_old, rhs)

        Theta_u = copy(Theta_train_u)

        Theta_u[N_boundary+1:N_boundary+N_data,N_boundary+1:N_boundary+N_data] += noise * Matrix{Float64}(I,N_data,N_data)
        Theta_u[N_boundary+N_data+1:end,N_boundary+N_data+1:end] += Theta_train_a

        ∇a∇u_old = [sum(∇a_old[i].*∇u_old[i]) for i in 1:N_domain]
        rhs_now  = rhs.*(1 .+ a_old).*exp.(-a_old) .- ∇a∇u_old
        obs_u = vcat(bdy, u_data, rhs_now)
        
        sol_u = Theta_test_u*(((Theta_u+nugget*diagm(diag(Theta_u))))\obs_u)
        u_old = sol_u[1:N_domain]
        ∇u_old = [[sol_u[N_domain+i],sol_u[2*N_domain+i]] for i in 1:N_domain]

        Theta_a = copy(Theta_train_a)
        Theta_a += Theta_train_u[N_boundary+N_data+1:end,N_boundary+N_data+1:end]

        sol_a = Theta_test_a*(((Theta_a+nugget*diagm(diag(Theta_a))))\rhs_now)
        a_old = sol_a[1:N_domain]
        ∇a_old = [[sol_a[N_domain+i],sol_a[2*N_domain+i]] for i in 1:N_domain]

        if iter_i == GNsteps
            Theta_test_u_grid, Theta_test_a_grid = get_Θ_predict(eqn, cov, X_grid, X_domain, X_boundary, X_data, a_old, ∇a_old, ∇u_old)
            sol_u = Theta_test_u_grid*(((Theta_u+nugget*diagm(diag(Theta_u))))\obs_u)
            u_old = sol_u
            sol_a = Theta_test_a_grid*(((Theta_a+nugget*diagm(diag(Theta_a))))\rhs_now)
            a_old = sol_a
        end
    end
    
    return u_old, a_old
end 


function f(x)
    return 1.0
end
function g(x)
    return 0.0
end
Ω = [[0,1] [0,1]]
eqn = DarcyFlow(Ω,g,f)

# get sampled points
Random.seed!(3)
N_domain = 400
N_boundary = 100
N_data = 100
X_domain, X_boundary, X_data = sample_points_rdm(eqn, N_domain, N_boundary, N_data)

# get the rhs and bdy data
rhs = [eqn.rhs(X_domain[:,i]) for i in 1:N_domain]
bdy = [eqn.bdy(X_boundary[:,i]) for i in 1:N_boundary]

# get observation data
function a(x,y) # a(x) truth
    c=1
    return exp(c*sin(2*pi*x)+c*sin(2*pi*y))+exp(-c*sin(2*pi*x)-c*sin(2*pi*y))
end
N = 2^6+1
Δx = 1/(N-1)
x = (0:N-1)/(N-1)
y = copy(x)
X_grid = [[x[i],y[j]] for j in 1:N for i in 1:N]
f_2d = [f([x[i],y[j]]) for j in 1:N for i in 1:N]
FD_darcy = Setup_Param(N,Δx,reshape(f_2d,N,N))
FD_a = [a(x[i],y[j]) for j in 1:N for i in 1:N]
sol = solve_Darcy_2D(FD_darcy, reshape(FD_a,N,N))
#interpolation
itp = LinearInterpolation((x, y), sol)

data_u = [itp(X_data[1,i],X_data[2,i]) for i in 1:N_data]
noise = 0.0
noisy_data_u = data_u + noise*rand(Normal(0, 1), length(data_u))

# GP solver
lengthscale = 0.9
cov = MaternCovariance5_2(lengthscale)
nugget = 1e-10

a_init = [0.0 for i in 1:N_domain]
∇a_init = [[0.0,0.0] for i in 1:N_domain]
u_init = [0.0 for i in 1:N_domain]
∇u_init = [[0.0,0.0] for i in 1:N_domain]

# a_init = [randn(1)[1] for i in 1:N_domain]
# ∇a_init = [randn(2) for i in 1:N_domain]
# u_init = [randn(1)[1] for i in 1:N_domain]
# ∇u_init = [randn(2) for i in 1:N_domain]
GNsteps = 5

sol_u, sol_a = iterGPR_exact(eqn, cov, X_grid, X_domain, X_boundary, X_data, noisy_data_u, rhs, bdy, a_init, ∇a_init, u_init, ∇u_init, nugget, noise, GNsteps)

truth_a = FD_a
N_grid = length(X_grid)
truth_u = [itp(X_grid[i][1],X_grid[i][2]) for i in 1:N_grid]

Xgrid = reduce(hcat,X_grid)
figure()
contourf(reshape(Xgrid[1,:],N,N), reshape(Xgrid[2,:],N,N), reshape(exp.(sol_a),N,N))
colorbar()
display(gcf())

# figure()
# contourf(reshape(Xgrid[1,:],N,N), reshape(Xgrid[2,:],N,N), reshape(abs.(truth_a),N,N))
# colorbar()
# display(gcf())

figure()
contourf(reshape(Xgrid[1,:],N,N), reshape(Xgrid[2,:],N,N), reshape(abs.(sol_u),N,N))
colorbar()
display(gcf())

# figure()
# contourf(reshape(Xgrid[1,:],N,N), reshape(Xgrid[2,:],N,N), reshape(abs.(truth_u),N,N))
# colorbar()
# display(gcf())


