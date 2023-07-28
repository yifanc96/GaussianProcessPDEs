include("../CovarianceFunctions/CovarianceFunctions.jl")


using LinearAlgebra
using Logging
using PyPlot
using Distributions
using StatsBase

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


## Sample points
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


function sample_points_rdm(eqn::NonlinElliptic2d, N_domain)
    Ω = eqn.Ω
    x1l = Ω[1,1]
    x1r = Ω[2,1]
    x2l = Ω[1,2]
    x2r = Ω[2,2]   

    X_domain = hcat(rand(Float64,(N_domain, 1))*(x1r-x1l).+x1l,rand(Float64,(N_domain, 1))*(x2r-x2l).+x2l)
    return X_domain'
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


## Algorithms
# assemby Gram matrices
function get_Gram_matrices(eqn::NonlinElliptic2d, cov::AbstractCovarianceFunction, X_domain, X_boundary)
    
    # @assert(typeof(eqn)==::NonlinElliptic2d)
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

function get_Gram_matrices(eqn::NonlinElliptic2d, cov::AbstractCovarianceFunction, X_test, X_domain, X_boundary, sol_now)
    d = 2
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    N_test = size(X_test,2)
    Δδ_coefs = -1.0
    δ_coefs_int = eqn.α*eqn.m*(sol_now.^(eqn.m-1)) 

    # get linearized PDEs correponding measurements
    meas_δ = [PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_Δδ = [ΔδPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), Δδ_coefs, δ_coefs_int[i]) for i = 1:N_domain]

    meas_test_int = [PointMeasurement{d}(SVector{d,Float64}(X_test[:,i])) for i = 1:N_test]

    Theta_train = zeros(N_domain+N_boundary,N_domain+N_boundary)
    measurements = Vector{Vector{<:AbstractPointMeasurement}}(undef,2)
    measurements[1] = meas_δ; measurements[2] = meas_Δδ
    cov(Theta_train, reduce(vcat,measurements))
    
    Theta_test = zeros(N_test,N_domain+N_boundary)

    cov(Theta_test, meas_test_int, reduce(vcat,measurements))
    return Theta_train, Theta_test
end

function  get_initial_covariance(cov::AbstractCovarianceFunction, X_domain, X_boundary)
    d = 2
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    meas_δ_bd = [PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_δ_int = [PointMeasurement{d}(SVector{d,Float64}(X_domain[:,i])) for i = 1:N_domain]
    Theta_initial = zeros(N_domain+N_boundary,N_domain+N_boundary)
    measurements = Vector{Vector{<:AbstractPointMeasurement}}(undef,2)
    measurements[1] = meas_δ_bd; measurements[2] = meas_δ_int
    cov(Theta_initial, reduce(vcat,measurements))
    return Theta_initial
end

function  get_initial_covariance(cov::AbstractCovarianceFunction, X_domain)
    d = 2
    N_domain = size(X_domain,2)

    meas_δ_int = [PointMeasurement{d}(SVector{d,Float64}(X_domain[:,i])) for i = 1:N_domain]

    Theta_initial = zeros(N_domain,N_domain)
    cov(Theta_initial, meas_δ_int)
    return Theta_initial
end


function get_MAP(eqn, cov, X_domain, X_boundary, sol_init, noise_var_int, noise_var_bd, GNsteps; nugget = 1e-12)
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
    return MAP, rhs_now
end 

function get_eqn_residue(eqn,cov,X_test, X_domain,X_boundary, MAP, rhs_now; nugget = 1e-12)
    d = 2
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    N_test = size(X_test,2)

    Δδ_coefs = -1.0
    δ_coefs_int = eqn.α*eqn.m*(MAP.^(eqn.m-1)) 

    # get linearized PDEs correponding measurements
    meas_δ = [PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_Δδ = [ΔδPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), Δδ_coefs, δ_coefs_int[i]) for i = 1:N_domain]

    meas_test_int = [PointMeasurement{d}(SVector{d,Float64}(X_test[:,i])) for i = 1:N_test]
    meas_test_Δδ = [ΔδPointMeasurement{Float64,d}(SVector{d,Float64}(X_test[:,i]), -1, 0.0) for i = 1:N_test]

    Theta_train = zeros(N_domain+N_boundary,N_domain+N_boundary)
    measurements = Vector{Vector{<:AbstractPointMeasurement}}(undef,2)
    measurements[1] = meas_δ; measurements[2] = meas_Δδ
    cov(Theta_train, reduce(vcat,measurements))
    
    Theta_test_δ = zeros(N_test,N_domain+N_boundary)
    Theta_test_Δδ = zeros(N_test,N_domain+N_boundary)

    cov(Theta_test_δ, meas_test_int, reduce(vcat,measurements))
    cov(Theta_test_Δδ, meas_test_Δδ, reduce(vcat,measurements))

    u_test = Theta_test_δ*(((Theta_train+nugget*diagm(diag(Theta_train))))\rhs_now)
    negΔu_test = Theta_test_Δδ*(((Theta_train+nugget*diagm(diag(Theta_train))))\rhs_now)
    rhs = [eqn.rhs(X_test[:,i]) for i in 1:N_test]
    res2 = (negΔu_test .+ eqn.α*u_test.^eqn.m .- rhs).^2
    return res2
end

### parameters
α = 1.0
m = 3
Ω = [[0,1] [0,1]]
h_in = 0.02
h_bd = 0.02
lengthscale = 0.3
kernel = "Matern5half"
cov = MaternCovariance9_2(lengthscale)
noise_var_int = 0.0
noise_var_bd = 0.0
GNsteps = 3

# function fun_u(x)
#     return sin(pi*x[1])*sin(pi*x[2]) + sin(3*pi*x[1])*sin(3*pi*x[2])
# end
# function fun_rhs(x)
#     ans = 2*pi^2*sin(pi*x[1])*sin(pi*x[2]) + 2*(3*pi)^2*sin(3*pi*x[1])*sin(3*pi*x[2])
#     return ans + α*fun_u(x)^m 
# end

p = 10
function fun_u(x)
    return 2^(4*p) * x[1]^p*(1-x[1])^p*x[2]^p*(1-x[2])^p
end

function fun_rhs(x)
    ans = -2^(4*p) * (
    (p*(p-1)* x[1]^(p-2)*(1-x[1])^p + p*(p-1)*x[1]^p*(1-x[1])^(p-2)-2*p^2*x[1]^(p-1)*(1-x[1])^(p-1)) *  x[2]^p*(1-x[2])^p
    + (p*(p-1)* x[2]^(p-2)*(1-x[2])^p + p*(p-1)*x[2]^p*(1-x[2])^(p-2)-2*p^2*x[2]^(p-1)*(1-x[2])^(p-1)) *  x[1]^p*(1-x[1])^p
    )
    return ans + α*fun_u(x)^m 
end

# boundary value
function fun_bdy(x)
    return fun_u(x)
end

eqn = NonlinElliptic2d(α,m,Ω,fun_bdy,fun_rhs)
N_domain = 100
N_boundary = 400
X_domain, X_boundary = sample_points_rdm(eqn,N_domain, N_boundary)
N_domain = size(X_domain,2)
N_boundary = size(X_boundary,2)

@info "[solver started] NonlinElliptic2d"
@info "[equation] -Δu + $α u^$m = f"
@info "[sample points] Intial: N_domain is $N_domain, N_boundary is $N_boundary"  
@info "[kernel] choose $kernel, lengthscale $lengthscale\n"  
@info "[noise] interior var $noise_var_int, boundary var $noise_var_bd" 
@info "[GNsteps] $GNsteps" 

sol_init = zeros(N_domain) # initial solution
truth = [fun_u(X_domain[:,i]) for i in 1:N_domain]

@time MAP, rhs_now = get_MAP(eqn, cov, X_domain, X_boundary, sol_init, noise_var_int, noise_var_bd, GNsteps)
pts_accuracy = sqrt(sum((truth-MAP).^2)/sum(truth.^2))

@info "[L2 accuracy of MAP to true sol] $pts_accuracy"
pts_max_accuracy = maximum(abs.(truth-MAP))/maximum(abs.(truth))
@info "[Linf accuracy of MAP to true sol] $pts_max_accuracy"
using PyCall
fsize = 15.0
tsize = 15.0
tdir = "in"
major = 5.0
minor = 3.0
lwidth = 0.8
lhandle = 2.0
plt.style.use("default")
rcParams = PyDict(matplotlib["rcParams"])
rcParams["font.size"] = fsize
rcParams["legend.fontsize"] = tsize
rcParams["xtick.direction"] = tdir
rcParams["ytick.direction"] = tdir
rcParams["xtick.major.size"] = major
rcParams["xtick.minor.size"] = minor
rcParams["ytick.major.size"] = 5.0
rcParams["ytick.minor.size"] = 3.0
rcParams["axes.linewidth"] = lwidth
rcParams["legend.handlelength"] = lhandle
rcParams["lines.markersize"] = 10


fig = figure("pyplot_scatterplot_initial",figsize=(8,6))
fig, ax = PyPlot.subplots(ncols=1, sharex=false, sharey=false)
sm = ax.scatter(X_domain[1,:],X_domain[2,:],s=15, c="blue", alpha = 1)
fig.colorbar(sm, ax=ax)
# ax.scatter([X_all[1,P[pts_idx]]],[X_all[2,P[pts_idx]]], s=200, c="purple")
fig.tight_layout()
display(gcf())

## adaptive sampling 
n_iter = 10
refpts_per_iter = 1000
sample_pts_per_iter = 50

## based on posterior variance
for i_iter in 1:n_iter
    X_test = sample_points_rdm(eqn, refpts_per_iter)
    Theta_train, Theta_test = get_Gram_matrices(eqn, cov, X_test, X_domain, X_boundary, MAP)
    Cov_init = get_initial_covariance(cov, X_test)
    nugget = 1e-12
    Cov_posterior = Cov_init .- Theta_test*(((Theta_train+nugget*diagm(diag(Theta_train))))\Theta_test')
    Cov_diag =  [abs(Cov_posterior[i,i]) for i in 1:refpts_per_iter]
    
    
    # greedy
    var_sort_idx = sortperm(Cov_diag, rev=true)
    X_add = X_test[:,var_sort_idx[1:sample_pts_per_iter]]

    # random
    # X_add = X_test[:1:sample_pts_per_iter]

    # gibbs
    # sample_indx= sample([i for i in 1:refpts_per_iter], Weights(Cov_diag/sum(Cov_diag)), sample_pts_per_iter; replace=true)
    # X_add = X_test[:,sample_indx]

    # residue based sampling
    # res2 = get_eqn_residue(eqn,cov,X_test, X_domain,X_boundary, MAP, rhs_now)
    # res_sort_idx = sortperm(res2, rev=true)
    # X_add = X_test[:,res_sort_idx[1:sample_pts_per_iter]]

    # residue + sigma
    

    X_domain = hcat(X_domain,X_add)
    N_domain = size(X_domain,2)
    sol_init = zeros(N_domain) # initial solution
    truth = [fun_u(X_domain[:,i]) for i in 1:N_domain]
    @time MAP, rhs_now = get_MAP(eqn, cov, X_domain, X_boundary, sol_init, noise_var_int, noise_var_bd, GNsteps)
    pts_accuracy = sqrt(sum((truth-MAP).^2)/sum(truth.^2))
    @info "[L2 accuracy of MAP to true sol] $pts_accuracy"
    pts_max_accuracy = maximum(abs.(truth-MAP))/maximum(abs.(truth))
    @info "[Linf accuracy of MAP to true sol] $pts_max_accuracy"
end


fig = figure("pyplot_scatterplot_end",figsize=(8,6))
fig, ax = PyPlot.subplots(ncols=1, sharex=false, sharey=false)
sm = ax.scatter(X_domain[1,:],X_domain[2,:],s=15, c="blue", alpha = 1)
fig.colorbar(sm, ax=ax)
# ax.scatter([X_all[1,P[pts_idx]]],[X_all[2,P[pts_idx]]], s=200, c="purple")
fig.tight_layout()
display(gcf())

