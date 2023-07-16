include("../CovarianceFunctions/CovarianceFunctions.jl")
include("./MCMC.jl")


using LinearAlgebra
using Logging
using PyPlot
using Distributions
using KernelDensity

## PDEs type
abstract type AbstractPDEs end
struct NonlinElliptic1d{Tα,Tm,TΩ} <: AbstractPDEs
    # eqn: -Δu + α*u^m = f in [Ω[1],Ω[2]]
    α::Tα
    m::Tm
    Ω::TΩ
    bdy::Function
    rhs::Function
end


## Sample points
function sample_points_rdm(eqn::NonlinElliptic1d, N_domain)
    Ω = eqn.Ω
    xl = Ω[1]
    xr = Ω[2]  

    X_domain = rand(Float64,(N_domain, 1))*(xr-xl) + xl
    X_boundary = [xl,xr]

    return X_domain, X_boundary
end
function sample_points_grid(eqn::NonlinElliptic1d, h_in)
    Ω = eqn.Ω
    xl = Ω[1]
    xr = Ω[2]
   
    X_domain = [x for x in  xl + h_in:h_in:xr-h_in]
    X_boundary = [xl,xr]
    return X_domain, X_boundary
end


## Algorithms
# assemby Gram matrices
function get_Gram_matrices(eqn::NonlinElliptic1d, cov::AbstractCovarianceFunction, X_domain, X_boundary)
    
    d = 1
    N_domain = size(X_domain)[1]
    N_boundary = size(X_boundary)[1]

    Δδ_coefs = -1.0

    meas_δ_bd = [PointMeasurement{d}(SVector{d,Float64}(X_boundary[i])) for i = 1:N_boundary]
    meas_δ_int = [PointMeasurement{d}(SVector{d,Float64}(X_domain[i])) for i = 1:N_domain]
    meas_Δδ = [ΔδPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[i]), Δδ_coefs, 0.0) for i = 1:N_domain]

    measurements = Vector{Vector{<:AbstractPointMeasurement}}(undef,3)
    measurements[1] = meas_δ_bd; measurements[2] = meas_δ_int; measurements[3] = meas_Δδ

    Theta_big = zeros(2*N_domain+N_boundary,2*N_domain+N_boundary)
    cov(Theta_big, reduce(vcat,measurements))
    return Theta_big
end

function get_Gram_matrices(eqn::NonlinElliptic1d, cov::AbstractCovarianceFunction, X_domain, X_boundary, sol_now)

    d = 1
    N_domain = size(X_domain)[1]
    N_boundary = size(X_boundary)[1]
    Δδ_coefs = -1.0
    δ_coefs_int = eqn.α*eqn.m*(sol_now.^(eqn.m-1)) 

    # get linearized PDEs correponding measurements
    meas_δ = [PointMeasurement{d}(SVector{d,Float64}(X_boundary[i])) for i = 1:N_boundary]
    meas_Δδ = [ΔδPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[i]), Δδ_coefs, δ_coefs_int[i]) for i = 1:N_domain]
    meas_test_int = [PointMeasurement{d}(SVector{d,Float64}(X_domain[i])) for i = 1:N_domain]

    Theta_train = zeros(N_domain+N_boundary,N_domain+N_boundary)

    measurements = Vector{Vector{<:AbstractPointMeasurement}}(undef,2)
    measurements[1] = meas_δ; measurements[2] = meas_Δδ
    cov(Theta_train, reduce(vcat,measurements))
    
    Theta_test = zeros(N_domain,N_domain+N_boundary)

    cov(Theta_test, meas_test_int, reduce(vcat,measurements))
    return Theta_train, Theta_test
end
function  get_initial_covariance(cov::AbstractCovarianceFunction, X_domain, X_boundary)
    d = 1
    N_domain = size(X_domain)[1]
    N_boundary = size(X_boundary)[1]
    meas_δ_bd = [PointMeasurement{d}(SVector{d,Float64}(X_boundary[i])) for i = 1:N_boundary]
    meas_δ_int = [PointMeasurement{d}(SVector{d,Float64}(X_domain[i])) for i = 1:N_domain]
    Theta_initial = zeros(N_domain+N_boundary,N_domain+N_boundary)
    measurements = Vector{Vector{<:AbstractPointMeasurement}}(undef,2)
    measurements[1] = meas_δ_bd; measurements[2] = meas_δ_int
    cov(Theta_initial, reduce(vcat,measurements))
    return Theta_initial
end
function  get_initial_covariance(cov::AbstractCovarianceFunction, X_domain)
    d = 1
    N_domain = size(X_domain)[1]

    meas_δ_int = [PointMeasurement{d}(SVector{d,Float64}(X_domain[i])) for i = 1:N_domain]

    Theta_initial = zeros(N_domain,N_domain)
    cov(Theta_initial, meas_δ_int)
    return Theta_initial
end


# Laplace Approximation
function LaplaceApprox(eqn, cov, X_domain, X_boundary, sol_init, noise_var_int, noise_var_bd, GNsteps; nugget = 1e-12)
    N_domain = size(X_domain)[1]
    N_boundary = size(X_boundary)[1]

    # get the rhs and bdy data
    # rhs = [eqn.rhs(X_domain[i]) for i in 1:N_domain] .+ sqrt(noise_var_int) * randn(N_domain)
    # bdy = [eqn.bdy(X_boundary[i]) for i in 1:N_boundary] .+ sqrt(noise_var_bd) * randn(N_boundary)

    rhs = [eqn.rhs(X_domain[i]) for i in 1:N_domain]
    bdy = [eqn.bdy(X_boundary[i]) for i in 1:N_boundary] 

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

    N_domain = size(X_domain)[1]
    N_boundary = size(X_boundary)[1]

    # get the rhs and bdy data
    rhs = [eqn.rhs(X_domain[i]) for i in 1:N_domain] .+ sqrt(noise_var_int) * randn(N_domain)
    bdy = [eqn.bdy(X_boundary[i]) for i in 1:N_boundary] .+ sqrt(noise_var_bd) * randn(N_boundary)
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





### parameters
α = 1.0
m = 3
Ω = [0,1]
h_in = 0.05
lengthscale = 0.3
kernel = "Matern5half"
cov = MaternCovariance5_2(lengthscale)
noise_var_int = 0.0
noise_var_bd = 0.0
GNsteps = 4

function fun_u(x)
    return sin(pi*x)
end

function fun_rhs(x)
    ans = pi^2*sin(pi*x)
    return ans + α*fun_u(x)^m 
end

# boundary value
function fun_bdy(x)
    return fun_u(x)
end

eqn = NonlinElliptic1d(α,m,Ω,fun_bdy,fun_rhs)
# N_domain = 2000
# N_boundary = 2
# X_domain, X_boundary = sample_points_rdm(eqn,N_domain)

X_domain, X_boundary = sample_points_grid(eqn, h_in)
N_domain = size(X_domain)[1]
    N_boundary = size(X_boundary)[1]

@info "[solver started] NonlinElliptic1d"
@info "[equation] -Δu + $α u^$m = f"

@info "[sample points] grid size $h_in"
@info "[sample points] N_domain is $N_domain, N_boundary is $N_boundary"  
@info "[kernel] choose $kernel, lengthscale $lengthscale\n"  
@info "[noise] interior var $noise_var_int, boundary var $noise_var_bd" 
@info "[GNsteps] $GNsteps" 

sol_init = zeros(N_domain) # initial solution
truth = [fun_u(X_domain[i]) for i in 1:N_domain]

@time MAP, sol_postvar, rkhs_norm2 = LaplaceApprox(eqn, cov, X_domain, X_boundary, sol_init, noise_var_int, noise_var_bd, GNsteps)

pts_accuracy = sqrt(sum((truth-MAP).^2)/sum(truth.^2))
@info "[L2 accuracy of MAP to true sol] $pts_accuracy"
pts_max_accuracy = maximum(abs.(truth-MAP))/maximum(abs.(truth))
@info "[Linf accuracy of MAP to true sol] $pts_max_accuracy"
sol_std = [sqrt(abs(sol_postvar[i,i])) for i in 1:N_domain]


######
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

#### mcmc by MALA: noise-free
# compare MCMC with Laplace Approximation
n_ite = 10^8
d_log_post = log_post_GP_PDE(eqn, cov, X_domain, X_boundary)
τ = 1e-5
mcmc_samples = MALA_Run(d_log_post, MAP, τ ,n_ite)
density_mcmc = kde(mcmc_samples[10^6:10^8,3])
fig = figure()
plot(density_mcmc.x,density_mcmc.density, label = "MCMC")
gmean = MAP[3]
gstd = sol_std[3]
density_gaussian_y = 1/(sqrt(2*pi)*gstd)*exp.(-(density_mcmc.x.-gmean).^2/(2*gstd^2))
plot(density_mcmc.x,density_gaussian_y, label="Laplace-Approx")
legend()
display(gcf())
plt.xlabel(L"$x$")
fig.tight_layout()
savefig("UQ_ellipticPDE_1D_LaplaceApprox-MCMC-comparison.pdf")


### as noise goes to zeros
array_noise_var = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.0]
l = size(array_noise_var)[1]
error_MAP = zeros(l)
error_cov = zeros(l)


for i in 1:l
    noise_var_bd = array_noise_var[i]
    noise_var_int = array_noise_var[i]
    now_MAP, now_sol_cov, _ = LaplaceApprox(eqn, cov, X_domain, X_boundary, sol_init, noise_var_int, noise_var_bd, GNsteps)
    error_MAP[i] = norm(now_MAP.-MAP)
    error_cov[i] = norm(now_sol_cov - sol_postvar)
end

fig  = figure()
plot(sqrt.(array_noise_var), error_MAP, "-o", label=L"$ \| m_{β}-m_0 \|_2$")
plot(sqrt.(array_noise_var), error_cov, "-o", label=L"$ \|C_{β}-C_0 \|_2$")
plt.xlabel(L"$β$")
plt.xscale("log")
plt.yscale("log")
legend()
fig.tight_layout()
display(gcf())
savefig("UQ_ellipticPDE_1D_LaplaceApprox-beta.pdf")









