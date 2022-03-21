
# For linear algebra
using StaticArrays: SVector 
using LinearAlgebra
using SparseArrays: SparseMatrixCSC, sparse
# Fast Cholesky
using KoLesky 

# multiple dispatch
import IterativeSolvers: cg!
import LinearAlgebra: mul!, ldiv!
import Base: size

# autoDiff
using ForwardDiff

# parser
using ArgParse

# logging
using Logging

# profile
using Profile
using BenchmarkTools

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--alpha"
            help = "α"
            arg_type = Float64
            default = 1.0
        "--m"
            help = "m"
            arg_type = Int
            default = 3
        "--kernel"
            arg_type = String
            default = "Matern7half"
        "--sigma"
            help = "lengthscale"
            arg_type = Float64
            default = 0.3
        "--h"
            help = "grid size"
            arg_type = Float64
            default = 0.01
        "--nugget"
            arg_type = Float64
            default = 1e-15
        "--GNsteps"
            arg_type = Int
            default = 3
        "--rho_big"
            arg_type = Float64
            default = 3.0
        "--rho_small"
            arg_type = Float64
            # default = -log(5*h_grid)
            default = 3.0
        "--k_neighbors"
            arg_type = Int
            default = 3
        "--compare_exact"
            arg_type = Bool
            default = false
    end
    return parse_args(s)
end

## PDEs type
abstract type AbstractPDEs end
struct NonlinElliptic2d{Tα,Tm,TΩ} <: AbstractPDEs
    # eqn: -Δu + α*u^m = f in [Ω[1,1],Ω[2,1]]*[Ω[1,2],Ω[2,2]]
    α::Tα
    m::Tm
    Ω::TΩ
    a::Function
    ∇a::Function
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
function get_Gram_matrices(eqn::NonlinElliptic2d, cov::KoLesky.AbstractCovarianceFunction, X_domain, X_boundary, sol_now)
    d = 2
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    Δδ_coefs = [-eqn.a(X_domain[:,i]) for i in 1:N_domain]
    ∇δ_coefs = [-eqn.∇a(X_domain[:,i]) for i in 1:N_domain]
    δ_coefs_int = eqn.α*eqn.m*(sol_now.^(eqn.m-1)) 

    # get linearized PDEs correponding measurements
    meas_δ = [KoLesky.PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_Δ∇δ = [KoLesky.Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), Δδ_coefs[i], ∇δ_coefs[i], δ_coefs_int[i]) for i = 1:N_domain]
    meas_test_int = [KoLesky.PointMeasurement{d}(SVector{d,Float64}(X_domain[:,i])) for i = 1:N_domain]

    Theta_train = zeros(N_domain+N_boundary,N_domain+N_boundary)

    measurements = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,2)
    measurements[1] = meas_δ; measurements[2] = meas_Δ∇δ
    cov(Theta_train, reduce(vcat,measurements))
    
    Theta_test = zeros(N_domain,N_domain+N_boundary)
    cov(view(Theta_test,1:N_domain,1:N_boundary), meas_test_int, meas_δ)
    cov(view(Theta_test,1:N_domain,N_boundary+1:N_domain+N_boundary), meas_test_int, meas_Δ∇δ)
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

## algorithm using Kolesky and pcg
# struct that stores the factor of Theta_train
abstract type implicit_mtx end
struct approx_Theta_train{Tv,Ti,Tmtx<:SparseMatrixCSC{Tv,Ti}} <: implicit_mtx
    P::Vector{Ti}
    U::Tmtx
    L::Tmtx
    δ_coefs::Vector{Tv}
    N_boundary::Ti
    N_domain::Ti
end
struct precond_Theta_train{Tv,Ti,Tmtx<:SparseMatrixCSC{Tv,Ti}} <: implicit_mtx
    P::Vector{Ti}
    U::Tmtx
    L::Tmtx
end

function size(A::approx_Theta_train, num)
    return size(A.U,num)
end

function mul!(x, Θtrain::approx_Theta_train, b)
    @views temp = vcat(b[1:Θtrain.N_boundary],Θtrain.δ_coefs.*b[Θtrain.N_boundary+1:end],b[Θtrain.N_boundary+1:end])
    temp[Θtrain.P] = Θtrain.L\(Θtrain.U\temp[Θtrain.P])

    @views x[1:Θtrain.N_boundary] = temp[1:Θtrain.N_boundary]
    @views x[Θtrain.N_boundary+1:end] .= Θtrain.δ_coefs.*(temp[Θtrain.N_boundary+1:Θtrain.N_boundary+Θtrain.N_domain]) .+ temp[Θtrain.N_boundary+Θtrain.N_domain+1:end]
end

function ldiv!(x, precond::precond_Theta_train, b)
    x[precond.P] = precond.U*(precond.L*b[precond.P])
end

function iterGPR_fast_pcg(eqn, cov, X_domain, X_boundary, sol_init, nugget, GNsteps; ρ_big =4.0, ρ_small=6.0, k_neighbors = 4, lambda = 1.5, alpha = 1.0)
    
    @info "[Algorithm]: iterative GPR + fast cholesky factorization + pcg"
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)

    # get the rhs and bdy data
    rhs = [eqn.rhs(X_domain[:,i]) for i in 1:N_domain]
    bdy = [eqn.bdy(X_boundary[:,i]) for i in 1:N_boundary]
    
    sol_now = sol_init

    # form the fast Cholesky part that can be used to compute mtx-vct mul for Theta_test
    d = 2
    Δδ_coefs = [-eqn.a(X_domain[:,i]) for i in 1:N_domain]
    ∇δ_coefs = [-eqn.∇a(X_domain[:,i]) for i in 1:N_domain]
    δ_coefs = 0.0
    meas_δ = [KoLesky.PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_δ_int = [KoLesky.PointMeasurement{d}(SVector{d,Float64}(X_domain[:,i])) for i = 1:N_domain]
    meas_Δ∇δ = [KoLesky.Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), Δδ_coefs[i], ∇δ_coefs[i], δ_coefs) for i = 1:N_domain]


    measurements = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,3)
    measurements[1] = meas_δ; measurements[2] = meas_δ_int
    measurements[3] = meas_Δ∇δ

    @info "[Big Theta: implicit factorization] time"
    # @time implicit_bigΘ = KoLesky.ImplicitKLFactorization_FollowDiracs(cov, measurements, ρ_big, k_neighbors; lambda = lambda, alpha = alpha)
    # @time implicit_bigΘ = KoLesky.ImplicitKLFactorization(cov, measurements, ρ_big, k_neighbors; lambda = lambda, alpha = alpha)
    @time implicit_bigΘ = KoLesky.ImplicitKLFactorization_DiracsFirstThenUnifScale(cov, measurements, ρ_big, k_neighbors; lambda = lambda, alpha = alpha)

    @info "[Big Theta: explicit factorization] time"
    @time explicit_bigΘ = KoLesky.ExplicitKLFactorization(implicit_bigΘ; nugget = nugget)
    U_bigΘ = explicit_bigΘ.U
    L_bigΘ = sparse(U_bigΘ')
    P_bigΘ = explicit_bigΘ.P

    Θtrain = approx_Theta_train(P_bigΘ, U_bigΘ, L_bigΘ,zeros(N_domain),N_boundary,N_domain)

    implicit_factor = nothing

    # update the initial sol

    # Θinv_rhs = zeros(N_boundary+N_domain)

    for step in 1:GNsteps
        
        @info "[Current GN step] $step"
        # get Cholesky of Theta_train
        δ_coefs_int = eqn.α*eqn.m*sol_now.^(eqn.m-1)


        meas_Δ∇δ = [KoLesky.Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), Δδ_coefs[i], ∇δ_coefs[i], δ_coefs_int[i]) for i = 1:N_domain]
        measurements = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,2)
        measurements[1] = meas_δ; measurements[2] = meas_Δ∇δ

        @info "[Theta Train: implicit factorization] time"
        @time if implicit_factor === nothing
            implicit_factor = KoLesky.ImplicitKLFactorization(cov, measurements, ρ_small, k_neighbors; lambda = lambda, alpha = alpha)
        else
            implicit_factor.supernodes.measurements .= reduce(vcat, collect.(measurements))[implicit_factor.P]
        end
        @info "[Theta Train: explicit factorization] time"
        @time explicit_factor = KoLesky.ExplicitKLFactorization(implicit_factor; nugget = nugget)

        U = explicit_factor.U
        L = sparse(U')
        P = explicit_factor.P
        rhs_now = vcat(bdy, rhs.+eqn.α*(eqn.m-1)*sol_now.^eqn.m)

        # use the approximate solution as the initial point for the pCG iteration
        Θinv_rhs = U*(L*rhs_now[P])
        Θinv_rhs[P] = Θinv_rhs

        # pcg step for Theta_train\rhs
        Θtrain.δ_coefs .= δ_coefs_int
        precond = precond_Theta_train(P,U,L)
        
        @info "[pcg started]"
        @time Θinv_rhs, ch = cg!(Θinv_rhs, Θtrain, rhs_now; Pl = precond, log=true)
        @info "[pcg finished], $ch"
        
        # get approximated sol_now = Theta_test * Θinv_rhs
        tmp = vcat(Θinv_rhs[1:N_boundary], δ_coefs_int .* Θinv_rhs[N_boundary+1:end], Θinv_rhs[N_boundary+1:end])
        tmp[P_bigΘ] = L_bigΘ\(U_bigΘ\tmp[P_bigΘ]) 
        @views sol_now = tmp[N_boundary+1:N_domain+N_boundary] 
    end
    @info "[solver finished]"
    return sol_now
end


function main(args)
    α = args.alpha::Float64
    m = args.m::Int
    Ω = [[0,1] [0,1]]
    # ground truth solution
    freq = 100
    s = 3
    function fun_u(x)
        ans = 0
        @inbounds for k = 1:freq
            ans += sin(pi*k*x[1])*sin(pi*k*x[2])/k^s 
            # H^t norm squared is sum 1/k^{2s-2t}, so in H^{s-1/2}
        end
        return ans
    end

    function fun_a(x)
        k = 5
        return exp(sin(k*pi*x[1]*x[2]))
    end

    function grad_a(x)
        return ForwardDiff.gradient(fun_a, x)
    end

    function grad_u(x)
        return ForwardDiff.gradient(fun_u, x)
    end

    # right hand side
    function fun_rhs(x)
        ans = 0
        @inbounds for k = 1:freq
            ans += (2*k^2*pi^2)*sin(pi*k*x[1])*sin(pi*k*x[2])/k^s 
        end
        return -sum(grad_a(x).*grad_u(x)) + fun_a(x)*ans + α*fun_u(x)^m
    end

    # boundary value
    function fun_bdy(x)
        return fun_u(x)
    end

    @info "[solver started] NonlinElliptic2d"
    @info "[equation] - ∇⋅(a∇u) + $α u^$m = f"
    eqn = NonlinElliptic2d(α,m,Ω,fun_a,grad_a,fun_bdy,fun_rhs)
    
    h_in = args.h::Float64; h_bd = h_in
    X_domain, X_boundary = sample_points_grid(eqn, h_in, h_bd)
    # X_domain, X_boundary = sample_points_rdm(eqn, 900, 124)
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    @info "[sample points] grid size $h_in"
    @info "[sample points] N_domain is $N_domain, N_boundary is $N_boundary"  

    lengthscale = args.sigma
    if args.kernel == "Matern5half"
        cov = KoLesky.MaternCovariance5_2(lengthscale)
    elseif args.kernel == "Matern7half"
        cov = KoLesky.MaternCovariance7_2(lengthscale)
    elseif args.kernel == "Matern9half"
        cov = KoLesky.MaternCovariance9_2(lengthscale)
    elseif args.kernel == "Matern11half"
        cov = KoLesky.MaternCovariance11_2(lengthscale)
    elseif args.kernel == "Gaussian"
        cov = KoLesky.GaussianCovariance(lengthscale)
    end
    @info "[kernel] choose $(args.kernel), lengthscale $lengthscale\n"  

    nugget = args.nugget::Float64
    @info "[nugget] $nugget" 

    GNsteps_approximate = args.GNsteps::Int
    @info "[total GN steps] $GNsteps_approximate" 

    ρ_big = args.rho_big::Float64
    ρ_small = args.rho_small::Float64
    k_neighbors = args.k_neighbors::Int
    @info "[Fast Cholesky] ρ_big = $ρ_big, ρ_small = $ρ_small, k_neighbors = $k_neighbors"

    sol_init = zeros(N_domain) # initial solution

    truth = [fun_u(X_domain[:,i]) for i in 1:N_domain]


    fast_solve() = @time iterGPR_fast_pcg(eqn, cov, X_domain, X_boundary, sol_init, nugget, GNsteps_approximate; ρ_big = ρ_big, ρ_small = ρ_small, k_neighbors=k_neighbors);
    sol = fast_solve()

    pts_accuracy = sqrt(sum((truth-sol).^2)/N_domain)
    @info "[L2 accuracy: pCG method] $pts_accuracy"
    pts_max_accuracy = maximum(abs.(truth-sol))
    @info "[Linf accuracy: pCG method] $pts_max_accuracy"

    # @btime @time iterGPR_fast_pcg($eqn, $cov, $X_domain, $X_boundary, $sol_init, $nugget, $GNsteps_approximate; ρ_big = $ρ_big, ρ_small = $ρ_small, k_neighbors=$k_neighbors);

    if args.compare_exact
        GNsteps_exact = 2
        @info "[comparison: exact method]"
        @time sol_exact = iterGPR_exact(eqn, cov, X_domain, X_boundary, sol_init, nugget, GNsteps_exact)
        pts_accuracy_exact = sqrt(sum((truth-sol_exact).^2)/N_domain)
        @info "[L2 accuracy: exact method] $pts_accuracy_exact"
        pts_max_accuracy_exact = maximum(abs.(truth-sol_exact))
        @info "[Linf accuracy: exact method] $pts_max_accuracy_exact"
    end

    
end

args = parse_commandline()
args = (; (Symbol(k) => v for (k,v) in args)...) # Named tuple from dict
main(args)
