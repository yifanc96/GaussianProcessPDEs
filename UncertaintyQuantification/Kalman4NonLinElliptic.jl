using LinearAlgebra
"""
ExKIObj{FT<:AbstractFloat, IT<:Int}
Extended Kalman Inversion struct (ExKI)
For solving the inverse problem 
    y = G(θ) + η
    
"""
mutable struct ExKIObj{FT<:AbstractFloat, IT<:Int}
    "a vector of arrays of size  N_parameters containing the mean of the parameters (in each exki iteration a new array of mean is added)"
     θ_mean::Vector{Array{FT, 1}}
     "a vector of arrays of size (N_parameters x N_parameters) containing the covariance of the parameters (in each exki iteration a new array of cov is added)"
     θθ_cov::Vector{Array{FT, 2}}
     "a vector of arrays of size N_y containing the predicted observation (in each exki iteration a new array of predicted observation is added)"
     y_pred::Vector{Array{FT, 1}}
     "vector of observations (length: N_y)"
     y::Array{FT, 1}
     "covariance of the observational noise"
     Σ_η::Array{FT, 2}
     "size of θ"
     N_θ::IT
     "size of y"
     N_y::IT
     "prior mean"
     r_0::Array{FT, 1}
     "prior covariance"
     Σ_0::Array{FT, 2}
     "hyperparameter"
     γ::FT
     "current iteration number"
     iter::IT
end



"""
ExKIObj Constructor 
N_iter: number of iterations
y::Array{FT,1} : observation
Σ_η::Array{FT, 2} : observation error covariance
γ::FT : hyperparameter
r_0::Array{FT, 1} : prior mean 
Σ_0::Array{FT, 2} : prior covariance
θ0_mean::Array{FT, 1} : initial mean
θθ0_cov::Array{FT, 2} : initial covariance
"""
function ExKIObj(N_iter::IT,
                 y::Array{FT,1},
                 Σ_η::Array{FT, 2},
                 γ::FT,
                 r_0::Array{FT, 1},
                 Σ_0::Array{FT, 2},
                 θ0_mean::Array{FT,1} = r_0, 
                 θθ0_cov::Array{FT, 2} = Σ_0) where {FT<:AbstractFloat, IT<:Int}

    
    N_θ = size(θ0_mean,1)
    N_y = size(y, 1)


    θ_mean = Array{FT,1}[]  # array of Array{FT, 2}'s
    push!(θ_mean, θ0_mean) # insert parameters at end of array (in this case just 1st entry)
    θθ_cov = Array{FT,2}[] # array of Array{FT, 2}'s
    push!(θθ_cov, θθ0_cov) # insert parameters at end of array (in this case just 1st entry)

    y_pred = Array{FT, 1}[]  # array of Array{FT, 2}'s
   

    iter = IT(0)

    ExKIObj{FT,IT}(θ_mean, θθ_cov, y_pred, 
                  y,   Σ_η, 
                  N_θ, N_y, 
                  r_0, Σ_0, 
                  γ, iter)

end




"""
update exki struct
forward(θ) = G(θ) dG(θ)
"""
function update_ensemble!(exki::ExKIObj{FT, IT}, forward::Function) where {FT<:AbstractFloat, IT<:Int}
    
    exki.iter += 1

    θ_mean  = exki.θ_mean[end]
    θθ_cov = exki.θθ_cov[end]
    y = exki.y
    γ = exki.γ
    r_0, Σ_0 = exki.r_0, exki.Σ_0
    N_θ, N_y = exki.N_θ, exki.N_y
    ############# Prediction step:

    θ_p_mean  = θ_mean
    θθ_p_cov =  (γ + 1)*θθ_cov
    

    g_mean, dg = forward(θ_p_mean)
    θg_cov = θθ_p_cov * dg'

    # extended system
    ff_cov = [ dg*θg_cov+(1 + 1/γ)*Σ_η  θg_cov';
                θg_cov     θθ_p_cov+(1 + 1/γ)*Σ_0] 

    θf_cov = [θg_cov   θθ_p_cov]

    tmp = θf_cov/ff_cov

    θ_mean =  θ_p_mean + tmp*([y ; r_0] - [g_mean; θ_p_mean])

    θθ_cov =  θθ_p_cov - tmp*θf_cov' 


    ########### Save resutls
    push!(exki.y_pred, g_mean) # N_ens x N_data
    push!(exki.θ_mean, θ_mean) # N_ens x N_params
    push!(exki.θθ_cov, θθ_cov) # N_ens x N_data
end



function ExKI_Run(s_param, forward::Function, 
    y, Σ_η,
    r_0,
    Σ_0,
    θ0_mean, θθ0_cov,
    N_iter,
    γ = 1.0)

    exkiobj = ExKIObj(
        N_iter,
        y,
        Σ_η,
        γ,
        r_0,
        Σ_0,
        θ0_mean, 
        θθ0_cov)

    func(θ) = forward(s_param, θ) 
    
    for i in 1:N_iter
        
        update_ensemble!(exkiobj, func) 

        @info "optimization error at iter $(i) = ", 0.5*(exkiobj.y_pred[i] - exkiobj.y)'*(exkiobj.Σ_η\(exkiobj.y_pred[i] - exkiobj.y))
        @info "Frobenius norm of the covariance at iter $(i) = ", norm(exkiobj.θθ_cov[i])
    end
    
    return exkiobj
    
end


function plot_param_iter(exkiobj::ExKIObj{FT, IT}, θ_ref::Array{FT,1}, θ_ref_names::Array{String}) where {FT<:AbstractFloat, IT<:Int}
    
    θ_mean = exkiobj.θ_mean
    θθ_cov = exkiobj.θθ_cov
    
    N_iter = length(θ_mean) - 1
    ites = Array(LinRange(1, N_iter+1, N_iter+1))
    
    θ_mean_arr = abs.(hcat(θ_mean...))
    
    
    N_θ = length(θ_ref)
    θθ_std_arr = zeros(Float64, (N_θ, N_iter+1))
    for i = 1:N_iter+1
        for j = 1:N_θ
            θθ_std_arr[j, i] = sqrt(θθ_cov[i][j,j])
        end
    end
    
    for i = 1:N_θ
        errorbar(ites, θ_mean_arr[i,:], yerr=3.0*θθ_std_arr[i,:], fmt="--o",fillstyle="none", label=θ_ref_names[i])   
        plot(ites, fill(θ_ref[i], N_iter+1), "--", color="gray")
    end
    
    xlabel("Iterations")
    legend()
    tight_layout()
end


function plot_opt_errors(exkiobj::ExKIObj{FT, IT}, 
    θ_ref::Union{Array{FT,1}, Nothing} = nothing, 
    transform_func::Union{Function, Nothing} = nothing) where {FT<:AbstractFloat, IT<:Int}
    
    θ_mean = exkiobj.θ_mean
    θθ_cov = exkiobj.θθ_cov
    y_pred = exkiobj.y_pred
    Σ_η = exkiobj.Σ_η
    y = exkiobj.y

    N_iter = length(θ_mean) - 1
    
    ites = Array(LinRange(1, N_iter, N_iter))
    N_subfigs = (θ_ref === nothing ? 2 : 3)

    errors = zeros(Float64, N_subfigs, N_iter)
    fig, ax = PyPlot.subplots(ncols=N_subfigs, figsize=(N_subfigs*6,6))
    for i = 1:N_iter
        errors[N_subfigs - 1, i] = 0.5*(y - y_pred[i])'*(Σ_η\(y - y_pred[i]))
        errors[N_subfigs, i]     = norm(θθ_cov[i])
        
        if N_subfigs == 3
            errors[1, i] = norm(θ_ref - (transform_func === nothing ? θ_mean[i] : transform_func(θ_mean[i])))/norm(θ_ref)
        end
        
    end

    markevery = max(div(N_iter, 10), 1)
    ax[N_subfigs - 1].plot(ites, errors[N_subfigs - 1, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
    ax[N_subfigs - 1].set_xlabel("Iterations")
    ax[N_subfigs - 1].set_ylabel("Optimization error")
    ax[N_subfigs - 1].grid()
    
    ax[N_subfigs].plot(ites, errors[N_subfigs, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
    ax[N_subfigs].set_xlabel("Iterations")
    ax[N_subfigs].set_ylabel("Frobenius norm of the covariance")
    ax[N_subfigs].grid()
    if N_subfigs == 3
        ax[1].set_xlabel("Iterations")
        ax[1].plot(ites, errors[1, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
        ax[1].set_ylabel("L₂ norm error")
        ax[1].grid()
    end
    
end

##################

function Two_Param_Linear_Test(problem_type::String, θ0_bar, θθ0_cov)
    
    N_θ = length(θ0_bar)

    
    if problem_type == "under-determined"
        # under-determined case
        θ_ref = [0.6, 1.2]
        G = [1.0 2.0;]
        
        y = [3.0;]
        Σ_η = Array(Diagonal(fill(0.1^2, size(y))))
        
        
    
    elseif problem_type == "over-determined"
        # over-determined case
        θ_ref = [1/3, 8.5/6]
        G = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        
        y = [3.0;7.0;10.0]
        Σ_η = Array(Diagonal(fill(0.1^2, size(y))))
        
    elseif problem_type == "Hilbert"

        G = zeros(N_θ, N_θ)
        for i = 1:N_θ
            for j = 1:N_θ
                G[i,j] = 1/(i + j - 1)
            end
        end
    
        θ_ref = fill(1.0, N_θ)
        y   = G*θ_ref 
        Σ_η = Array(Diagonal(fill(0.5^2, N_θ)))
        
        
    else
        error("Problem type : ", problem_type, " has not implemented!")
    end
    
    Σ_post = inv(G'*(Σ_η\G) + inv(θθ0_cov))
    θ_post = θ0_bar + Σ_post*(G'*(Σ_η\(y - G*θ0_bar)))
    

    return θ_post, Σ_post, G, y, Σ_η
end


struct Setup_Param{MAT, IT<:Int}
    G::MAT
    N_θ::IT
    N_y::IT
end

function forward(s_param::Setup_Param, θ::Array{FT, 1}) where {FT<:AbstractFloat}
    G = s_param.G 
    return G * θ, G
end

N_iter = 30
N_θ = 2

r_0, Σ_0 = zeros(Float64, N_θ), Array(Diagonal(fill(1.0^2, N_θ)))
θ0_mean, θθ0_cov = r_0, Σ_0

problem_type = "under-determined"
    
θ_post, Σ_post, G, y, Σ_η = Two_Param_Linear_Test(problem_type, r_0, Σ_0)

N_y = length(y)

s_param = Setup_Param(G, N_θ, N_y)

# UKI-1
exki_obj = ExKI_Run(s_param, forward, 
y, Σ_η,
r_0,
Σ_0,
θ0_mean, θθ0_cov,
N_iter,)

@info "ERRORs are :" , norm(exki_obj.θ_mean[end] - θ_post), norm(exki_obj.θθ_cov[end] - Σ_post)
    
    
    
 