#=
    solve Darcy equation with finite difference method:
    -∇(κ∇h) = f
    with Dirichlet boundary condition, h=0 on ∂Ω
=#

mutable struct Setup_Param{FT<:AbstractFloat, IT<:Int}
    # physics
    N::IT         # number of grid points for both x and y directions (including both ends)
    Δx::FT
    #for source term
    f_2d::Array{FT, 2}
end

function solve_Darcy_2D(darcy::Setup_Param{FT, IT}, κ_2d::Array{FT,2}) where {FT<:AbstractFloat, IT<:Int}
    Δx, N = darcy.Δx, darcy.N
    
    indx = IT[]
    indy = IT[]
    vals = FT[]
    
    f_2d = darcy.f_2d
    
    𝓒 = Δx^2
    for iy = 2:N-1
        for ix = 2:N-1
            
            ixy = ind(darcy, ix, iy) 
            
            #top
            if iy == N-1
                #ft = -(κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0 * (0 - h_2d[ix,iy])
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals, (κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0/𝓒)
            else
                #ft = -(κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0 * (h_2d[ix,iy+1] - h_2d[ix,iy])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix, iy+1)])
                append!(vals, [(κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0/𝓒, -(κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0/𝓒])
            end
            
            #bottom
            if iy == 2
                #fb = (κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0 * (h_2d[ix,iy] - 0)
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals,  (κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0/𝓒)
            else
                #fb = (κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0 * (h_2d[ix,iy] - h_2d[ix,iy-1])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix, iy-1)])
                append!(vals, [(κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0/𝓒, -(κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0/𝓒])
            end
            
            #right
            if ix == N-1
                #fr = -(κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0 * (0 - h_2d[ix,iy])
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals, (κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0/𝓒)
            else
                #fr = -(κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0 * (h_2d[ix+1,iy] - h_2d[ix,iy])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix+1, iy)])
                append!(vals, [(κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0/𝓒, -(κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0/𝓒])
            end  
            
            #left
            if ix == 2
                #fl = (κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0 * (h_2d[ix,iy] - 0)
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals, (κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0/𝓒)
            else
                #fl = (κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0 * (h_2d[ix,iy] - h_2d[ix-1,iy])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix-1, iy)])
                append!(vals, [(κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0/𝓒, -(κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0/𝓒])
            end
            
        end
    end
    
    
    
    df = sparse(indx, indy, vals, (N-2)^2, (N-2)^2)
    # Multithread does not support sparse matrix solver
    h = df\(f_2d[2:N-1,2:N-1])[:]
    
    h_2d = zeros(FT, N, N)
    h_2d[2:N-1,2:N-1] .= reshape(h, N-2, N-2) 
    
    return h_2d
end


function ind(darcy::Setup_Param{FT, IT}, ix::IT, iy::IT) where {FT<:AbstractFloat, IT<:Int}
    return (ix-1) + (iy-2)*(darcy.N - 2)
end

function ind_all(darcy::Setup_Param{FT, IT}, ix::IT, iy::IT) where {FT<:AbstractFloat, IT<:Int}
    return ix + (iy-1)*darcy.N
end