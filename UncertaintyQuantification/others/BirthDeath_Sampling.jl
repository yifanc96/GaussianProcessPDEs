# not finished

function BirthDeath_ParticleSampling(logρ_post::Function,
                                    N_particles::IT,
                                    particle_dim::IT,
                                    particles_init::Array{FT,2},
                                    dt::FT,
                                    N_iter::IT,
                                    K::Function # kernel for density smoothing; choose to approximate Diracs
                                    ) where {FT<:AbstractFloat, IT<:Int}
    
    sampled_particles = Array{FT,3}(undef,(particle_dim, N_particles, N_iter+1)) # store results
    sampled_particles[:,:,1] = particles_init
    β = Array{FT,1}(undef,N_particles)

    for iter_t in 2:N_iter+1
        # calculate birth and death rate
        for cur_particle in 1:N_particles
            tmp = 0
            for other_particle in 1:N_particles
                tmp += K(sampled_particles[:,cur_particle, iter_t] - sampled_particles[:,other_particle, iter_t])
            end
            β[cur_particle] = log(tmp/N_particles) + logρ_post(sampled_particles[:,cur_particle, iter_t])
        end
        β = β - sum(β)/N_particles
        
        # kill and duplicate
        rdn = rand(N_particles)
        new_particles = Array{FT,3}(undef,(particle_dim, N_particles))
        # for p in 1:N_particles
        #     if β[]
    en
end
