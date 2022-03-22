include("../UTILS/general.jl")

function POD(S, tol = 1e-5, norm_matrix = nothing)
    #=MODIFY
    =#

    if !(norm_matrix === nothing)
        if issparse(norm_matrix) === false
            norm_matrix = sparse(norm_matrix)
        end

        H = cholesky(norm_matrix)
        mul!(S, H, S)
    end

    U, Σ, _ = svd(S)
    total_energy = sum(Σ .^ 2)
    cumulative_energy = 0.0
    N = 1

    while cumulative_energy / total_energy < 1. - tol * 2 && N <= shape(S)[2]
        @info "POD loop number $N, cumulative energy = $cumulative_energy"
        cumulative_energy += Σ[N]^2
        N += 1
    end

    @info "Basis number obtained via POD is $N, projection error <= $sqrt(1-(cumulative_energy / total_energy))"
    if !(norm_matrix === nothing)
        return H \ U[:, N]
    else
        return U[:, N]
    end

end


function rPOD(S, ϵ = 1e-5, norm_matrix = nothing, q = 1, m = nothing)
    #=MODIFY
    =#

    if m === nothing
        m = size(S)[2] ./ 2
    end

    Ω = randn!(size(S)[1], m)
    (SS, SΩ, Y, B) = (zeros(size(S)[1], size(S)[1]), zeros(size(S)[1], size(Ω)[2]), similar(Ω), zeros(size(Ω)[2], size(S)[2]))
    mul!(Y, mul!(SS, S, S') ^ q, mul!(SΩ, S, Ω))
    (Q, R) = qr!(Y)
    mul!(B, Q', S)
    
    if !(norm_matrix === nothing)
        if issparse(norm_matrix) === false
            norm_matrix = sparse(norm_matrix)
        end

        H = cholesky(norm_matrix)  
        mul!(B, H, B)    
    end

    U, Σ, _ = svd(B)
    total_energy = sum(Σ .^ 2)
    cumulative_energy = 0.0
    N = 1

    while cumulative_energy / total_energy < 1. - ϵ * 2 && N <= shape(B)[2]
        @info "POD loop number $N, cumulative energy = $cumulative_energy"
        cumulative_energy += Σ[N]^2
        N += 1
    end

    @info "Basis number obtained via POD is $N, projection error <= $sqrt(1-(cumulative_energy / total_energy))"
    V = zeros(size(U[:, N]))
    mul!(V, Q, U[:, N])
    if !(norm_matrix === nothing)   
        return H \ V
    else
        return V
    end
    
end


