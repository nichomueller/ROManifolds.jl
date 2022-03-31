include("../UTILS/general.jl")


function POD(S, ϵ = 1e-5, norm_matrix = nothing)
    #=MODIFY
    =#

    if !(norm_matrix === nothing)
        if issparse(norm_matrix) === false
            norm_matrix = sparse(norm_matrix)
        end

        H = cholesky(norm_matrix)
        mul!(S, H, S)
    end

    if issparse(S)
        U, Σ, _ = svds(S; nsv = size(S)[2] - 1)[1]
    else
        U, Σ, _ = svd(S)
    end

    total_energy = sum(Σ .^ 2)
    cumulative_energy = 0.0
    N = 1

    while cumulative_energy / total_energy < 1. - ϵ * 2 && N < size(S)[2] - 1
        @info "POD loop number $N, cumulative energy = $cumulative_energy"
        cumulative_energy += Σ[N]^2
        N += 1
    end

    @info "Basis number obtained via POD is $N, projection error <= $sqrt(1-(cumulative_energy / total_energy))"

    if issparse(U)
        U = Matrix(U)
    end

    if !(norm_matrix === nothing)
        return H \ U[:, 1:N]
    else
        return U[:, 1:N]
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

    while cumulative_energy / total_energy < 1. - ϵ * 2 && N <= size(B)[2]
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


function DEIM_offline(S, ϵ = 1e-5, save_to_file = false, save_path = nothing, var_name = nothing)
    #=MODIFY
    =#

    @assert !(save_to_file === true && save_path === nothing) "Provide valid path to save the DEIM matrix and DEIM indices"
   
    basis = POD(S, ϵ)

    (N, n) = size(basis)
    DEIM_idx = zeros(Int64, n)

    DEIM_idx[1] = convert(Int64, argmax(abs.(basis[:, 1]))[1])

    #= if n > 2
        res = basis[:, 2] - basis[:, 1] * basis[DEIM_idx[1], 2] / basis[DEIM_idx[1], 1]
        DEIM_idx[2] = convert(Int64, argmax(abs.(res))[1])
    end =# 

    #proj = zeros(N)
    for m in range(2, n)
        #mul!(proj, basis[:, 1:m], mul!(tmp, basis[DEIM_idx[1:m], 1:m], basis[DEIM_idx[1:m], m]))
        res = basis[:, m] - basis[:, 1:(m - 1)] * (basis[DEIM_idx[1:(m - 1)], 1:(m - 1)] \ basis[DEIM_idx[1:(m - 1)], m])
        #res = basis[:, m] - proj
        DEIM_idx[m] = convert(Int64, argmax(abs.(res))[1])
    end

    DEIM_mat = basis#[DEIM_idx[:], :]
   
    if save_to_file
        @info "Offline phase of DEIM and MDEIM are the same: be careful with the path to which the (M)DEIM matrix and indices are saved"
        #CSV.write(save_path, DEIM_mat)
        #CSV.write(save_path, DEIM_idx)
        save_variable(DEIM_mat, var_name, "csv", save_path)
        save_variable(DEIM_idx, var_name * "_idx", "csv", save_path)
    end

    (DEIM_mat, DEIM_idx)

end


function DEIM_online(vec_nonaffine, DEIM_mat, DEIM_idx, save_path = nothing, save_to_file = false)
    #=MODIFY
    =#

    vec_affine = zeros(size(vec_nonaffine))

    #= if isfile(joinpath(DEIM_MDEIM_path, "DEIM_mat")) && isfile(joinpath(DEIM_MDEIM_path, "DEIM_idx"))
        DEIM_mat = load_variable("DEIM_mat", "csv", DEIM_MDEIM_path)
        DEIM_idx = load_variable("DEIM_idx", "csv", DEIM_MDEIM_path)
    else
        @error "Error: cannot read the DEIM vector, must run the DEIM offline phase!"
    end =#
    
    DEIM_coeffs = DEIM_mat[DEIM_idx[:], :] \ vec_nonaffine[DEIM_idx[:]]
    mul!(vec_affine, DEIM_mat, DEIM_coeffs)

    if save_to_file
        save_variable(DEIM_coeffs, "DEIM_coeffs", "jld", save_path)
        #save_variable(vec_affine, "vec_affine", "jld", save_path)
    end

    (DEIM_coeffs, vec_affine)

end


function MDEIM_online(mat_nonaffine, MDEIM_mat, MDEIM_idx, save_path = nothing, save_to_file = false)
    #=MODIFY
    S is already in the correct format, so it is a matrix of size (R*C, quantity), while mat_nonaffine is of size (R, C)
    =#
    
    (R, C) = size(mat_nonaffine)
    vec_affine = zeros(R * C, 1)

    #= if isfile(joinpath(DEIM_MDEIM_path, "MDEIM_mat")) && isfile(joinpath(DEIM_MDEIM_path, "MDEIM_idx"))
        MDEIM_mat = load_variable("MDEIM_mat", "csv", DEIM_MDEIM_path)
        MDEIM_idx = load_variable("MDEIM_idx", "csv", DEIM_MDEIM_path)
    else
        @error "Error: cannot read the MDEIM matrix, must run the DEIM offline phase!"
    end =#
    
    MDEIM_coeffs = MDEIM_mat[MDEIM_idx[:], :] \ reshape(mat_nonaffine, R * C, 1)[MDEIM_idx[:]]
    mul!(vec_affine, MDEIM_mat, MDEIM_coeffs)
    mat_affine = reshape(vec_affine, R, C)

    if save_to_file
        save_variable(MDEIM_coeffs, "MDEIM_coeffs", "jld", save_path)
        #save_variable(mat_affine, "mat_affine", "jld", save_path)
    end
    
    (MDEIM_coeffs, mat_affine)

end


#= function import_DEIM_MDEIM_structures(DEIM_MDEIM_path)
    #=MODIFY
    =#

    if isfile(joinpath(DEIM_MDEIM_path, "MDEIM_mat.csv")) && isfile(joinpath(DEIM_MDEIM_path, "MDEIM_idx.csv"))
        path = joinpath(DEIM_MDEIM_path, "MDEIM_mat.csv")
        MDEIM_mat = Matrix(CSV.read(path, DataFrame))
        path = joinpath(DEIM_MDEIM_path, "MDEIM_idx.csv")
        MDEIM_idx = Matrix(CSV.read(path, DataFrame))
        return (MDEIM_mat, MDEIM_idx)
    elseif isfile(joinpath(DEIM_MDEIM_path, "DEIM_mat")) && isfile(joinpath(DEIM_MDEIM_path, "DEIM_idx"))
        path = joinpath(DEIM_MDEIM_path, "DEIM_mat.csv")
        DEIM_mat = Matrix(CSV.read(path, DataFrame))
        path = joinpath(DEIM_MDEIM_path, "DEIM_idx.csv")
        DEIM_idx = Matrix(CSV.read(path, DataFrame))
        return (DEIM_mat, DEIM_idx)
    else
        @error "Error: cannot read the (M)DEIM structures, must run the (M)DEIM offline phase!"
    end
    
end =#