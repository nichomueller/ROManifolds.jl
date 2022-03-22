include("../UTILS/general.jl")


function DEIM_online(vec_nonaffine, DEIM_MDEIM_path, save_path)
    #=MODIFY
    =#

    vec_affine = zeros(shape(vec_nonaffine))

    if isfile(joinpath(DEIM_MDEIM_path, "DEIM_mat")) && isfile(joinpath(DEIM_MDEIM_path, "DEIM_idx"))
        DEIM_mat = load_variable("DEIM_mat", "csv", DEIM_MDEIM_path)
        DEIM_idx = load_variable("DEIM_idx", "csv", DEIM_MDEIM_path)
    else
        @error "Error: cannot read the DEIM vector, must run the DEIM offline phase!"
    end
    
    DEIM_coeffs = DEIM_mat[DEIM_idx, :] \ vec_nonaffine[DEIM_idx]
    mul!(vec_affine, DEIM_mat, DEIM_coeffs)

    save_variable(DEIM_coeffs, "DEIM_coeffs", "jld", save_path)
    save_variable(vec_affine, "vec_affine", "jld", save_path)

end


function DEIM_offline(S, ϵ, save_path, norm_matrix = nothing)
    #=MODIFY
    =#

    @assert !(save_to_file === true && save_path === nothing) "Provide valid path to save the DEIM matrix and DEIM indices"
   
    basis = POD(S, ϵ, norm_matrix)

    n = shape(basis)[2]
    DEIM_idx = zeros(Int64, n)

    DEIM_idx[1] = convert(Int64, argmax(abs(basis[:, 1])))

    if n > 1
        res = self.M_basis[:, 2] - dot(basis[:, 1], basis[DEIM_idx[1], 2] / basis[DEIM_idx[1], 1])
        DEIM_idx[2] = convert(Int64, argmax(abs(res)))
    end

    proj = zeros(n)
    tmp = zeros(n)
    for m in range(3, n)
        mul!(proj, basis[:, 1:m], mul!(tmp, basis[DEIM_idx[1:m], 1:m], basis[DEIM_idx[1:m], m]))
        res = basis[:, m] - proj
        DEIM_idx[m] = convert(Int64, argmax(abs(res)))
    end

    DEIM_mat = basis[:, DEIM_idx]S, ϵ, norm_matrix = nothing, save_to_file = false, path = nothing
   
    save_variable(DEIM_mat, "DEIM_mat", "csv", save_path)
    save_variable(DEIM_idx, "DEIM_idx", "csv", save_path)

end


function MDEIM_online(mat_nonaffine, DEIM_MDEIM_path, save_path)
    #=MODIFY
    S is already in the correct format, so it is a matrix of size (R*C, quantity), while mat_nonaffine is of size (R, C)
    =#
    
    (R, C) = size(mat_nonaffine)
    vec_affine = zeros(R * C, 1)

    if isfile(joinpath(DEIM_MDEIM_path, "MDEIM_mat")) && isfile(joinpath(DEIM_MDEIM_path, "MDEIM_idx"))
        MDEIM_mat = load_variable("MDEIM_mat", "csv", DEIM_MDEIM_path)
        MDEIM_idx = load_variable("MDEIM_idx", "csv", DEIM_MDEIM_path)
    else
        @error "Error: cannot read the MDEIM matrix, must run the DEIM offline phase!"
    end
    
    MDEIM_coeffs = DEIM_mat[MDEIM_idx, :] \ reshape(mat_nonaffine, R * C, 1)[MDEIM_idx]
    mul!(vec_affine, MDEIM_mat, MDEIM_coeffs)
    mat_affine = reshape(vec_affine, R, C)

    save_variable(MDEIM_coeffs, "MDEIM_coeffs", "jld", save_path)
    save_variable(mat_affine, "mat_affine", "jld", save_path)
    
end


