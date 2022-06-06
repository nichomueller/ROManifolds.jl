using DataFrames
using Random
using Conda
using Pkg
using Interact
using LinearAlgebra
using FillArrays
using Test
using InteractiveUtils
using Plots
using Logging
using SuiteSparse, SparseArrays

#=using Flux, Flux.Data.MNIST
using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData=#

function create_dir(name)
    #=Create a directory at the given path
    :Param name: path of the directory to be created, if not already existing
    :type name: str
    =#

    if !isdir(name)
        mkdir(name)
    end

end

function get_full_subdirectories(rootdir)
    #=Get a full list of subdirectories at a given root directory
    :Param rootdir: path of the root directory whose subdirectories are searched and listed
    :type rootdir: str
    =#

    return filter(isdir, readdir(rootdir,join=true))

end

function save_matrix(matrix, file_name)
    #=Utility method, which allows to save 2D numpy arrays in text files. If the path to the text file does not exist,
    it displays an error message, but it does not raise any error.
    :Param matrix: matrix to be saved
    :type matrix: numpy.ndarray
    :Param file_name: path to the file where the matrix has to be saved, provided that the path is a valid path
    :type file_name: str
    =#

    try
        output_file = open(file_name, "w+")

        for iR in range(1, size(matrix)[0])
            for iC in range(1, size(matrix)[1])
                write(output_file, "%-10g" % matrix[iR, iC])

                if iC < matrix.shape[1]
                    write(output_file, " ")
                else
                    write(output_file, "\n")
                end

            end
        end
        close(output_file)

    catch e
        println("Error: $e. Impossible to save the desired matrix")
    end

end

function save_variable(var, variable_name, file_name)
    #=Utility method, which allows to save arrays/matrices in .jld files. If the path to the text file does not exist,
    it displays an error message.
    :Param var: variable to be saved
    :type var: AbstractMatrix, AbstactVector
    :Param variable_name: name with which we save the variable
    :type variable_name: str
    :Param file_name: path to the file where the variable has to be saved, provided that the path is a valid path; no extension should be provided
    :type file_name: str
    =#

    if occursin(".", file_name) && !issparse(var)
        (file_name, extension) = (String(split(file_name, ".")[1]), String(split(file_name, ".")[2]))
    else
        extension = "csv"
    end

    try
        if extension == "jld"
            jldopen(file_name * ".jld", "w") do file
                write(file, variable_name, var)
            end
        else
            open(file_name * ".csv", "w") do file
                if issparse(var)
                    i, j, v = findnz(var)
                    df = DataFrame([:i => i, :j => j, :v => v])
                    CSV.write(file, df)
                else
                    CSV.write(file, Tables.table(var))
                end
                close(file)
            end
        end
    catch e
        println("Error: $e. Impossible to save the desired variable")
    end
end

function load_variable(variable_name, file_name, path = nothing)
    #=Utility method, which allows to load arrays/matrices from .jld files. If the path to the text file does not exist,
    it displays an error message.
    :Param variable_name: name of the loaded variable
    :type variable_name: str
    :Param file_name: path to the file where the variable has to be loaded from, provided that the path is a valid path; no extension should be provided
    :type file_name: str
    :return the variable read from the .jld file
    :rtype: AbstractMatrix, AbstactVector
    =#

    if occursin(".", file_name)
        (file_name, extension) = (String(split(file_name, ".")[1]), String(split(file_name, ".")[2]))
    else
        extension = "csv"
    end

    if path === nothing
        path = pwd() * "/" * file_name * "." * extension
    end

    try
        if extension == "jld"
            c = jldopen(file_name * ".jld", "r") do file
                read(file, variable_name)
            end
        else
            c = Matrix(CSV.read(path, DataFrame))
        end
        return c
    catch e
        println("Error: $e. Impossible to load the desired variable")
    end
end

function convert_to_full(sparse_var, format = "csc")
    #= MODIFY
    =#

    try
        if format === "csc"
            var = Matrix(sparse_var)
        else
            i, j, v = findnz(sparse_var)
            var = zeros(maximum(i),maximum(j))
            for k in range(1, length(i))
                var[i[k], j[k]] = v[k]
            end
        return var
        end
    catch e
        println("Error: $e. Impossible to convert the input sparse matrix to an output full matrix")
    end
end



#=
function compute_DEIM_snapshot(Ω_p, quantity, vec_nonaffine_map)
    #=MODIFY
    =#

    Θ = Parameter_generator(Ω_p, quantity)

    return vec_nonaffine_map(Θ)
end


function compute_MDEIM_snapshot(Ω_p, quantity, mat_nonaffine_map)
    #=MODIFY
    =#

    Θ = Parameter_generator(Ω_p, quantity)
    (R, C) = size(mat_nonaffine_map(Θ[1]))
    snap = zeros(R * C, quantity)
    [snap[:, k] = reshape(mat_nonaffine_map(Θ[k])[:], R * C, quantity) for k = 1:quantity]

    return snap
end=#


function DEIM(Ω_p, S, ϵ, norm_matrix = nothing)
    #=MODIFY
    =#

    #S = compute_DEIM_snapshot(Ω_p, quantity, vec_nonaffine_map)
    basis = POD(S, ϵ, norm_matrix)
    DEIM_idx, DEIM_mat = DEIM_offline(basis)
    DEIM_coeffs = DEIM_mat[DEIM_idx, :] \ vec_nonaffine[DEIM_idx]
    mul!(vec_affine, DEIM_mat, DEIM_coeffs)

    return vec_affine, DEIM_mat, DEIM_coeffs

end


function DEIM_offline(basis)
    #=MODIFY
    =#

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

    DEIM_mat = basis[:, DEIM_idx]

    return DEIM_idx, DEIM_mat

end


function MDEIM(Ω_p, quantity, mat_nonaffine_map, mat_nonaffine, ϵ, norm_matrix = nothing)
    #=MODIFY
    =#

    S = compute_MDEIM_snapshot(Ω_p, quantity, mat_nonaffine_map)
    (R, C) = size(mat_nonaffine)
    basis = POD(S, ϵ, norm_matrix)
    DEIM_idx, DEIM_mat = DEIM_offline(basis)

    if issparse(mat_nonaffine)
        i, j, _ = findnz(mat_nonaffine)
    else
        (i, j) = [repeat(1:R, 1, C)[:], repeat((1:C)', R, 1)[:]]
    end
    function Parametric_domain(x)
        #=MODIFY
        assumed to be a parallelepiped
        =#

        @unpack (Ix, Iy, Iz) = x
        return Ix, Iy, Iz

    end


    function Parameter_generator(Ω_p, quantity)
        #=MODIFY
        =#

        Param = zeros(quantity, size(Ω_p)[1])
        for i in range(1, quantity)
            for j in range(1, size(Ω_p)[1])
                Param[i, j] = Uniform(Ω_p[j,:][1], Ω_p[j,:][2])
            end
        end
        return Param

    end

struct RB_info{T<:String}

    config_path::T
    #include(config_path)

    function set_info(config_path)
        #=MODIFY
        =#

        include(config_path)

        Nᵤˢ = FOM_Info["space_dimension_FOM"][1]
        n_snaps = ROM_Info["n_snapshots"]
        snaps_matrix = zeros(Nᵤˢ, n_snaps)
        A = zeros(Nᵤˢ, Nᵤˢ)
        A_affine = Matrix{Float64}[]
        θᴬ = Array{Float64}[]
        Xᵤ = zeros(Nᵤˢ, Nᵤˢ)
        F = zeros(Nᵤˢ)
        F_affine = Array{Float64}[]
        θᶠ = Array{Float64}[]
        W̃ = zeros(Nᵤˢ)

        basis_space = Array{Float64}(undef, 0, 2)
        Aₙ = Array{Float64}(undef, 0, 2)
        Aₙ_affine = Matrix{Float64}[]
        fₙ = []
        fₙ_affine = Array{Float64}[]
        Wₙ = []

    end

end

function Parametric_domain(x)
    #=MODIFY
    assumed to be a parallelepiped
    =#

    @unpack (Ix, Iy, Iz) = x
    return Ix, Iy, Iz

end


function Parameter_generator(Ω_p, quantity)
    #=MODIFY
    =#

    Param = zeros(quantity, size(Ω_p)[1])
    for i in range(1, quantity)
        for j in range(1, size(Ω_p)[1])
            Param[i, j] = Uniform(Ω_p[j,:][1], Ω_p[j,:][2])
        end
    end
    return Param

end
