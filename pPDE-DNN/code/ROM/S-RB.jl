include("../UTILS/general.jl")
include("affine_decomposition.jl")
include("DEIM-MDEIM.jl")
include("POD.jl")


function set_info(config_path) 
    #=MODIFY
    =#

    include(config_path)

    Nᵤˢ = FOM_specifics["space_dimension_FOM"][1]
    n_snaps = ROM_specifics["n_snapshots"]
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
    nₛᵘ = 0
    Aₙ = Array{Float64}(undef, 0, 2)
    Aₙ_affine = Matrix{Float64}[]
    fₙ = Float64[]
    fₙ_affine = Array{Float64}[]
    Wₙ = Float64[]

end


function get_snapshot_matrix()
    #=MODIFY
    =#

    @info "Importing the snapshot matrix, number of snapshots considered: $n_snap"

    path = FEM_snap_path
    variable = FOM_specifics["unknowns"][1]
    extension = FOM_files_extension
    delimiter = FOM_files_delimiter
    snap = nothing

    #=if n_snaps === nothing
        total_snaps = parse(Int64, String(split(get_full_subdirectories(path), "param")[end]))
        n_snaps = convert(Int64, floor(ROM_specifics["train_percentage"] .* total_snaps))
    end=#

    for i_ns = range(1, n_snaps)
        @info "Reading snapshot number $i_ns"
        path_i = joinpath(path, joinpath(variable, joinpath("param", string(i_ns))))
        cur_snap = load_variable(variable, extension, path_i, false, delimiter)
        
        if snap === nothing
            snap = cur_snap
        else
            snap = hcat(snap, cur_snap)
        end

    end

    @info "Dimension of snapshot matrix: $(size(snap))"
    snaps_matrix = snap

    return snaps_matrix

end


function get_norm_matrix()
    #=MODIFY
    =#

    path = FEM_structures_path
    variable = FOM_specifics["unknowns"][1]
    extension = FOM_files_extension
    delimiter = FOM_files_delimiter

    if isempty(Xᵤ) || maximum(abs(Xᵤ)) === 0
        @info "Importing the norm matrix"
        Xᵤ = load_variable(variable, extension, path, true, delimiter)       
    end
    return Xᵤ

end


function preprocess()
    #=MODIFY
    =#

end


function POD_space(S, ϵ)
    #=MODIFY
    =#

    @info "Performing the spatial POD, using a tolerance of $ϵ"
    Xᵤ = get_norm_matrix()
    basis_space = POD(S, ϵ, Xᵤ)
    nₛᵘ = size(basis_space)[2]
    return basis_space, nₛᵘ

end


function build_RB_space(S, ϵ)
    #=MODIFY
    =#

    RB_building_time = @elapsed begin
        basis_space, nₛᵘ = POD_space(S, ϵ)
    end

    if ROM_specifics["save_offline_structures"] === true
        save_variable(basis_space, "basis_space", "csv", joinpath(basis_path, "basis_space"))
    end
    if ROM_specifics["save_offline_structures"] === true
        save_variable(RB_building_time, "RB_building_time", "jld", joinpath(results_path, "time"))
    end

end


function import_RB_space()
    #=MODIFY
    =#

    return load_variable("basis_space", "csv", joinpath(basis_path, "basis_space"))   

end


function check_affine_decomposition()
    #=MODIFY
    =#

    @unpack (A_is_nonlin, f_is_nonlin) = problem_nonlinearities

    check = Dict("A" => true, "f" => true)

    if A_is_nonlin == true
        if isdir(joinpath(FEM_structures_path), "param_A" * FOM_files_extension)
            if isempty(load_variable("param_A", extension, joinpath(FEM_structures_path), "param_A" * FOM_files_extension))
                check["A"] = false
            end
        else
            check["A"] = false
        end
    end

    if f_is_nonlin == true
        if isdir(joinpath(FEM_structures_path), "param_f" * FOM_files_extension)
            if isempty(load_variable("param_f", extension, joinpath(FEM_structures_path), "param_f" * FOM_files_extension))
                check["f"] = false
            end
        else
            check["f"] = false
        end
    end

    return check

end


function get_parameters(extension, ϵ = 1e-5)
    #=MODIFY
    =#

    @unpack (A_is_nonlin, f_is_nonlin) = problem_nonlinearities
    
    if A_is_nonlin == false 
        if isdir(joinpath(FEM_structures_path), "param_A" * extension)
            θᴬ = load_variable("param_A", extension, joinpath(FEM_structures_path), "param_A" * extension)
        end
    end
    if f_is_nonlin == false 
        if isdir(joinpath(FEM_structures_path), "param_f" * extension)
            θᶠ = load_variable("param_f", extension, joinpath(FEM_structures_path), "param_f" * extension)
        end
    end
    
    if A_is_nonlin == true 
        
        n_A_snaps = minimum(10, n_snaps)

        @info "Importing the parametric stiffness matrix, number of parameters considered: $n_A_snaps"
        path = FEM_snap_path
        variable = FOM_specifics["unknowns"][1]
        extension = FOM_files_extension
        delimiter = FOM_files_delimiter
        A_snap = nothing

        for i_A = range(1, n_A_snaps)
                    
            @info "Reading parametric stiffness matrix number $i_A"
            path_Ai = joinpath(path, joinpath(variable, joinpath("param", string(i_A))))
            cur_A_snap = load_variable(variable, extension, path_Ai, false, delimiter)
            
            if A_snap === nothing
                A_snap = cur_A_snap[:]
            else
                A_snap = hcat(A_snap, cur_A_snap[:])
            end
       
        end
        @info "Dimension of reshaped parametric stiffness matrix: $(size(A_snap))"
        MDEIM(mat_nonaffine, A_snap, ϵ, nothing, true)

    end



    if f_is_nonlin == true 
        push!(nonlin_set, "f")
    end
    get_parameters_nonlin(nonlin_set)

end


function get_parameters_nonlin(nonlin_set)
    #=MODIFY
    =#

    @unpack (A_is_nonlin, f_is_nonlin) = problem_nonlinearities

    if A_is_nonlin === true

        n_A_snaps = minimum(10, n_snaps)

        for i = range(1, n_A_snaps)
            @info "Importing the snapshot matrix, number of snapshots considered: $n_snap"

            path = FEM_snap_path
            variable = FOM_specifics["unknowns"][1]
            extension = FOM_files_extension
            delimiter = FOM_files_delimiter
            snap = nothing

            #=if n_snaps === nothing
                total_snaps = parse(Int64, String(split(get_full_subdirectories(path), "param")[end]))
                n_snaps = convert(Int64, floor(ROM_specifics["train_percentage"] .* total_snaps))
            end=#

            for i_ns = range(1, n_snaps)
                @info "Reading snapshot number $i_ns"
                path_i = joinpath(path, joinpath(variable, joinpath("param", string(i_ns))))
                cur_snap = load_variable(variable, extension, path_i, false, delimiter)
                
                if snap === nothingalse
                    snap = cur_snap
                else
                    snap = hcat(snap, cur_snap)
                end

            end

            @info "Dimension of snapshot matrix: $(size(snap))"
            snaps_matrix = snap
            MDEIM(mat_nonaffine, S, ϵ, norm_matrix = nothing, save_to_file = false)

    return load_variable("basis_space", "txt", joinpath(basis_path, "basis_space")) 

end


function get_affine_components()
    #=MODIFY
    =#

    return load_variable("basis_space", "csv", joinpath(basis_path, "basis_space"))   

end



