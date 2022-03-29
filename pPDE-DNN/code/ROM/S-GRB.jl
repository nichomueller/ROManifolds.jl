
mutable struct RB_variables_struct
    Sᵘ
    Nᵤˢ
    θᴬ 
    Xᵘ 
    θᶠ
    ũ 

    Φₛᵘ
    nₛᵘ 
    Aₙ
    Aₙ_affine
    Aₙ_idx
    Fₙ 
    Fₙ_affine
    Fₙ_idx
    uₙ

    RB_offline_time
    RB_online_time
end 


function initialize_RB_variables_struct()
    #=MODIFY
    =#

    Sᵘ = Array{Float64}(undef, 0, 0)
    Nᵤˢ = 0
    Φₛᵘ = Array{Float64}(undef, 0, 0)
    nₛᵘ = 0
    
    Xᵘ = sparse([],[],[])
    
    ũ = Float64[]
    uₙ = Float64[]
    
    Aₙ = Array{Float64}(undef, 0, 0)
    Aₙ_affine = Array{Float64}(undef, 0, 0)
    Aₙ_idx = Float64[]
    Fₙ = Float64[]
    Fₙ_affine = Float64[]
    Fₙ_idx = Float64[]
    
    θᴬ = Float64[]
    θᶠ = Float64[]

    RB_offline_time = 0.0
    RB_online_time = 0.0

    return RB_variables_struct(Sᵘ, Nᵤˢ, θᴬ, Xᵘ, θᶠ, ũ, Φₛᵘ, nₛᵘ, Aₙ, Aₙ_affine, Aₙ_idx, Fₙ, Fₙ_affine, Fₙ_idx, uₙ, RB_offline_time, RB_online_time)

end
#= function RB_variables_function(Nᵤˢ) 
    #=MODIFY
    =#

    A = spzeros(Nᵤˢ, Nᵤˢ)
    A_affine = spzeros(Nᵤˢ, Nᵤˢ)
    θᴬ = Array{Float64}[]
    Xᵘ = spzeros(Nᵤˢ, Nᵤˢ)
    F = zeros(Nᵤˢ)
    F_affine = zeros(Nᵤˢ)
    θᶠ = Array{Float64}[]
    ũ = zeros(Nᵤˢ)

    basis_space = Array{Float64}(undef, 0, 2)
    nₛᵘ = 0
    Aₙ = Array{Float64}(undef, 0, 2)
    Aₙ_affine = Matrix{Float64}[]
    fₙ = Float64[]
    fₙ_affine = Array{Float64}[]
    uₙ = Float64[]

    _ -> (Nᵤˢ, nₛᵘ, A, A_affine, Xᵘ, F, F_affine, ũ, Aₙ, Aₙ_affine, fₙ, fₙ_affine, uₙ, basis_space, θᴬ, θᶠ)

end
_ -> (Nᵤˢ, nₛᵘ, A, A_affine, Xᵘ, F, F_affine, ũ, Aₙ, Aₙ_affine, fₙ, fₙ_affine, uₙ, basis_space, θᴬ, θᶠ) =#


function get_snapshot_matrix(ROM_info, RB_variables)
    #=MODIFY
    =#

    @info "Importing the snapshot matrix, number of snapshots considered: $n_snap"

    var = "uₕ"
    try
        Sᵘ = Matrix(CSV.read(ROM_info.FEM_snap_path * var * ".csv", DataFrame))
        if ROM_info.nₛ < size(Sᵘ)[2]
            Sᵘ = snaps_matrix[:, ROM_info.nₛ]
        end
    catch e
        println("Error: $e. Impossible to load the snapshots matrix")
    end
    
    @info "Dimension of snapshot matrix: $(size(Sᵘ))"
    
    RB_variables.Sᵘ = Sᵘ
    RB_variables.Nᵤˢ = Nᵤˢ

end


function get_norm_matrix(ROM_info, RB_variables)
    #=MODIFY
    =#

    @info "Importing the norm matrix"

    if isempty(RB_variables.Xᵘ) || maximum(abs.(RB_variables.Xᵘ)) === 0  
            
        try 
            Xᵘ = Matrix(CSV.read(ROM_info.FEM_snap_path * "Xᵘ.csv", DataFrame))  
        catch e
            println("Error: $e. Impossible to load the H1 norm matrix")
        end
        
    end

    RB_variables.Xᵘ = Xᵘ

end


function preprocess()
    #=MODIFY
    =#

end


function set_to_zero_RB_times(RB_variables)
    #=MODIFY
    =#

    RB_variables.RB_offline_time = 0.0
    RB_variables.RB_online_time = 0.0

end


function PODs_space(ROM_info, RB_variables)
    #=MODIFY
    =#

    @info "Performing the spatial POD for field u, using a tolerance of $ROM_info.ϵˢ"

    get_norm_matrix(ROM_info, RB_variables)
    Φₛᵘ = POD(Sᵘ, ROM_info.ϵˢ, Xᵘ)
    nₛᵘ = size(Φₛᵘ)[2]

    return (Φₛᵘ, nₛᵘ)
    
end


function build_RB_subspace(ROM_info, RB_variables)
    #=MODIFY
    =#

    @info "Building the RB subspace, using a tolerance of $(ROM_info.ϵˢ)"

    RB_building_time = @elapsed begin
        (Φₛᵘ, nₛᵘ) = PODs_space(ROM_info, RB_variables)
    end

    (RB_variables.Φₛᵘ, RB_variables.nₛᵘ) = (Φₛᵘ, nₛᵘ)
    RB_variables.RB_offline_time += RB_building_time

    if ROM_info.save_offline_structures 
        save_variable(Φₛᵘ, "Φₛᵘ", "csv", joinpath(ROM_info.paths.basis_path, "Φₛᵘ"))
    end

end


function import_RB_subspace(ROM_info, RB_variables)
    #=MODIFY
    =#

    @info "Importing the desired RB"

    Φₛᵘ = load_variable("Φₛᵘ", "csv", joinpath(ROM_info.paths.basis_path, "Φₛᵘ"))   
    (RB_variables.Φₛᵘ, RB_variables.nₛᵘ) = (Φₛᵘ, size(Φₛᵘ)[1])

end


function get_reduced_affine_components(ROM_info, RB_variables)
    #=MODIFY
    =#

    if ROM_info.problem_nonlinearities["A"] === false
        A = load_variable("A", "csv", joinpath(ROM_info.paths.FEM_structures_path, "A")) 
        RB_variables.Aₙ = (RB_variables.Φₛᵘ)' * A * RB_variables.Φₛᵘ
    else
        Aₙ = sparse([],[],[])
        for i_nₛ = 1:maximum(10, ROM_info.nₛ)
            parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
            A_i = assemble_stiffness(FE_space, parametric_info, problem_info)[:]
            Aₙ = hcat(Aₙ, (RB_variables.Φₛᵘ)' * A_i * RB_variables.Φₛᵘ)
        end
        if ROM_info.save_offline_structures 
            (RB_variables.Aₙ_affine, RB_variables.Aₙ_idx) = DEIM_offline(Aₙ_i, ROM_info.ϵˢ, true, ROM_info.paths.FEM_structures_path, "Aₙ_mdeim")
        else
            (RB_variables.Aₙ_affine, RB_variables.Aₙ_idx) = DEIM_offline(Aₙ_i, ROM_info.ϵˢ)
        end
    end

    if ROM_info.problem_nonlinearities["f"] === false || ROM_info.problem_nonlinearities["h"] === false
        F = load_variable("F", "csv", joinpath(ROM_info.paths.FEM_structures_path, "F")) 
        RB_variables.Fₙ = (RB_variables.Φₛᵘ)' * F
    else
        Fₙ = Float64[]
        for i_nₛ = 1:maximum(10, ROM_info.nₛ)
            parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
            F_i = assemble_forcing(FE_space, parametric_info, problem_info)
            Fₙ = hcat(Fₙ, (RB_variables.Φₛᵘ)' * F_i)
        end
        if ROM_info.save_offline_structures 
            (RB_variables.Fₙ_affine, RB_variables.Fₙ_idx) = DEIM_offline(Fₙ_i, ROM_info.ϵˢ, true, ROM_info.paths.FEM_structures_path, "Fₙ_deim")
        else
            (RB_variables.Fₙ_affine, RB_variables.Fₙ_idx) = DEIM_offline(Fₙ_i, ROM_info.ϵˢ)
        end
    end


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



