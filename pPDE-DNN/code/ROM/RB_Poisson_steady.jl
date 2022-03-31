include("../UTILS/general.jl")

mutable struct Poisson_RB_specifics
    Sᵘ
    Nᵤˢ
    Φₛᵘ
    nₛᵘ
    ũ
    uₙ
    û
    Aₙ
    Aₙ_affine
    Aₙ_idx
    LHSₙ
    Fₙ
    Fₙ_affine
    Fₙ_idx
    RHSₙ
    Xᵘ
    offline_time
end 


function initialize_RB_variables_struct()
    #=MODIFY
    =#

    Sᵘ = Array{Float64}(undef, 0, 0)
    Nᵤˢ = 0
    Φₛᵘ = Array{Float64}(undef, 0, 0)
    nₛᵘ = 0   
    
    ũ = Float64[]
    uₙ = Float64[]
    û = Float64[]
    
    Aₙ = Array{Float64}(undef, 0, 0)
    Aₙ_affine = Array{Float64}(undef, 0, 0)
    Aₙ_idx = Float64[]
    LHSₙ = Matrix{Float64}[]
    Fₙ = Float64[]
    Fₙ_affine = Float64[]
    Fₙ_idx = Float64[]
    RHSₙ = Matrix{Float64}[]
    Xᵘ = sparse([],[],[])
    
    offline_time = 0.0

    return RB_variables_struct(Sᵘ, Nᵤˢ, Φₛᵘ, nₛᵘ, ũ, uₙ, û, Aₙ, Aₙ_affine, Aₙ_idx, LHSₙ, Fₙ, Fₙ_affine, Fₙ_idx, RHSₙ, Xᵘ, offline_time)

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
        Sᵘ = Matrix(CSV.read(ROM_info.FEM_snap_path * var * ".csv", DataFrame))[:, 1:ROM_info.nₛ]
    catch e
        println("Error: $e. Impossible to load the snapshots matrix")
    end
    
    @info "Dimension of snapshot matrix: $(size(Sᵘ))"
    
    RB_variables.Sᵘ = Sᵘ
    RB_variables.Nᵤˢ = size(Sᵘ)[1]

end


function get_norm_matrix(ROM_info, RB_variables)
    #=MODIFY
    =#

    @info "Importing the norm matrix"

    if check_norm_matrix(RB_variables) 
            
        try 
            Xᵘ = Matrix(CSV.read(ROM_info.FEM_snap_path * "Xᵘ.csv", DataFrame))  
        catch e
            println("Error: $e. Impossible to load the H1 norm matrix")
        end
        
    end

    RB_variables.Xᵘ = Xᵘ

end


function check_norm_matrix(RB_variables)
    #=MODIFY
    =#

    isempty(RB_variables.Xᵘ) || maximum(abs.(RB_variables.Xᵘ)) === 0

end


function preprocess()
    #=MODIFY
    =#

end


function set_to_zero_RB_times(RB_variables)
    #=MODIFY
    =#

    RB_variables.offline_time = 0.0
    RB_variables.online_time = 0.0

end


function PODs_space(ROM_info, RB_variables)
    #=MODIFY
    =#

    @info "Performing the spatial POD for field u, using a tolerance of $ROM_info.ϵˢ"

    get_norm_matrix(ROM_info, RB_variables)
    Φₛᵘ = POD(RB_variables.Sᵘ, ROM_info.ϵˢ, RB_variables.Xᵘ)
    nₛᵘ = size(Φₛᵘ)[2]

    return (Φₛᵘ, nₛᵘ)
    
end


function build_reduced_basis(ROM_info, RB_variables)
    #=MODIFY
    =#

    @info "Building the reduced basis, using a tolerance of $(ROM_info.ϵˢ)"

    RB_building_time = @elapsed begin
        (Φₛᵘ, nₛᵘ) = PODs_space(ROM_info, RB_variables)
    end

    (RB_variables.Φₛᵘ, RB_variables.nₛᵘ) = (Φₛᵘ, nₛᵘ)
    RB_variables.offline_time += RB_building_time

    if ROM_info.save_offline_structures 
        save_variable(Φₛᵘ, "Φₛᵘ", "csv", joinpath(ROM_info.paths.basis_path, "Φₛᵘ"))
    end

end


function import_reduced_basis(ROM_info, RB_variables)
    #=MODIFY
    =#

    @info "Importing the reduced basis"

    Φₛᵘ = load_variable("Φₛᵘ", "csv", joinpath(ROM_info.paths.basis_path, "Φₛᵘ"))   
    RB_variables.Φₛᵘ = Φₛᵘ
    (RB_variables.Nₛᵘ, RB_variables.nₛᵘ) = size(Φₛᵘ)

end


function check_reduced_affine_components(ROM_info, RB_variables)
    #=MODIFY
    =#

    operators = []

    if ROM_info.problem_nonlinearities["A"] === false

        if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ.csv"))
            @info "Importing reduced affine LHS matrix"
            RB_variables.Aₙ = load_variable("Aₙ", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Aₙ")) 
        else
            @info "Failed to import the reduced affine LHS matrix: must build it"
            push!(operators, "A")
        end

    else

        if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_affine.csv")) && isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_idx.csv"))
            @info "Importing MDEIM offline structures for the LHS matrix"
            RB_variables.Aₙ_affine = load_variable("Aₙ_affine", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_affine")) 
            RB_variables.Aₙ_idx = load_variable("Aₙ_idx", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_idx")) 
        else
            @info "Failed to import MDEIM offline structures for the LHS matrix: must build them"
            push!(operators, "A")
        end

    if (ROM_info.problem_nonlinearities["f"] === true || ROM_info.problem_nonlinearities["h"] === true)

        if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ.csv"))
            @info "Importing reduced affine RHS vector"
            RB_variables.Fₙ = load_variable("Fₙ", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Fₙ")) 
        else
            @info "Failed to import the reduced affine RHS vector: must build it"
            push!(operators, "F")
        end

    else

        if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_affine.csv")) && isfile(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_idx.csv"))
            @info "Importing DEIM offline structures for the RHS vector"
            RB_variables.Aₙ_affine = load_variable("Fₙ_affine", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_affine")) 
            RB_variables.Aₙ_idx = load_variable("Fₙ_idx", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_idx")) 
        else
            @info "Failed to import DEIM offline structures for the RHS vector: must build them"
            push!(operators, "F")
        end

    end

    operators

end


#= function get_reduced_affine_components(ROM_info, RB_variables)
    #=MODIFY
    =#

    @info "Importing affine stiffness matrix"

    if ROM_info.problem_nonlinearities["A"] === false
        RB_variables.Aₙ = load_variable("Aₙ", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Aₙ")) 
    else
        RB_variables.Aₙ_affine = load_variable("Aₙ_mdeim", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_mdeim")) 
        RB_variables.Aₙ_idx = load_variable("Aₙ_mdeim_idx", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_mdeim_idx")) 
    end

    @info "Importing affine forcing term"

    if ROM_info.problem_nonlinearities["F"] === false
        RB_variables.Fₙ = load_variable("Fₙ", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Fₙ")) 
    else
        RB_variables.Fₙ_affine = load_variable("Fₙ_deim", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_deim")) 
        RB_variables.Fₙ_idx = load_variable("Fₙ_deim_idx", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_deim_idx")) 
    end


end =#


#= function check_and_return_DEIM_MDEIM(ROM_info, RB_variables)
    #=MODIFY
    =#

    if ROM_info.problem_nonlinearities["A"] === true && (isempty(RB_variables.Aₙ_affine) || maximum(abs.(RB_variables.Aₙ_affine)) === 0)
        if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_mdeim")) && isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_idx"))
            get_reduced_affine_components(ROM_info, RB_variables)
        else
            assemble_reduced_affine_components(ROM_info, RB_variables)
        end

        if (ROM_info.problem_nonlinearities["f"] === true || ROM_info.problem_nonlinearities["h"] === true) && (isempty(RB_variables.Fₙ_affine) || maximum(abs.(RB_variables.Fₙ_affine)) === 0)
            get_reduced_affine_components(ROM_info, RB_variables)
        else
            assemble_reduced_affine_components(ROM_info, RB_variables)
        end

    end

end =#


function get_generalized_coordinates(ROM_info, RB_variables, snaps = nothing)
    #=MODIFY
    =#

    if !check_norm_matrix(RB_variables)
        get_norm_matrix(ROM_info, RB_variables)
    end

    if snaps === nothing || maximum(snaps) > ROM_info.nₛ
        snaps = 1:ROM_info.nₛ
    end

    û = zeros(RB_variables.nₛᵘ, length(snaps))
    Φₛᵘ_normed = RB_variables.Xᵘ * RB_variables.Φₛᵘ 

    for i_nₛ = snaps  
        @info "Assembling generalized coordinate relative to snapshot $(i_nₛ)"     
        û[:, i_nₛ] = Φₛᵘ_normed * RB_variables.Sᵘ[:, i_nₛ]
    end

    RB_variables.û = û

    if ROM_info.save_offline_structures
        save_variable(û, "û", "csv", joinpath(ROM_info.paths.gen_coords_path, "û"))
    end

end


function initialize_RB_system(RB_variables)
    #=MODIFY
    =#

    RB_variables.LHSₙ[1] = zeros(RB_variables.nₛᵘ, RB_variables.nₛᵘ)
    RB_variables.RHSₙ[1] = zeros(RB_variables.nₛᵘ)

end


function get_RB_system(ROM_info, RB_variables, param, FE_space = nothing, parametric_info = nothing, problem_info = nothing)
    #=MODIFY
    =#

    @info "Preparing the RB system: fetching online reduced structures"

    if ROM_info.problem_nonlinearities["Aₙ"] === false
        RB_variables.LHSₙ[1] = param * RB_variables.Aₙ
    else
        A_param = assemble_stiffness(FE_space, parametric_info, problem_info)
        (_, A_param_affine) = MDEIM_online(A_param, RB_variables.Aₙ_affine, RB_variables.Aₙ_idx)
        RB_variables.LHSₙ[1] = A_param_affine
    end

    if ROM_info.problem_nonlinearities["f"] === false && ROM_info.problem_nonlinearities["h"] === false
        RB_variables.RHSₙ[1] = RB_variables.Fₙ
    else   
        F_param = assemble_forcing(FE_space, parametric_info, problem_info)
        (_, F_param_affine) = DEIM_online(F_param, RB_variables.Fₙ_affine, RB_variables.Fₙ_idx)    
        RB_variables.RHSₙ[1] = F_param_affine
    end
    
end


function solve_RB_system(ROM_info, RB_variables; param = nothing, FE_space = nothing, parametric_info = nothing, problem_info = nothing)
    #=MODIFY
    =#

    if (ROM_info.problem_nonlinearities["A"] === true || ROM_info.problem_nonlinearities["f"] === true || ROM_info.problem_nonlinearities["h"] === true) 

        if FE_space === nothing || parametric_info === nothing || problem_info === nothing
            @error "When the problem is non-affine with respect to a parameter, must provide FOM info when fetching the RB linear system"
        end

    end

    if ROM_info.problem_nonlinearities["A"] === false && param === nothing
        @error "When the stiffness is affine with respect to a parameter, must provide the parameter's when fetching the RB linear system"
    end

    get_RB_system(ROM_info, RB_variables, param, FE_space, parametric_info, problem_info)

    @info "Solving RB problem via backslash"
    @info "Condition number of the system's matrix: $(cond(RB_variables.LHSₙ[1]))"
    RB_variables.uₙ = RB_variables.LHSₙ[1] \ RB_variables.RHSₙ[1]

end


function reconstruct_FEM_soluiton(RB_variables)
    #=MODIFY
    =#

    @info "Reconstructing FEM solution from the newly computed RB one"

    RB_variables.ũ = RB_variables.Φₛᵘ * RB_variables.uₙ

end


function testing_phase(ROM_info, RB_variables, param_nbs)
    #=MODIFY
    =#

    mean_H1_err = 0.0
    mean_online_time = 0.0
    mean_reconstruction_time = 0.0
    set_to_zero_RB_times(RB_variables)

    ũ_param = zeros(RB_variables.Nᵤˢ)
    uₙ_param = zeros(RB_variables.nᵤˢ)

    if ROM_info.problem_nonlinearities["A"] === false
        params = load_variable("α", "jld", joinpath(ROM_info.paths.FEM_structures_path, "params")) 
    end

    for param_nb in param_nbs
        @info "Considering parameter number: $param_nbs"

        uₕ_test = Matrix(CSV.read(ROM_info.FEM_snap_path * var * ".csv", DataFrame))[:, param_nb]

        mean_online_time += @elapsed begin
            if ROM_info.problem_nonlinearities["A"] === false
                solve_RB_system(ROM_info, RB_variables; param = params[param_nb, :])
            else
                solve_RB_system(ROM_info, RB_variables; FE_space = FE_space, parametric_info = parametric_info, problem_info = problem_info)
            end
        end

        mean_reconstruction_time += @elapsed begin
            reconstruct_FEM_soluiton(RB_variables)
        end

        pointwise_u_err = uₕ_test - RB_variables.ũ
        mean_H1_err += compute_errors(RB_variables, pointwise_u_err) / length(param_nbs)

        ũ_param = hcat(ũ_param, RB_variables.ũ)
        uₙ_param = hcat(uₙ_param, RB_variables.uₙ)

    end

    mean_online_time /= length(param_nbs)
    mean_reconstruction_time /= length(param_nbs)

    string_param_nbs = "params"
    for param_nb in param_nbs
        string_param_nbs *= "_" * string(param_nb)
    end

    if ROM_info.save_results

        if !ROM_info.import_offline_structures
            save_variable(RB_variables.offline_time, "offline_time", "csv", ROM_info.paths.results_path)
        end

        path = joinpath(ROM_info.paths.results_path, string_param_nbs)
        save_variable(ũ_param, "ũ", "csv", path)
        save_variable(uₙ_param, "uₙ", "csv", path)
        save_variable(mean_H1_err, "mean_H1_err", "csv", path)
        save_variable(mean_online_time, "mean_online_time", "csv", path)
        save_variable(mean_reconstruction_time, "mean_reconstruction_time", "csv", path)

    end

end


function compute_errors(RB_variables, pointwise_u_err)
    #=MODIFY
    =#

    mynorm(pointwise_u_err, RB_variables.Xᵘ)

end

