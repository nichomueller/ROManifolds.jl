function assemble_reduced_affine_components(ROM_info, RB_variables, operators = nothing)
    #=MODIFY
    =#

    if isempty(RB_variables.Φₛᵘ) || maximum(abs.(RB_variables.Φₛᵘ)) === 0
        @error "Error: must generate or import spatial RBs before computing the reduced affine components"
    end

    if operators === nothing
        operators = ["A", "F"]
    end

    if "A" in operators

        if ROM_info.problem_nonlinearities["A"] === false

            @info "Assembling affine reduced stiffness"
            projection_time = @elapsed begin
                A = load_variable("A", "csv", joinpath(ROM_info.paths.FEM_structures_path, "A"))             
                RB_variables.Aₙ = (RB_variables.Φₛᵘ)' * A * RB_variables.Φₛᵘ
                if ROM_info.save_offline_structures 
                    save_variable(RB_variables.Aₙ, "Aₙ", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Aₙ"))
                end
            end    

        else

            @info "The stiffness is non-affine: running the MDEIM offline phase"
            projection_time = @elapsed begin
                Aₙ_i = sparse([],[],[])
                for i_nₛ = 1:maximum(10, ROM_info.nₛ)
                    parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
                    A_i = assemble_stiffness(FE_space, parametric_info, problem_info)
                    Aₙ_i = hcat(Aₙ_i, (RB_variables.Φₛᵘ)' * A_i * RB_variables.Φₛᵘ)
                end
                Aₙ_i = reshape(Aₙ_i, :, 1)
                if ROM_info.save_offline_structures 
                    (RB_variables.Aₙ_affine, RB_variables.Aₙ_idx) = DEIM_offline(Aₙ_i, ROM_info.ϵˢ, true, ROM_info.paths.ROM_structures_path, "Aₙ_mdeim")
                else
                    (RB_variables.Aₙ_affine, RB_variables.Aₙ_idx) = DEIM_offline(Aₙ_i, ROM_info.ϵˢ)
                end
            end

        end

    end

    if "F" in operators

        if ROM_info.problem_nonlinearities["f"] === false || ROM_info.problem_nonlinearities["h"] === false

            @info "Assembling affine reduced forcing term"
            projection_time += @elapsed begin
                F = load_variable("F", "csv", joinpath(ROM_info.paths.FEM_structures_path, "F")) 
                RB_variables.Fₙ = (RB_variables.Φₛᵘ)' * F
                if ROM_info.save_offline_structures 
                    save_variable(RB_variables.Fₙ, "Fₙ", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Fₙ"))
                end
            end

        else

            @info "The forcing term is non-affine: running the DEIM offline phase"
            projection_time += @elapsed begin
                Fₙ_i = Float64[]
                for i_nₛ = 1:maximum(10, ROM_info.nₛ)
                    parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
                    F_i = assemble_forcing(FE_space, parametric_info, problem_info)
                    Fₙ_i = hcat(Fₙ_i, (RB_variables.Φₛᵘ)' * F_i)
                end
                if ROM_info.save_offline_structures 
                    (RB_variables.Fₙ_affine, RB_variables.Fₙ_idx) = DEIM_offline(Fₙ_i, ROM_info.ϵˢ, true, ROM_info.paths.ROM_structures_path, "Fₙ_deim")
                else
                    (RB_variables.Fₙ_affine, RB_variables.Fₙ_idx) = DEIM_offline(Fₙ_i, ROM_info.ϵˢ)
                end
            end
        end

    end

    RB_variables.offline_time += projection_time

end


function build_RB_approximation(ROM_info, RB_variables)
    #=MODIFY
    =#

    @info "Building $(ROM_info.RB_method) approximation with $(ROM_info.nₛ) snapshots and tolerances of $(ROM_info.ϵˢ) in space"

    if ROM_info.import_snapshots
        get_snapshot_matrix(ROM_info, RB_variables)
        import_snapshots_success = true
    else
        import_snapshots_success = false
    end

    if ROM_info.import_offline_structures
        import_reduced_basis(ROM_info, RB_variables)
        import_basis_success = true
    else
        import_basis_success = false
    end

    if !import_snapshots_success && !import_basis_success
        @error "Impossible to assemble the reduced problem if neither the snapshots nor the bases can be loaded"
    end

    if import_snapshots_success && !import_basis_success
        @info "Failed to import the reduced basis, building it via POD"
        build_reduced_basis(ROM_info, RB_variables)
    end

    if ROM_info.import_offline_structures 
        operators = check_reduced_affine_components(ROM_info, RB_variables)
        if !isempty(operators)
            assemble_reduced_affine_components(ROM_info, RB_variables, operators)
        end
    else
        assemble_reduced_affine_components(ROM_info, RB_variables)
    end

end


