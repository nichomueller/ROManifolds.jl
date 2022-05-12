include("RB_Poisson_steady.jl")

function assemble_reduced_affine_components(ROM_info, RB_variables::PoissonSTGRB, operators=nothing; μ=nothing)
  #=MODIFY
  =#

  if operators === nothing
    operators = ["A", "F"]
  end

  if "A" in operators

    if ROM_info.problem_nonlinearities["A"] === false

      @info "Assembling affine reduced stiffness"
      projection_time = @elapsed begin
        A = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "A.csv"); convert_to_sparse = true)
        RB_variables.Aₙ = (RB_variables.Φₛᵘ)' * A * RB_variables.Φₛᵘ
        if ROM_info.save_offline_structures
          save_variable(RB_variables.Aₙ, "Aₙ", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Aₙ"))
        end
      end

    else

      @info "The stiffness is non-affine: running the MDEIM offline phase"
      projection_time = @elapsed begin
        Aₙ_i = sparse([], [], [])
        for i_nₛ = 1:maximum(10, ROM_info.nₛ)
          parametric_info = get_parametric_specifics(ROM_info, μ[i_nₛ])
          A_i = assemble_stiffness(FE_space, ROM_info, parametric_info)
          Aₙ_i = hcat(Aₙ_i, (RB_variables.Φₛᵘ)' * A_i * RB_variables.Φₛᵘ)
        end
        Aₙ_i = reshape(Aₙ_i, :, 1)
        if ROM_info.save_offline_structures
          (RB_variables.Aₙ_affine, RB_variables.Aₙ_idx) = DEIM_offline(Aₙ_i, ROM_info.ϵₛ, true, ROM_info.paths.ROM_structures_path, "Aₙ_mdeim")
        else
          (RB_variables.Aₙ_affine, RB_variables.Aₙ_idx) = DEIM_offline(Aₙ_i, ROM_info.ϵₛ)
        end
      end

    end

  end

  if "F" in operators

    if ROM_info.problem_nonlinearities["f"] === false || ROM_info.problem_nonlinearities["h"] === false

      @info "Assembling affine reduced forcing term"
      projection_time += @elapsed begin
        F = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "F.csv"))
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
          parametric_info = get_parametric_specifics(ROM_info, μ[i_nₛ])
          F_i = assemble_forcing(FE_space, parametric_info)
          Fₙ_i = hcat(Fₙ_i, (RB_variables.Φₛᵘ)' * F_i)
        end
        if ROM_info.save_offline_structures
          (RB_variables.Fₙ_affine, RB_variables.Fₙ_idx) = DEIM_offline(Fₙ_i, ROM_info.ϵₛ, true, ROM_info.paths.ROM_structures_path, "Fₙ_deim")
        else
          (RB_variables.Fₙ_affine, RB_variables.Fₙ_idx) = DEIM_offline(Fₙ_i, ROM_info.ϵₛ)
        end
      end
    end

  end

  RB_variables.offline_time += projection_time

end
