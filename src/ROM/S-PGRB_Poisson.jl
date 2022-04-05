function get_inverse_P_matrix(ROM_info, RB_variables::PoissonSTPGRB)
  #=MODIFY
  =#

  if isempty(RB_variables.Pᵘ_inv) || maximum(abs.(RB_variables.Pᵘ_inv)) === 0

    if isfile(joinpath(ROM_info.paths.FEM_structures_path, "Pᵘ_inv.csv"))
      RB_variables.Pᵘ_inv = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_structures_path, "Pᵘ_inv.csv"), DataFrame))
    else
      get_norm_matrix(ROM_info, RB_variables)
      Pᵘ = diag(RB_variables.Xᵘ)
      RB_variables.Pᵘ_inv = I(size(RB_variables.Xᵘ)[1]) \ Pᵘ
      save_variable(RB_variables.Pᵘ_inv, "Pᵘ_inv", "csv", joinpath(ROM_info.paths.FEM_structures_path, "Pᵘ_inv"))
    end

  end

end

function assemble_reduced_affine_components(ROM_info, RB_variables::PoissonSTPGRB, operators=nothing; μ=nothing)
  #=MODIFY
  =#

  if isempty(RB_variables.Φₛᵘ) || maximum(abs.(RB_variables.Φₛᵘ)) === 0
    @error "Error: must generate or import spatial RBs before computing the reduced affine components"
  end

  if isnothing(operators)
    operators = ["A", "F"]
  end

  if "A" in operators

    if !ROM_info.problem_nonlinearities["A"]

      @info "Assembling affine reduced stiffness"
      projection_time = @elapsed begin
        A = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "A.csv"))
        AΦₛᵘ = A * RB_variables.Φₛᵘ
        RB_variables.Aₙ = (AΦₛᵘ)' * RB_variables.Pᵘ_inv * AΦₛᵘ
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
          A_iΦₛᵘ = A_i * RB_variables.Φₛᵘ
          Aₙ_i = hcat(Aₙ_i, (A_iΦₛᵘ)' * RB_variables.Pᵘ_inv * A_iΦₛᵘ)
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

    if !ROM_info.problem_nonlinearities["f"] && !ROM_info.problem_nonlinearities["h"]

      @info "Assembling affine reduced forcing term"
      projection_time += @elapsed begin
        F = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "F.csv"))

        if !ROM_info.problem_nonlinearities["A"]

          A = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "A.csv"); convert_to_sparse = true)
          AΦₛᵘ = A * RB_variables.Φₛᵘ
          RB_variables.Fₙ = (AΦₛᵘ)' * RB_variables.Pᵘ_inv * F
          if ROM_info.save_offline_structures
            save_variable(RB_variables.Fₙ, "Fₙ", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Fₙ"))
          end

        else

          Fₙ_i = Float64[]
          for i_nₛ = 1:maximum(10, ROM_info.nₛ)
            parametric_info = get_parametric_specifics(ROM_info, μ[i_nₛ])
            A_i = assemble_stiffness(FE_space, ROM_info, parametric_info)
            A_iΦₛᵘ = A_i * RB_variables.Φₛᵘ
            Fₙ_i = hcat(Fₙ_i, (A_iΦₛᵘ)' * RB_variables.Pᵘ_inv * F)
          end
          if ROM_info.save_offline_structures
            (RB_variables.Fₙ_affine, RB_variables.Fₙ_idx) = DEIM_offline(Fₙ_i, ROM_info.ϵˢ, true, ROM_info.paths.ROM_structures_path, "Fₙ_deim")
          else
            (RB_variables.Fₙ_affine, RB_variables.Fₙ_idx) = DEIM_offline(Fₙ_i, ROM_info.ϵˢ)
          end

        end

      end

    else

      @info "The forcing term is non-affine: running the DEIM offline phase"
      projection_time += @elapsed begin
        Fₙ_i = Float64[]

        if !ROM_info.problem_nonlinearities["A"]

          A = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "A.csv"); convert_to_sparse = true)
          AΦₛᵘPᵘ_inv = (A * Φₛᵘ)' * RB_variables.Pᵘ_inv
          for i_nₛ = 1:maximum(10, ROM_info.nₛ)
            parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
            F_i = assemble_forcing(FE_space, parametric_info)
            Fₙ_i = hcat(Fₙ_i, AΦₛᵘPᵘ_inv * F_i)
          end

        else

          for i_nₛ = 1:maximum(10, ROM_info.nₛ)
            parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
            A_i = assemble_stiffness(FE_space, ROM_info, parametric_info)
            F_i = assemble_forcing(FE_space, parametric_info)
            A_iΦₛᵘ = A_i * RB_variables.Φₛᵘ
            Fₙ_i = hcat(Fₙ_i, (A_iΦₛᵘ)' * RB_variables.Pᵘ_inv * F_i)
          end

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
