function get_inverse_P_matrix(RBInfo, RBVars::PoissonSTPGRB)


  if isempty(RBVars.Pᵘ_inv) || maximum(abs.(RBVars.Pᵘ_inv)) === 0

    if isfile(joinpath(RBInfo.paths.FEM_structures_path, "Pᵘ_inv.csv"))
      RBVars.Pᵘ_inv = Matrix(CSV.read(joinpath(RBInfo.paths.FEM_structures_path, "Pᵘ_inv.csv"), DataFrame))
    else
      get_norm_matrix(RBInfo, RBVars)
      Pᵘ = diag(RBVars.Xᵘ)
      RBVars.Pᵘ_inv = I(size(RBVars.Xᵘ)[1]) \ Pᵘ
      save_variable(RBVars.Pᵘ_inv, "Pᵘ_inv", "csv", joinpath(RBInfo.paths.FEM_structures_path, "Pᵘ_inv"))
    end

  end

end

function assemble_reduced_affine_components(RBInfo, RBVars::PoissonSTPGRB, operators=nothing; μ=nothing)


  if isempty(RBVars.Φₛᵘ) || maximum(abs.(RBVars.Φₛᵘ)) === 0
    error("Error: must generate or import spatial RBs before computing the reduced affine components")
  end

  if isnothing(operators)
    operators = ["A", "F"]
  end

  if "A" in operators

    if !RBInfo.problem_nonlinearities["A"]

      println("Assembling affine reduced stiffness")
      projection_time = @elapsed begin
        A = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.paths.FEM_structures_path, "A.csv"))
        AΦₛᵘ = A * RBVars.Φₛᵘ
        RBVars.Aₙ = (AΦₛᵘ)' * RBVars.Pᵘ_inv * AΦₛᵘ
        if RBInfo.save_offline_structures
          save_variable(RBVars.Aₙ, "Aₙ", "csv", joinpath(RBInfo.paths.ROM_structures_path, "Aₙ"))
        end
      end

    else

      println("The stiffness is non-affine: running the MDEIM offline phase")
      projection_time = @elapsed begin
        Aₙ_i = sparse([], [], [])
        for i_nₛ = 1:maximum(10, RBInfo.nₛ)
          Param = get_ParamInfo(RBInfo.FEMInfo, RBInfo.FEMInfo.problem_id, μ[i_nₛ])
          A_i = assemble_stiffness(FEMSpace, RBInfo, Param)
          A_iΦₛᵘ = A_i * RBVars.Φₛᵘ
          Aₙ_i = hcat(Aₙ_i, (A_iΦₛᵘ)' * RBVars.Pᵘ_inv * A_iΦₛᵘ)
        end
        Aₙ_i = reshape(Aₙ_i, :, 1)
        if RBInfo.save_offline_structures
          (RBVars.Aₙ_affine, RBVars.Aₙ_idx) = DEIM_offline(Aₙ_i, RBInfo.ϵₛ, true, RBInfo.paths.ROM_structures_path, "Aₙ_mdeim")
        else
          (RBVars.Aₙ_affine, RBVars.Aₙ_idx) = DEIM_offline(Aₙ_i, RBInfo.ϵₛ)
        end
      end

    end

  end

  if "F" in operators

    if !RBInfo.problem_nonlinearities["f"] && !RBInfo.problem_nonlinearities["h"]

      println("Assembling affine reduced forcing term")
      projection_time += @elapsed begin
        F = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.paths.FEM_structures_path, "F.csv"))

        if !RBInfo.problem_nonlinearities["A"]

          A = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.paths.FEM_structures_path, "A.csv"); convert_to_sparse = true)
          AΦₛᵘ = A * RBVars.Φₛᵘ
          RBVars.Fₙ = (AΦₛᵘ)' * RBVars.Pᵘ_inv * F
          if RBInfo.save_offline_structures
            save_variable(RBVars.Fₙ, "Fₙ", "csv", joinpath(RBInfo.paths.ROM_structures_path, "Fₙ"))
          end

        else

          Fₙ_i = Float64[]
          for i_nₛ = 1:maximum(10, RBInfo.nₛ)
            Param = get_ParamInfo(RBInfo.FEMInfo, RBInfo.FEMInfo.problem_id, μ[i_nₛ])
            A_i = assemble_stiffness(FEMSpace, RBInfo, Param)
            A_iΦₛᵘ = A_i * RBVars.Φₛᵘ
            Fₙ_i = hcat(Fₙ_i, (A_iΦₛᵘ)' * RBVars.Pᵘ_inv * F)
          end
          if RBInfo.save_offline_structures
            (RBVars.Fₙ_affine, RBVars.Fₙ_idx) = DEIM_offline(Fₙ_i, RBInfo.ϵₛ, true, RBInfo.paths.ROM_structures_path, "Fₙ_deim")
          else
            (RBVars.Fₙ_affine, RBVars.Fₙ_idx) = DEIM_offline(Fₙ_i, RBInfo.ϵₛ)
          end

        end

      end

    else

      println("The forcing term is non-affine: running the DEIM offline phase")
      projection_time += @elapsed begin
        Fₙ_i = Float64[]

        if !RBInfo.problem_nonlinearities["A"]

          A = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.paths.FEM_structures_path, "A.csv"); convert_to_sparse = true)
          AΦₛᵘPᵘ_inv = (A * Φₛᵘ)' * RBVars.Pᵘ_inv
          for i_nₛ = 1:maximum(10, RBInfo.nₛ)
            Param = compute_Param(problem_nonlinearities, Params, i_nₛ)
            F_i = assemble_forcing(FEMSpace, Param)
            Fₙ_i = hcat(Fₙ_i, AΦₛᵘPᵘ_inv * F_i)
          end

        else

          for i_nₛ = 1:maximum(10, RBInfo.nₛ)
            Param = compute_Param(problem_nonlinearities, Params, i_nₛ)
            A_i = assemble_stiffness(FEMSpace, RBInfo, Param)
            F_i = assemble_forcing(FEMSpace, Param)
            A_iΦₛᵘ = A_i * RBVars.Φₛᵘ
            Fₙ_i = hcat(Fₙ_i, (A_iΦₛᵘ)' * RBVars.Pᵘ_inv * F_i)
          end

        end

        if RBInfo.save_offline_structures
          (RBVars.Fₙ_affine, RBVars.Fₙ_idx) = DEIM_offline(Fₙ_i, RBInfo.ϵₛ, true, RBInfo.paths.ROM_structures_path, "Fₙ_deim")
        else
          (RBVars.Fₙ_affine, RBVars.Fₙ_idx) = DEIM_offline(Fₙ_i, RBInfo.ϵₛ)
        end

      end

    end

  end

  RBVars.offline_time += projection_time

end
