include("RB_Poisson_steady.jl")

function assemble_reduced_affine_components(RBInfo, RBVars::PoissonSTGRB, operators=nothing; μ=nothing)


  if operators === nothing
    operators = ["A", "F"]
  end

  if "A" in operators

    if RBInfo.problem_nonlinearities["A"] === false

      @info "Assembling affine reduced stiffness"
      projection_time = @elapsed begin
        A = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "A.csv"); convert_to_sparse = true)
        RBVars.Aₙ = (RBVars.Φₛᵘ)' * A * RBVars.Φₛᵘ
        if RBInfo.save_offline_structures
          save_variable(RBVars.Aₙ, "Aₙ", "csv", joinpath(RBInfo.paths.ROM_structures_path, "Aₙ"))
        end
      end

    else

      @info "The stiffness is non-affine: running the MDEIM offline phase"
      projection_time = @elapsed begin
        Aₙ_i = sparse([], [], [])
        for i_nₛ = 1:maximum(10, RBInfo.nₛ)
          Param = get_ParamInfo(problem_ntuple, RBInfo, μ[i_nₛ])
          A_i = assemble_stiffness(FESpace, RBInfo, Param)
          Aₙ_i = hcat(Aₙ_i, (RBVars.Φₛᵘ)' * A_i * RBVars.Φₛᵘ)
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

    if RBInfo.problem_nonlinearities["f"] === false || RBInfo.problem_nonlinearities["h"] === false

      @info "Assembling affine reduced forcing term"
      projection_time += @elapsed begin
        F = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "F.csv"))
        RBVars.Fₙ = (RBVars.Φₛᵘ)' * F
        if RBInfo.save_offline_structures
          save_variable(RBVars.Fₙ, "Fₙ", "csv", joinpath(RBInfo.paths.ROM_structures_path, "Fₙ"))
        end
      end

    else

      @info "The forcing term is non-affine: running the DEIM offline phase"
      projection_time += @elapsed begin
        Fₙ_i = Float64[]
        for i_nₛ = 1:maximum(10, RBInfo.nₛ)
          Param = get_ParamInfo(problem_ntuple, RBInfo, μ[i_nₛ])
          F_i = assemble_forcing(FESpace, Param)
          Fₙ_i = hcat(Fₙ_i, (RBVars.Φₛᵘ)' * F_i)
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
