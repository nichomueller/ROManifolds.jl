include("../FEM/LagrangianQuad.jl")

function build_M_snapshots(FEMSpace::SteadyProblem, RBInfo::Info, μ::Matrix)

  for i_nₛ = 1:RBInfo.nₛ_MDEIM
    @info "Snapshot number $i_nₛ, mass"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    Param = get_ParamInfo(problem_ntuple, RBInfo, μ_i)
    M_i = assemble_mass(FEMSpace, RBInfo, Param)
    i, v = findnz(M_i[:])
    if i_nₛ == 1
      global M = sparse(i, ones(length(i)), v, FEMSpace.Nₛᵘ^2, RBInfo.nₛ_MDEIM)
    else
      global M[:, i_nₛ] = sparse(i, ones(length(i)), v)
    end
  end

  M

end

function build_M_snapshots(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  μ::Vector,
  timesθ::Vector)

  Nₜ = length(timesθ)
  Param = get_ParamInfo(problem_ntuple, RBInfo, μ)
  M_t = assemble_mass(FEMSpace, RBInfo, Param)

  for i_t = 1:Nₜ
    @info "Snapshot at time step $i_t, mass"
    M_i = M_t(timesθ[i_t])
    i, v = findnz(M_i[:])
    if i_t == 1
      global row_idx = i
      global M = zeros(length(row_idx),Nₜ)
    end
    global M[:,i_t] = v
  end

  M, row_idx

end

function build_A_snapshots(FEMSpace::SteadyProblem, RBInfo::Info, μ::Matrix)

  for i_nₛ = 1:RBInfo.nₛ_MDEIM
    @info "Snapshot number $i_nₛ, stiffness"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    Param = get_ParamInfo(problem_ntuple, RBInfo, μ_i)
    A_i = assemble_stiffness(FEMSpace, RBInfo, Param)
    i, v = findnz(A_i[:])
    if i_nₛ == 1
      global A = sparse(i, ones(length(i)), v, FEMSpace.Nₛᵘ^2, RBInfo.nₛ_MDEIM)
    else
      global A[:, i_nₛ] = sparse(i, ones(length(i)), v)
    end
  end

  A

end

function build_A_snapshots(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  μ::Vector,
  timesθ::Vector)

  Nₜ = length(timesθ)
  Param = get_ParamInfo(problem_ntuple, RBInfo, μ)
  A_t = assemble_stiffness(FEMSpace, RBInfo, Param)

  for i_t = 1:Nₜ
    @info "Snapshot at time step $i_t, stiffness"
    A_i = A_t(timesθ[i_t])
    i, v = findnz(A_i[:])
    if i_t == 1
      global row_idx = i
      global A = zeros(length(row_idx),Nₜ)
    end
    global A[:,i_t] = v
  end

  A, row_idx

end

function get_snaps_MDEIM(FEMSpace::SteadyProblem, RBInfo::Info, μ::Matrix, var="A")
  if var == "A"
    snaps = build_A_snapshots(FEMSpace, RBInfo, μ)
  elseif var == "M"
    snaps = build_M_snapshots(FEMSpace, RBInfo, μ)
  else
    @error "Run MDEIM on A or M only"
  end
  snaps
end

function get_snaps_MDEIM(FEMSpace::UnsteadyProblem, RBInfo::Info, μ::Matrix, var="A")

  timesθ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+RBInfo.δt*RBInfo.θ

  if RBInfo.space_time_M_DEIM
    for k = 1:RBInfo.nₛ_MDEIM
      @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
      μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
      if var == "A"
        snapsₖ,row_idx = build_A_snapshots(FEMSpace,RBInfo,μₖ,timesθ)
      elseif var == "M"
        snapsₖ,row_idx = build_M_snapshots(FEMSpace,RBInfo,μₖ,timesθ)
      else
        @error "Run MDEIM on A or M only"
      end
      if k == 1
        global row_idx = row_idx
        global snaps = zeros(length(row_idx),RBInfo.nₛ_MDEIM)
      else
        global snaps[:,k] = snapsₖ[:]
      end
    end

  elseif RBInfo.functional_M_DEIM

    ξₖ = get_cell_map(FEMSpace.Ω)
    Qₕ_cell_point = get_cell_points(FEMSpace.Qₕ)
    qₖ = get_data(Qₕ_cell_point)
    phys_quadp = lazy_map(evaluate,ξₖ,qₖ)
    ncells = length(phys_quadp)
    nquad_cell = length(phys_quadp[1])
    nquad = nquad_cell*ncells
    refFE_quad = ReferenceFE(lagrangianQuad, Float64, FEMInfo.order)
    V₀_quad = TestFESpace(model, refFE_quad, conformity=:L2)
    for k = 1:RBInfo.nₛ_MDEIM
      @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
      μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
      Param = get_ParamInfo(problem_ntuple, RBInfo, μₖ)
      if var == "A"
        snapsₖ = [Param.α(phys_quadp[n][q],t_θ)
        for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
      elseif var == "M"
        snapsₖ = [Param.m(phys_quadp[n][q],t_θ)
        for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
      else
        @error "Run MDEIM on A or M only"
      end
      compressed_snapsₖ, _ = POD(reshape(snapsₖ,nquad,Nₜ), RBInfo.ϵₛ)
      if k == 1
        global snaps = compressed_snapsₖ
      else
        global snaps = hcat(snaps, compressed_snapsₖ)
      end
    end
    Θmat, Σ = POD(compressed_snaps, RBInfo.ϵₛ)
    Q = size(Θmat)[2]
    for q = 1:Q
      Θq = FEFunction(V₀_quad,Θmat[:,q])
      if var == "A"
        Matq = (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Θq*∇(FEMSpace.ϕᵤ(0.0))))*FEMSpace.dΩ,
         FEMSpace.V(0.0), FEMSpace.V₀))
      elseif var == "M"
        Matq = (assemble_matrix(∫(FEMSpace.ϕᵥ*(Θq*FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
        FEMSpace.V(0.0), FEMSpace.V₀))
      end
      i,v = findnz(Matq[:])
      if q == 1
        global row_idx = i
        global Mat_affine = zeros(length(row_idx),Q)
      end
      global Mat_affine[:,q] = v
    end

  else

    for k = 1:RBInfo.nₛ_MDEIM
      @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
      μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
      if var == "A"
        snapsₖ,row_idx = build_A_snapshots(FEMSpace, RBInfo, μₖ)
      elseif var == "M"
        snapsₖ,row_idx = build_M_snapshots(FEMSpace, RBInfo, μₖ)
      else
        @error "Run MDEIM on A or M only"
      end
      compressed_snapsₖ, _ = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
      if k == 1
        global row_idx = row_idx
        global snaps = compressed_snapsₖ
      else
        global snaps = hcat(snaps, compressed_snapsₖ)
      end
    end
  end

  return snaps,row_idx

end

function build_F_snapshots(FEMSpace::SteadyProblem, RBInfo::Info, μ::Vector)

  F = zeros(FEMSpace.Nₛᵘ, RBInfo.nₛ_DEIM)

  for i_nₛ = 1:RBInfo.nₛ_DEIM
    @info "Snapshot number $i_nₛ, forcing"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    Param = get_ParamInfo(problem_ntuple, RBInfo, μ_i)
    F[:, i_nₛ] = assemble_forcing(FEMSpace, RBInfo, Param)[:]
  end

  F

end

function build_F_snapshots(FEMSpace::UnsteadyProblem, RBInfo::Info, μ::Vector)

  Nₜ = convert(Int64, RBInfo.T/RBInfo.δt)
  δtθ = RBInfo.δt*RBInfo.θ
  timesθ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+δtθ

  Param = get_ParamInfo(problem_ntuple, RBInfo, μ)
  F_t = assemble_forcing(FEMSpace, RBInfo, Param)
  F = zeros(FEMSpace.Nₛᵘ, Nₜ)

  for i_t = 1:Nₜ
    @info "Snapshot at time step $i_t, forcing"
    F[:, i_t] = F_t(timesθ[i_t])[:]
  end

  F

end

function build_H_snapshots(FEMSpace::SteadyProblem, RBInfo::Info, μ::Vector)

  H = zeros(FEMSpace.Nₛᵘ, RBInfo.nₛ_DEIM)

  for i_nₛ = 1:RBInfo.nₛ_DEIM
    @info "Snapshot number $i_nₛ, neumann datum"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    Param = get_ParamInfo(problem_ntuple, RBInfo, μ_i)
    H[:, i_nₛ] = assemble_neumann_datum(FEMSpace, RBInfo, Param)[:]
  end

  H

end

function build_H_snapshots(FEMSpace::UnsteadyProblem, RBInfo::Info, μ::Vector)

  Nₜ = convert(Int64, RBInfo.T/RBInfo.δt)
  δtθ = RBInfo.δt*RBInfo.θ
  timesθ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+δtθ

  Param = get_ParamInfo(problem_ntuple, RBInfo, μ)
  H_t = assemble_neumann_datum(FEMSpace, RBInfo, Param)
  H = zeros(FEMSpace.Nₛᵘ, Nₜ)

  for i_t = 1:Nₜ
    @info "Snapshot at time step $i_t, neumann datum"
    H[:, i_t] = H_t(timesθ[i_t])[:]
  end

  H

end

function get_snaps_DEIM(FEMSpace::SteadyProblem, RBInfo::Info, μ::Matrix, var="F")
  if var == "F"
    snaps = build_F_snapshots(FEMSpace, RBInfo, μ)
  elseif var == "H"
    snaps = build_H_snapshots(FEMSpace, RBInfo, μ)
  else
    @error "Run DEIM on F or H only"
  end
  snaps
end

function get_snaps_DEIM(FEMSpace::UnsteadyProblem, RBInfo::Info, μ::Matrix, var="F")
  if RBInfo.space_time_M_DEIM
    for k = 1:RBInfo.nₛ_MDEIM
      @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
      μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
      if var == "A"
        snapsₖ = build_F_snapshots(FEMSpace, RBInfo, μₖ)
      elseif var == "M"
        snapsₖ = build_H_snapshots(FEMSpace, RBInfo, μₖ)
      else
        @error "Run MDEIM on A or M only"
      end
      if k == 1
        global snaps = snapsₖ[:]
      else
        global snaps = hcat(snaps, snapsₖ[:])
      end
    end
  else
    for k = 1:RBInfo.nₛ_MDEIM
      @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
      μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
      if var == "A"
        snapsₖ = build_A_snapshots(FEMSpace, RBInfo, μₖ)
      elseif var == "M"
        snapsₖ = build_M_snapshots(FEMSpace, RBInfo, μₖ)
      else
        @error "Run MDEIM on A or M only"
      end
      compressed_snapsₖ, _ = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
      if k == 1
        global snaps = compressed_snapsₖ
      else
        global snaps = hcat(snaps, compressed_snapsₖ)
      end
    end
  end
  return snaps
end
