function build_M_snapshots(FESpace::SteadyProblem, RBInfo::Info, μ::Matrix)

  for i_nₛ = 1:RBInfo.nₛ_MDEIM
    @info "Snapshot number $i_nₛ, mass"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    Param = get_Parametric_specifics(problem_ntuple, RBInfo, μ_i)
    M_i = assemble_mass(FESpace, RBInfo, Param)
    i, v = findnz(M_i[:])
    if i_nₛ == 1
      global M = sparse(i, ones(length(i)), v, FESpace.Nₛᵘ^2, RBInfo.nₛ_MDEIM)
    else
      global M[:, i_nₛ] = sparse(i, ones(length(i)), v)
    end
  end

  M

end

function build_M_snapshots(FESpace::UnsteadyProblem, RBInfo::Info, μ::Vector)

  Nₜ = convert(Int64, RBInfo.T/RBInfo.δt)
  times_θ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+RBInfo.δt*RBInfo.θ
  Param = get_Parametric_specifics(problem_ntuple, RBInfo, μ)
  M_t = assemble_mass(FESpace, RBInfo, Param)

  for i_t = 1:Nₜ
    @info "Snapshot at time step $i_t, mass"
    M_i = M_t(times_θ[i_t])
    i, v = findnz(M_i[:])
    if i_t == 1
      global row_idx = i
      global M = zeros(length(row_idx),Nₜ)
    end
    global M[:,i_t] = v
  end

  M, row_idx

end

function build_A_snapshots(FESpace::SteadyProblem, RBInfo::Info, μ::Matrix)

  for i_nₛ = 1:RBInfo.nₛ_MDEIM
    @info "Snapshot number $i_nₛ, stiffness"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    Param = get_Parametric_specifics(problem_ntuple, RBInfo, μ_i)
    A_i = assemble_stiffness(FESpace, RBInfo, Param)
    i, v = findnz(A_i[:])
    if i_nₛ == 1
      global A = sparse(i, ones(length(i)), v, FESpace.Nₛᵘ^2, RBInfo.nₛ_MDEIM)
    else
      global A[:, i_nₛ] = sparse(i, ones(length(i)), v)
    end
  end

  A

end

function build_A_snapshots(FESpace::UnsteadyProblem, RBInfo::Info, μ::Vector)

  Nₜ = convert(Int64, RBInfo.T/RBInfo.δt)
  times_θ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+RBInfo.δt*RBInfo.θ
  Param = get_Parametric_specifics(problem_ntuple, RBInfo, μ)
  A_t = assemble_stiffness(FESpace, RBInfo, Param)

  for i_t = 1:Nₜ
    @info "Snapshot at time step $i_t, stiffness"
    A_i = A_t(times_θ[i_t])
    i, v = findnz(A_i[:])
    if i_t == 1
      global row_idx = i
      global A = zeros(length(row_idx),Nₜ)
    end
    global A[:,i_t] = v
  end

  A, row_idx

end

function get_snaps_MDEIM(FESpace::SteadyProblem, RBInfo::Info, μ::Matrix, var="A")
  if var == "A"
    snaps = build_A_snapshots(FESpace, RBInfo, μ)
  elseif var == "M"
    snaps = build_M_snapshots(FESpace, RBInfo, μ)
  else
    @error "Run MDEIM on A or M only"
  end
  snaps
end

function get_snaps_MDEIM(FESpace::UnsteadyProblem, RBInfo::Info, μ::Matrix, var="A")
  if RBInfo.space_time_M_DEIM
    for k = 1:RBInfo.nₛ_MDEIM
      @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
      μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
      if var == "A"
        snapsₖ,row_idx = build_A_snapshots(FESpace, RBInfo, μₖ)
      elseif var == "M"
        snapsₖ,row_idx = build_M_snapshots(FESpace, RBInfo, μₖ)
      else
        @error "Run MDEIM on A or M only"
      end
      if k == 1
        global row_idx = row_idx
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
        snapsₖ,row_idx = build_A_snapshots(FESpace, RBInfo, μₖ)
      elseif var == "M"
        snapsₖ,row_idx = build_M_snapshots(FESpace, RBInfo, μₖ)
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

function build_F_snapshots(FESpace::SteadyProblem, RBInfo::Info, μ::Vector)

  F = zeros(FESpace.Nₛᵘ, RBInfo.nₛ_DEIM)

  for i_nₛ = 1:RBInfo.nₛ_DEIM
    @info "Snapshot number $i_nₛ, forcing"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    Param = get_Parametric_specifics(problem_ntuple, RBInfo, μ_i)
    F[:, i_nₛ] = assemble_forcing(FESpace, RBInfo, Param)[:]
  end

  F

end

function build_F_snapshots(FESpace::UnsteadyProblem, RBInfo::Info, μ::Vector)

  Nₜ = convert(Int64, RBInfo.T/RBInfo.δt)
  δtθ = RBInfo.δt*RBInfo.θ
  times_θ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+δtθ

  Param = get_Parametric_specifics(problem_ntuple, RBInfo, μ)
  F_t = assemble_forcing(FESpace, RBInfo, Param)
  F = zeros(FESpace.Nₛᵘ, Nₜ)

  for i_t = 1:Nₜ
    @info "Snapshot at time step $i_t, forcing"
    F[:, i_t] = F_t(times_θ[i_t])[:]
  end

  F

end

function build_H_snapshots(FESpace::SteadyProblem, RBInfo::Info, μ::Vector)

  H = zeros(FESpace.Nₛᵘ, RBInfo.nₛ_DEIM)

  for i_nₛ = 1:RBInfo.nₛ_DEIM
    @info "Snapshot number $i_nₛ, neumann datum"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    Param = get_Parametric_specifics(problem_ntuple, RBInfo, μ_i)
    H[:, i_nₛ] = assemble_neumann_datum(FESpace, RBInfo, Param)[:]
  end

  H

end

function build_H_snapshots(FESpace::UnsteadyProblem, RBInfo::Info, μ::Vector)

  Nₜ = convert(Int64, RBInfo.T/RBInfo.δt)
  δtθ = RBInfo.δt*RBInfo.θ
  times_θ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+δtθ

  Param = get_Parametric_specifics(problem_ntuple, RBInfo, μ)
  H_t = assemble_neumann_datum(FESpace, RBInfo, Param)
  H = zeros(FESpace.Nₛᵘ, Nₜ)

  for i_t = 1:Nₜ
    @info "Snapshot at time step $i_t, neumann datum"
    H[:, i_t] = H_t(times_θ[i_t])[:]
  end

  H

end

function get_snaps_DEIM(FESpace::SteadyProblem, RBInfo::Info, μ::Matrix, var="F")
  if var == "F"
    snaps = build_F_snapshots(FESpace, RBInfo, μ)
  elseif var == "H"
    snaps = build_H_snapshots(FESpace, RBInfo, μ)
  else
    @error "Run DEIM on F or H only"
  end
  snaps
end

function get_snaps_DEIM(FESpace::UnsteadyProblem, RBInfo::Info, μ::Matrix, var="F")
  if RBInfo.space_time_M_DEIM
    for k = 1:RBInfo.nₛ_MDEIM
      @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
      μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
      if var == "A"
        snapsₖ = build_F_snapshots(FESpace, RBInfo, μₖ)
      elseif var == "M"
        snapsₖ = build_H_snapshots(FESpace, RBInfo, μₖ)
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
        snapsₖ = build_A_snapshots(FESpace, RBInfo, μₖ)
      elseif var == "M"
        snapsₖ = build_M_snapshots(FESpace, RBInfo, μₖ)
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
