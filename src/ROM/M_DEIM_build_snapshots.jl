function build_M_snapshots(FE_space::SteadyProblem, ROM_info::Info, μ::Array)

  for i_nₛ = 1:ROM_info.nₛ_MDEIM
    @info "Snapshot number $i_nₛ, mass"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(problem_ntuple, ROM_info, μ_i)
    M_i = assemble_mass(FE_space, ROM_info, parametric_info)
    i, v = findnz(M_i[:])
    if i_nₛ == 1
      global M = sparse(i, ones(length(i)), v, FE_space.Nₛᵘ^2, ROM_info.nₛ_MDEIM)
    else
      global M[:, i_nₛ] = sparse(i, ones(length(i)), v)
    end
  end

  M

end

function build_M_snapshots(FE_space::UnsteadyProblem, ROM_info::Info, μ::Array)

  Nₜ = convert(Int64, ROM_info.T/ROM_info.δt)
  δtθ = ROM_info.δt*ROM_info.θ
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ

  parametric_info = get_parametric_specifics(problem_ntuple, ROM_info, μ)
  M_t = assemble_mass(FE_space, ROM_info, parametric_info)

  for i_t = 1:Nₜ
    @info "Snapshot at time step $i_t, mass"
    M_i = M_t(times_θ[i_t])
    i, v = findnz(M_i[:])
    if i_t == 1
      global M = sparse(i, ones(length(i)), v, FE_space.Nₛᵘ^2, Nₜ)
    else
      global M[:, i_t] = sparse(i, ones(length(i)), v)
    end
  end

  M

end

function build_M_snapshots(FE_space::UnsteadyProblem, ROM_info::Info, μ::Array, t::Float64)

  for nₛ= 1:ROM_info.nₛ_MDEIM
    @info "Snapshot $nₛ at time instant $t, mass"
    μ_nₛ = parse.(Float64, split(chop(μ[nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(problem_ntuple, ROM_info, μ_nₛ)
    M_nₛ = assemble_mass(FE_space, ROM_info, parametric_info)(t)
    i, v = findnz(M_nₛ[:])
    if nₛ == 1
      global M = sparse(i, ones(length(i)), v, FE_space.Nₛᵘ^2, ROM_info.nₛ_MDEIM)
    else
      global M[:, nₛ] = sparse(i, ones(length(i)), v)
    end
  end

  M

end

function build_A_snapshots(FE_space::SteadyProblem, ROM_info::Info, μ::Array)

  for i_nₛ = 1:ROM_info.nₛ_MDEIM
    @info "Snapshot number $i_nₛ, stiffness"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(problem_ntuple, ROM_info, μ_i)
    A_i = assemble_stiffness(FE_space, ROM_info, parametric_info)
    i, v = findnz(A_i[:])
    if i_nₛ == 1
      global A = sparse(i, ones(length(i)), v, FE_space.Nₛᵘ^2, ROM_info.nₛ_MDEIM)
    else
      global A[:, i_nₛ] = sparse(i, ones(length(i)), v)
    end
  end

  A

end

function build_A_snapshots(FE_space::UnsteadyProblem, ROM_info::Info, μ::Array)

  Nₜ = convert(Int64, ROM_info.T/ROM_info.δt)
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+ROM_info.δt*ROM_info.θ
  parametric_info = get_parametric_specifics(problem_ntuple, ROM_info, μ)
  A_t = assemble_stiffness(FE_space, ROM_info, parametric_info)

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

function get_snaps_MDEIM(FE_space::UnsteadyProblem, ROM_info::Info, μ::Matrix, var="A")
  if ROM_info.space_time_M_DEIM
    for k = 1:ROM_info.nₛ_MDEIM
      @info "Considering parameter number $k/$(ROM_info.nₛ_MDEIM)"
      μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
      if var == "A"
        snapsₖ,row_idx = build_A_snapshots(FE_space, ROM_info, μₖ)
      elseif var == "M"
        snapsₖ,row_idx = build_M_snapshots(FE_space, ROM_info, μₖ)
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
    for k = 1:ROM_info.nₛ_MDEIM
      @info "Considering parameter number $k/$(ROM_info.nₛ_MDEIM)"
      μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
      if var == "A"
        snapsₖ,row_idx = build_A_snapshots(FE_space, ROM_info, μₖ)
      elseif var == "M"
        snapsₖ,row_idx = build_M_snapshots(FE_space, ROM_info, μₖ)
      else
        @error "Run MDEIM on A or M only"
      end
      compressed_snapsₖ, _ = M_DEIM_POD(snapsₖ, ROM_info.ϵₛ)
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

function build_F_snapshots(FE_space::SteadyProblem, ROM_info::Info, μ::Array)

  F = zeros(FE_space.Nₛᵘ, ROM_info.nₛ_DEIM)

  for i_nₛ = 1:ROM_info.nₛ_DEIM
    @info "Snapshot number $i_nₛ, forcing"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(problem_ntuple, ROM_info, μ_i)
    F[:, i_nₛ] = assemble_forcing(FE_space, ROM_info, parametric_info)[:]
  end

  F

end

function build_F_snapshots(FE_space::UnsteadyProblem, ROM_info::Info, μ::Array)

  Nₜ = convert(Int64, ROM_info.T/ROM_info.δt)
  δtθ = ROM_info.δt*ROM_info.θ
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ

  parametric_info = get_parametric_specifics(problem_ntuple, ROM_info, μ)
  F_t = assemble_forcing(FE_space, ROM_info, parametric_info)
  F = zeros(FE_space.Nₛᵘ, Nₜ)

  for i_t = 1:Nₜ
    @info "Snapshot at time step $i_t, forcing"
    F[:, i_t] = F_t(times_θ[i_t])[:]
  end

  F

end

function build_F_snapshots(FE_space::UnsteadyProblem, ROM_info::Info, μ::Array, t::Float64)

  F = zeros(FE_space.Nₛᵘ, ROM_info.nₛ_DEIM)

  for nₛ= 1:ROM_info.nₛ_MDEIM
    @info "Snapshot $nₛ at time instant $t, forcing"
    μ_nₛ = parse.(Float64, split(chop(μ[nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(problem_ntuple, ROM_info, μ_nₛ)
    F[:, nₛ] = assemble_forcing(FE_space, ROM_info, parametric_info)(t)[:]
  end

  F

end

function build_H_snapshots(FE_space::SteadyProblem, ROM_info::Info, μ::Array)

  H = zeros(FE_space.Nₛᵘ, ROM_info.nₛ_DEIM)

  for i_nₛ = 1:ROM_info.nₛ_DEIM
    @info "Snapshot number $i_nₛ, neumann datum"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(problem_ntuple, ROM_info, μ_i)
    H[:, i_nₛ] = assemble_neumann_datum(FE_space, ROM_info, parametric_info)[:]
  end

  H

end

function build_H_snapshots(FE_space::UnsteadyProblem, ROM_info::Info, μ::Array)

  Nₜ = convert(Int64, ROM_info.T/ROM_info.δt)
  δtθ = ROM_info.δt*ROM_info.θ
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ

  parametric_info = get_parametric_specifics(problem_ntuple, ROM_info, μ)
  H_t = assemble_neumann_datum(FE_space, ROM_info, parametric_info)
  H = zeros(FE_space.Nₛᵘ, Nₜ)

  for i_t = 1:Nₜ
    @info "Snapshot at time step $i_t, neumann datum"
    H[:, i_t] = H_t(times_θ[i_t])[:]
  end

  H

end

function build_H_snapshots(FE_space::UnsteadyProblem, ROM_info::Info, μ::Array, t::Float64)

  H = zeros(FE_space.Nₛᵘ, ROM_info.nₛ_DEIM)

  for nₛ = 1:ROM_info.nₛ_MDEIM
    @info "Snapshot $nₛ at time instant $t, neumann datum"
    μ_nₛ = parse.(Float64, split(chop(μ[nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(problem_ntuple, ROM_info, μ_nₛ)
    H[:, nₛ] = assemble_neumann_datum(FE_space, ROM_info, parametric_info)(t)[:]
  end

  H

end
