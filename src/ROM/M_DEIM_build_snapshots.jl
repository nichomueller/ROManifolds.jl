function build_M_snapshots(FE_space::SteadyProblem, ROM_info::Info, μ::Array)

  for i_nₛ = 1:ROM_info.nₛ_MDEIM
    @info "Snapshot number $i_nₛ, mass"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_i)
    M_i = assemble_mass(FE_space, ROM_info, parametric_info)
    i, v = findnz(M_i[:])
    if i_nₛ === 1
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

  parametric_info = get_parametric_specifics(ROM_info, μ)
  M_t = assemble_mass(FE_space, ROM_info, parametric_info)

  for i_t = 1:Nₜ
    @info "Snapshot at time step $i_t, mass"
    M_i = M_t(times_θ[i_t])
    i, v = findnz(M_i[:])
    if i_t === 1
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
    parametric_info = get_parametric_specifics(ROM_info, μ_nₛ)
    M_nₛ = assemble_mass(FE_space, ROM_info, parametric_info)(t)
    i, v = findnz(M_nₛ[:])
    if nₛ === 1
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
    parametric_info = get_parametric_specifics(ROM_info, μ_i)
    A_i = assemble_stiffness(FE_space, ROM_info, parametric_info)
    i, v = findnz(A_i[:])
    if i_nₛ === 1
      global A = sparse(i, ones(length(i)), v, FE_space.Nₛᵘ^2, ROM_info.nₛ_MDEIM)
    else
      global A[:, i_nₛ] = sparse(i, ones(length(i)), v)
    end
  end

  A

end

function build_A_snapshots(FE_space::UnsteadyProblem, ROM_info::Info, μ::Array)

  Nₜ = convert(Int64, ROM_info.T/ROM_info.δt)
  δtθ = ROM_info.δt*ROM_info.θ
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ

  parametric_info = get_parametric_specifics(ROM_info, μ)
  A_t = assemble_stiffness(FE_space, ROM_info, parametric_info)

  for i_t = 1:Nₜ
    @info "Snapshot at time step $i_t, stiffness"
    A_i = A_t(times_θ[i_t])
    i, v = findnz(A_i[:])
    if i_t === 1
      global A = sparse(i, ones(length(i)), v, FE_space.Nₛᵘ^2, Nₜ)
    else
      global A[:, i_t] = sparse(i, ones(length(i)), v)
    end
  end

  A

end

function build_A_snapshots(FE_space::UnsteadyProblem, ROM_info::Info, μ::Array, t::Float64)

  for nₛ = 1:ROM_info.nₛ_MDEIM
    @info "Snapshot $nₛ at time instant $t, stiffness"
    μ_nₛ = parse.(Float64, split(chop(μ[nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_nₛ)
    A_nₛ = assemble_stiffness(FE_space, ROM_info, parametric_info)(t)
    global i, v = findnz(A_nₛ[:])
    if nₛ === 1
      global A = zeros(length(i), ROM_info.nₛ_MDEIM)
    end
    global A[:, nₛ] = v
  end

  A, i

end

function build_F_snapshots(FE_space::SteadyProblem, ROM_info::Info, μ::Array)

  F = zeros(FE_space.Nₛᵘ, ROM_info.nₛ_DEIM)

  for i_nₛ = 1:ROM_info.nₛ_DEIM
    @info "Snapshot number $i_nₛ, forcing"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_i)
    F[:, i_nₛ] = assemble_forcing(FE_space, ROM_info, parametric_info)[:]
  end

  F

end

function build_F_snapshots(FE_space::UnsteadyProblem, ROM_info::Info, μ::Array)

  Nₜ = convert(Int64, ROM_info.T/ROM_info.δt)
  δtθ = ROM_info.δt*ROM_info.θ
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ

  parametric_info = get_parametric_specifics(ROM_info, μ)
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
    parametric_info = get_parametric_specifics(ROM_info, μ_nₛ)
    F[:, nₛ] = assemble_forcing(FE_space, ROM_info, parametric_info)(t)[:]
  end

  F

end

function build_H_snapshots(FE_space::SteadyProblem, ROM_info::Info, μ::Array)

  H = zeros(FE_space.Nₛᵘ, ROM_info.nₛ_DEIM)

  for i_nₛ = 1:ROM_info.nₛ_DEIM
    @info "Snapshot number $i_nₛ, neumann datum"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_i)
    H[:, i_nₛ] = assemble_neumann_datum(FE_space, ROM_info, parametric_info)[:]
  end

  H

end

function build_H_snapshots(FE_space::UnsteadyProblem, ROM_info::Info, μ::Array)

  Nₜ = convert(Int64, ROM_info.T/ROM_info.δt)
  δtθ = ROM_info.δt*ROM_info.θ
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ

  parametric_info = get_parametric_specifics(ROM_info, μ)
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
    parametric_info = get_parametric_specifics(ROM_info, μ_nₛ)
    H[:, nₛ] = assemble_neumann_datum(FE_space, ROM_info, parametric_info)(t)[:]
  end

  H

end
