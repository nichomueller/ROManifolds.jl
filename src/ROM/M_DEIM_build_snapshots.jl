function initialize_M(ROM_info, μ::Array)
  @info "Snapshot number 1, mass"
  parametric_info = get_parametric_specifics(ROM_info, μ)
  FE_space = get_FE_space(problem_info, parametric_info.model)
  M_μ = assemble_mass(FE_space, ROM_info, parametric_info)
  Nₕ = size(M_μ)[1]
  row_idx, val = findnz(M_μ[:])
  M = sparse(row_idx, ones(length(row_idx)), val, Nₕ^2, ROM_info.nₛ_MDEIM)
  M, FE_space
end

function build_M_snapshots(problem_info::ProblemSpecifics, ROM_info)

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))
  @info "Building $(ROM_info.nₛ_MDEIM) snapshots of M"

  μ₁ = parse.(Float64, split(chop(μ[1]; head=1, tail=1), ','))
  M, FE_space = initialize_M(ROM_info, μ₁)

  for i_nₛ = 2:ROM_info.nₛ_MDEIM
    @info "Snapshot number $i_nₛ, mass"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_i)
    M_μ_i = assemble_mass(FE_space, ROM_info, parametric_info)
    i, v = findnz(M_μ_i[:])
    M[:, i_nₛ] = sparse(i, ones(length(i)), v)
  end

  M

end

function initialize_M(ROM_info, FE_space, parametric_info, Nₜ)
  @info "Snapshot at time step 1, mass"
  M_μ_t = assemble_mass(FE_space, ROM_info, parametric_info)
  M_μ = M_μ_t(ROM_info.t₀+ROM_info.δt*ROM_info.θ)
  Nₕ = size(M_μ)[1]
  row_idx, val = findnz(M_μ[:])
  M = sparse(row_idx, ones(length(row_idx)), val, Nₕ^2, Nₜ)
  M
end

function build_M_snapshots(problem_info::ProblemSpecificsUnsteady, ROM_info, μ::Array)

  Nₜ = convert(Int64, ROM_info.T/ROM_info.δt)
  δtθ = ROM_info.δt*ROM_info.θ
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ

  parametric_info = get_parametric_specifics(ROM_info, μ)
  FE_space = get_FE_space(problem_info, parametric_info.model)

  M = initialize_M(ROM_info, FE_space, parametric_info, Nₜ)

  for i_t = 2:Nₜ
    @info "Snapshot at time step $i_t, mass"
    M_i_t = assemble_mass(FE_space, ROM_info, parametric_info)
    M_t = M_i_t(times_θ[i_t])
    i, v = findnz(M_t[:])
    M[:, i_t] = sparse(i, ones(length(i)), v)
  end

  M

end

function initialize_M(ROM_info, μ::Array, t::Float64)
  @info "Snapshot 1 at time instant $t, mass"
  parametric_info = get_parametric_specifics(ROM_info, μ)
  FE_space = get_FE_space(problem_info, parametric_info.model)
  M_μ_t = assemble_mass(FE_space, ROM_info, parametric_info)(t)
  Nₕ = size(M_μ_t)[1]
  row_idx, val = findnz(M_μ_t[:])
  M = sparse(row_idx, ones(length(row_idx)), val, Nₕ^2, ROM_info.nₛ_MDEIM)
  M, FE_space
end

function build_M_snapshots(problem_info::ProblemSpecificsUnsteady, ROM_info, μ::Array, t::Float64)

  μ₁ = parse.(Float64, split(chop(μ[1]; head=1, tail=1), ','))
  M, FE_space = initialize_M(ROM_info, μ₁, t)

  for nₛ= 2:ROM_info.nₛ_MDEIM
    @info "Snapshot $nₛ at time instant $t, mass"
    μ_nₛ = parse.(Float64, split(chop(μ[nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_nₛ)
    M_i_t = assemble_mass(FE_space, ROM_info, parametric_info)(t)
    i, v = findnz(M_i_t[:])
    M[:, nₛ] = sparse(i, ones(length(i)), v)
  end

  M

end

function initialize_A(ROM_info, μ::Array)
  @info "Snapshot number 1, stiffness"
  parametric_info = get_parametric_specifics(ROM_info, μ)
  FE_space = get_FE_space(problem_info, parametric_info.model)
  A_μ = assemble_stiffness(FE_space, ROM_info, parametric_info)
  Nₕ = size(A_μ)[1]
  row_idx, val = findnz(A_μ[:])
  A = sparse(row_idx, ones(length(row_idx)), val, Nₕ^2, ROM_info.nₛ_MDEIM)
  A, FE_space
end

function build_A_snapshots(problem_info::ProblemSpecifics, ROM_info)

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))
  @info "Building $(ROM_info.nₛ_MDEIM) snapshots of A"

  μ₁ = parse.(Float64, split(chop(μ[1]; head=1, tail=1), ','))
  A, FE_space = initialize_A(ROM_info, μ₁)

  for i_nₛ = 2:ROM_info.nₛ_MDEIM
    @info "Snapshot number $i_nₛ, stiffness"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_i)
    A_μ_i = assemble_stiffness(FE_space, ROM_info, parametric_info)
    i, v = findnz(A_μ_i[:])
    A[:, i_nₛ] = sparse(i, ones(length(i)), v)
  end

  A

end

function initialize_A(ROM_info, FE_space, parametric_info, Nₜ)
  @info "Snapshot at time step 1, stiffness"
  A_μ_t = assemble_stiffness(FE_space, ROM_info, parametric_info)
  A_μ = A_μ_t(ROM_info.t₀+ROM_info.δt*ROM_info.θ)
  Nₕ = size(A_μ)[1]
  row_idx, val = findnz(A_μ[:])
  A = sparse(row_idx, ones(length(row_idx)), val, Nₕ^2, Nₜ)
  A
end

function build_A_snapshots(problem_info::ProblemSpecificsUnsteady, ROM_info, μ::Array)

  Nₜ = convert(Int64, ROM_info.T/ROM_info.δt)
  δtθ = ROM_info.δt*ROM_info.θ
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ

  parametric_info = get_parametric_specifics(ROM_info, μ)
  FE_space = get_FE_space(problem_info, parametric_info.model)

  A = initialize_A(ROM_info, FE_space, parametric_info, Nₜ)

  for i_t = 2:Nₜ
    @info "Snapshot at time step $i_t, stiffness"
    A_μ_t = assemble_stiffness(FE_space, ROM_info, parametric_info)
    A_μ_i = A_μ_t(times_θ[i_t])
    i, v = findnz(A_μ_i[:])
    A[:, i_t] = sparse(i, ones(length(i)), v)
  end

  A

end

function initialize_A(ROM_info, μ::Array, t::Float64)
  @info "Snapshot 1 at time instant $t, stiffness"
  parametric_info = get_parametric_specifics(ROM_info, μ)
  FE_space = get_FE_space(problem_info, parametric_info.model)
  A_μ_t = assemble_stiffness(FE_space, ROM_info, parametric_info)(t)
  Nₕ = size(A_μ_t)[1]
  row_idx, val = findnz(A_μ_t[:])
  A = sparse(row_idx, ones(length(row_idx)), val, Nₕ^2, ROM_info.nₛ_MDEIM)
  A, FE_space
end

function build_A_snapshots(problem_info::ProblemSpecificsUnsteady, ROM_info, μ::Array, t::Float64)

  μ₁ = parse.(Float64, split(chop(μ[1]; head=1, tail=1), ','))
  A, FE_space = initialize_A(ROM_info, μ₁, t)

  for nₛ= 2:ROM_info.nₛ_MDEIM
    @info "Snapshot $nₛ at time instant $t, stiffness"
    μ_nₛ = parse.(Float64, split(chop(μ[nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_nₛ)
    A_i_t = assemble_stiffness(FE_space, ROM_info, parametric_info)(t)
    i, v = findnz(A_i_t[:])
    A[:, nₛ] = sparse(i, ones(length(i)), v)
  end

  A

end

function initialize_F(ROM_info, μ::Array)
  @info "Snapshot number 1"
  parametric_info = get_parametric_specifics(ROM_info, μ)
  FE_space = get_FE_space(problem_info, parametric_info.model)
  F_μ, _ = assemble_forcing(FE_space, ROM_info, parametric_info)
  F = zeros(length(F_μ), ROM_info.nₛ_DEIM)
  F[:, 1] = F_μ
  F
end

function build_F_snapshots(problem_info, ROM_info)

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))
  @info "Building $(ROM_info.nₛ_DEIM) snapshots of F"

  μ₁ = parse.(Float64, split(chop(μ[1]; head=1, tail=1), ','))
  F = initialize_F(ROM_info, μ₁)

  for i_nₛ = 2:ROM_info.nₛ_DEIM
    @info "Snapshot number $i_nₛ"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_i)
    FE_space = get_FE_space(problem_info, parametric_info.model)
    F_i, _ = assemble_forcing(FE_space, ROM_info, parametric_info)
    F[:, i_nₛ] = F_i[:]
  end

  F

end

function initialize_F(ROM_info, FE_space, parametric_info, Nₜ)
  @info "Snapshot at time step 1, forcing"
  F_μ_t, _ = assemble_forcing(FE_space, ROM_info, parametric_info)
  F_μ = F_μ_t(ROM_info.t₀+ROM_info.δt*ROM_info.θ)
  F = zeros(length(F_μ), Nₜ)
  F[:,1] = F_μ
  F
end

function build_F_snapshots(problem_info::ProblemSpecificsUnsteady, ROM_info, μ::Array)

  Nₜ = convert(Int64, ROM_info.T/ROM_info.δt)
  δtθ = ROM_info.δt*ROM_info.θ
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ

  parametric_info = get_parametric_specifics(ROM_info, μ)
  FE_space = get_FE_space(problem_info, parametric_info.model)

  F = initialize_F(ROM_info, FE_space, parametric_info, Nₜ)

  for i_t = 2:Nₜ
    @info "Snapshot at time step $i_t, stiffness"
    F_i_t, _ = assemble_forcing(FE_space, ROM_info, parametric_info)
    F_t = F_i_t(times_θ[i_t])
    i, v = findnz(F_t[:])
    F[:, i_t] = sparse(i, ones(length(i)), v)
  end

  F

end

function initialize_F(ROM_info, μ::Array, t::Float64)
  @info "Snapshot 1 at time instant $t, forcing"
  parametric_info = get_parametric_specifics(ROM_info, μ)
  FE_space = get_FE_space(problem_info, parametric_info.model)
  F_μ_t, _ = assemble_forcing(FE_space, ROM_info, parametric_info)(t)
  F = zeros(length(F_μ), ROM_info.nₛ_DEIM)
  F[:,1] = F_μ_t
  F, FE_space
end

function build_F_snapshots(problem_info::ProblemSpecificsUnsteady, ROM_info, μ::Array, t::Float64)

  μ₁ = parse.(Float64, split(chop(μ[1]; head=1, tail=1), ','))
  F, FE_space = initialize_F(ROM_info, μ₁, t)

  for nₛ= 2:ROM_info.nₛ_MDEIM
    @info "Snapshot $nₛ at time instant $t, forcing"
    μ_nₛ = parse.(Float64, split(chop(μ[nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_nₛ)
    F_i_t, _ = assemble_forcing(FE_space, ROM_info, parametric_info)(t)
    F[:, nₛ] = F_i_t
  end

  F

end

function initialize_H(ROM_info, μ::Array)
  @info "Snapshot number 1, Neumann term"
  parametric_info = get_parametric_specifics(ROM_info, μ)
  FE_space = get_FE_space(problem_info, parametric_info.model)
  _, H_μ = assemble_forcing(FE_space, ROM_info, parametric_info)
  H = zeros(length(H_μ), ROM_info.nₛ_DEIM)
  H[:, 1] = H_μ
  H
end

function build_H_snapshots(problem_info, ROM_info)

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))
  @info "Building $(ROM_info.nₛ_DEIM) snapshots of H"

  μ₁ = parse.(Float64, split(chop(μ[1]; head=1, tail=1), ','))
  H = initialize_F(ROM_info, μ₁)

  for i_nₛ = 2:ROM_info.nₛ_DEIM
    @info "Snapshot number $i_nₛ"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_i)
    FE_space = get_FE_space(problem_info, parametric_info.model)
    _, H_i = assemble_forcing(FE_space, ROM_info, parametric_info)
    H[:, i_nₛ] = H_i[:]
  end

  H

end

function initialize_H(ROM_info, FE_space, parametric_info, Nₜ)
  @info "Snapshot at time step 1, Neumann term"
  _, H_μ_t= assemble_forcing(FE_space, ROM_info, parametric_info)
  H_μ = H_μ_t(ROM_info.t₀+ROM_info.δt*ROM_info.θ)
  H = zeros(length(H_μ), Nₜ)
  H[:,1] = H_μ
  H
end

function build_H_snapshots(problem_info::ProblemSpecificsUnsteady, ROM_info, μ::Array)

  Nₜ = convert(Int64, ROM_info.T/ROM_info.δt)
  δtθ = ROM_info.δt*ROM_info.θ
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ

  parametric_info = get_parametric_specifics(ROM_info, μ)
  FE_space = get_FE_space(problem_info, parametric_info.model)

  H = initialize_H(ROM_info, FE_space, parametric_info, Nₜ)

  for i_t = 2:Nₜ
    @info "Snapshot at time step $i_t, stiffness"
    H_i_t, _ = assemble_forcing(FE_space, ROM_info, parametric_info)
    H_t = H_i_t(times_θ[i_t])
    i, v = findnz(H_t[:])
    H[:, i_t] = sparse(i, ones(length(i)), v)
  end

  H

end

function initialize_H(ROM_info, μ::Array, t::Float64)
  @info "Snapshot 1 at time instant $t, Neumann term"
  parametric_info = get_parametric_specifics(ROM_info, μ)
  FE_space = get_FE_space(problem_info, parametric_info.model)
  _, H_μ_t = assemble_forcing(FE_space, ROM_info, parametric_info)(t)
  H = zeros(length(H_μ), ROM_info.nₛ_DEIM)
  H[:,1] = H_μ_t
  H, FE_space
end

function build_H_snapshots(problem_info::ProblemSpecificsUnsteady, ROM_info, μ::Array, t::Float64)

  μ₁ = parse.(Float64, split(chop(μ[1]; head=1, tail=1), ','))
  H, FE_space = initialize_H(ROM_info, μ₁, t)

  for nₛ= 2:ROM_info.nₛ_MDEIM
    @info "Snapshot $nₛ at time instant $t, Neumann term"
    μ_nₛ = parse.(Float64, split(chop(μ[nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_nₛ)
    _, H_i_t = assemble_forcing(FE_space, ROM_info, parametric_info)(t)
    H[:, nₛ] = H_i_t
  end

  H

end
