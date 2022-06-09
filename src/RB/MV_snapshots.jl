include("../FEM/LagrangianQuad.jl")

function build_M_snapshots(
  FEMSpace::SteadyProblem,
  RBInfo::Info,
  μ::Matrix) ::Matrix

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
  timesθ::Vector) ::Tuple

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

function build_A_snapshots(
  FEMSpace::SteadyProblem,
  RBInfo::Info,
  μ::Matrix) ::Matrix

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
  timesθ::Vector) ::Tuple

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

function get_snaps_MDEIM(
  FEMSpace::SteadyProblem,
  RBInfo::Info,
  μ::Matrix,
  var="A") ::Matrix
  if var == "A"
    snaps = build_A_snapshots(FEMSpace, RBInfo, μ)
  elseif var == "M"
    snaps = build_M_snapshots(FEMSpace, RBInfo, μ)
  else
    error("Run MDEIM on A or M only")
  end
  snaps
end

function get_LagrangianQuad_info(FEMSpace::UnsteadyProblem) ::Tuple
  include("/home/user1/git_repos/Mabla.jl/src/FEM/LagrangianQuad.jl")
  ξₖ = get_cell_map(FEMSpace.Ω)
  Qₕ_cell_point = get_cell_points(FEMSpace.Qₕ)
  qₖ = get_data(Qₕ_cell_point)
  phys_quadp = lazy_map(evaluate,ξₖ,qₖ)
  ncells = length(phys_quadp)
  nquad_cell = length(phys_quadp[1])
  nquad = nquad_cell*ncells
  refFE_quad = Gridap.ReferenceFE(lagrangianQuad,Float64,FEMInfo.order)
  V₀_quad = TestFESpace(model,refFE_quad,conformity=:L2)
  return phys_quadp,ncells,nquad_cell,nquad,V₀_quad
end

function standard_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  μ::Matrix,
  timesθ::Vector,
  var="A") ::Tuple

  for k = 1:RBInfo.nₛ_MDEIM
    @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    if var == "A"
      snapsₖ,row_idx = build_A_snapshots(FEMSpace,RBInfo,μₖ,timesθ)
    elseif var == "M"
      snapsₖ,row_idx = build_M_snapshots(FEMSpace,RBInfo,μₖ,timesθ)
    else
      error("Run MDEIM on A or M only")
    end
    #compressed_snapsₖ, _ = POD(snapsₖ, RBInfo.ϵₛ)
    compressed_snapsₖ, _ = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
    if k == 1
      global row_idx = row_idx
      global snaps = compressed_snapsₖ
    else
      #snaps_POD,_ = POD(hcat(snaps, compressed_snapsₖ), RBInfo.ϵₛ)
      #global snaps = snaps_POD
      global snaps = hcat(snaps, compressed_snapsₖ)
    end
  end
  return snaps,row_idx
end

#=
μ = load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
timesθ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+RBInfo.δt*RBInfo.θ
for k = 1:RBInfo.nₛ_MDEIM
  @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
  μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
  snapsₖ,row_idx = build_A_snapshots(FEMSpace,RBInfo,μₖ,timesθ)

  compressed_snapsₖ, _ = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
  if k == 1
    global row_idx = row_idx
    global snaps₁ = compressed_snapsₖ
    global snaps₂ = compressed_snapsₖ
  else
    global snaps₁ = hcat(snaps₁, compressed_snapsₖ)
    snaps_POD,_ = POD(hcat(snaps₂, compressed_snapsₖ), RBInfo.ϵₛ)
    global snaps₂ = snaps_POD
  end
end

μ₉₅ = parse.(Float64, split(chop(μ[95]; head=1, tail=1), ','))
Param = get_ParamInfo(problem_ntuple,RBInfo,μ₉₅)
A_μ_sparse = build_sparse_mat(FEMInfo,FEMSpace,Param,RBVars.S.sparse_el_A,timesθ;var="A")
A₉₅_₁₀ = A_μ_sparse[:,(10-1)*FEMSpace.Nₛᵘ+1:10*FEMSpace.Nₛᵘ][:]

MDEIM_mat, Σ₁ = M_DEIM_POD(snaps₁, RBInfo.ϵₛ)
MDEIM_mat, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(MDEIM_mat, Σ₁)
MDEIMᵢ_mat = MDEIM_mat[MDEIM_idx,:]
MDEIM_idx_sparse = from_full_idx_to_sparse_idx(MDEIM_idx,row_idx,FEMSpace.Nₛᵘ)
θ₁ = zeros(size(MDEIM_mat)[2], RBVars.Nₜ)
Nₛᵘ=FEMSpace.Nₛᵘ
for (iₜ,_) in enumerate(timesθ)
  θ₁[:,iₜ] = M_DEIM_online(A_μ_sparse[:,(iₜ-1)*Nₛᵘ+1:iₜ*Nₛᵘ],MDEIMᵢ_mat,MDEIM_idx)
end
A₉₅_approx₁_₁₀ = MDEIM_mat*θ₁[:,10]

MDEIM_mat, Σ₂ = M_DEIM_POD(snaps₂, RBInfo.ϵₛ)
MDEIM_mat, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(MDEIM_mat, Σ₂)
MDEIMᵢ_mat = MDEIM_mat[MDEIM_idx,:]
MDEIM_idx_sparse = from_full_idx_to_sparse_idx(MDEIM_idx,row_idx,FEMSpace.Nₛᵘ)
θ₂ = zeros(size(MDEIM_mat)[2], RBVars.Nₜ)
for (iₜ,_) in enumerate(timesθ)
  θ₂[:,iₜ] = M_DEIM_online(A_μ_sparse[:,(iₜ-1)*Nₛᵘ+1:iₜ*Nₛᵘ],MDEIMᵢ_mat,MDEIM_idx)
end
A₉₅_approx₂_₁₀ = MDEIM_mat*θ₂[:,10]

norm(A₉₅_approx₁_₁₀ - A₉₅_₁₀)/norm(A₉₅_₁₀)
norm(A₉₅_approx₂_₁₀ - A₉₅_₁₀)/norm(A₉₅_₁₀)
=#

function standard_MDEIM_sampling(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  μ::Matrix,
  timesθ::Vector,
  nₜ::Int64,
  var="A") ::Tuple

  for k = 1:RBInfo.nₛ_MDEIM
    @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    timesθₖ = timesθ[rand(1:length(timesθ),nₜ)]
    if var == "A"
      snapsₖ,row_idx = build_A_snapshots(FEMSpace,RBInfo,μₖ,timesθₖ)
    elseif var == "M"
      snapsₖ,row_idx = build_M_snapshots(FEMSpace,RBInfo,μₖ,timesθₖ)
    else
      error("Run MDEIM on A or M only")
    end
    if k == 1
      global row_idx = row_idx
      global snaps = snapsₖ
    else
      global snaps = hcat(snaps, snapsₖ)
    end
  end
  return snaps,row_idx
end

function functional_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  μ::Matrix,
  timesθ::Vector,
  var="A") ::Tuple

  phys_quadp,ncells,nquad_cell,nquad,V₀_quad = get_LagrangianQuad_info(FEMSpace)
  for k = 1:RBInfo.nₛ_MDEIM
    @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    Param = get_ParamInfo(problem_ntuple, RBInfo, μₖ)
    if var == "A"
      paramsₖ = [Param.α(phys_quadp[n][q],t_θ)
        for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
    elseif var == "M"
      paramsₖ = [Param.m(phys_quadp[n][q],t_θ)
        for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
    else
      error("Run MDEIM on A or M only")
    end
    compressed_paramsₖ,_ = POD(reshape(paramsₖ,nquad,:), RBInfo.ϵₛ^2)
    if k == 1
      global params = compressed_paramsₖ
    else
      global params = hcat(params, compressed_paramsₖ)
    end
  end
  Θmat,_ = POD(params, RBInfo.ϵₛ^2)
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
      global snaps = zeros(length(row_idx),Q)
    end
    global snaps[:,q] = v
  end
  return snaps,row_idx
end

function functional_MDEIM_sampling(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  μ::Matrix,
  timesθ::Vector,
  nₜ::Int64,
  var="A") ::Tuple
  phys_quadp,ncells,nquad_cell,nquad,V₀_quad = get_LagrangianQuad_info(FEMSpace)
  for k = 1:RBInfo.nₛ_MDEIM
    @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    Param = get_ParamInfo(problem_ntuple, RBInfo, μₖ)
    timesθₖ = timesθ[rand(1:length(timesθ),nₜ)]
    if var == "A"
      paramsₖ = [Param.α(phys_quadp[n][q],t_θ)
        for t_θ = timesθₖ for n = 1:ncells for q = 1:nquad_cell]
    elseif var == "M"
      paramsₖ = [Param.m(phys_quadp[n][q],t_θ)
        for t_θ = timesθₖ for n = 1:ncells for q = 1:nquad_cell]
    else
      error("Run MDEIM on A or M only")
    end
    paramsₖ = reshape(paramsₖ,nquad,:)
    if k == 1
      global params = paramsₖ
    else
      global params = hcat(params, paramsₖ)
    end
  end
  Θmat = params
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
      global snaps = zeros(length(row_idx),Q)
    end
    global snaps[:,q] = v
  end
  return snaps,row_idx
end

function get_snaps_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  μ::Matrix,
  timesθ::Vector,
  var="A") ::Tuple

  snaps = Matrix{Float64}[]
  row_idx = Float64[]

  if RBInfo.space_time_M_DEIM
    for k = 1:RBInfo.nₛ_MDEIM
      @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
      μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
      if var == "A"
        snapsₖ,row_idx = build_A_snapshots(FEMSpace,RBInfo,μₖ,timesθ)
      elseif var == "M"
        snapsₖ,row_idx = build_M_snapshots(FEMSpace,RBInfo,μₖ,timesθ)
      else
        error("Run MDEIM on A or M only")
      end
      if k == 1
        row_idx = row_idx
        snaps = zeros(length(row_idx)*length(timesθ),RBInfo.nₛ_MDEIM)
      end
      snaps[:,k] = snapsₖ[:]
    end
  elseif RBInfo.functional_M_DEIM
    if RBInfo.sampling_MDEIM
      nₜ = floor(Int,length(timesθ)*RBInfo.sampling_percentage)
      snaps,row_idx = functional_MDEIM_sampling(FEMSpace,RBInfo,μ,timesθ,nₜ,var)
    else
      snaps,row_idx = functional_MDEIM(FEMSpace,RBInfo,μ,timesθ,var)
    end
  else
    if RBInfo.sampling_MDEIM
      nₜ = floor(Int,length(timesθ)*RBInfo.sampling_percentage)
      snaps,row_idx = standard_MDEIM_sampling(FEMSpace,RBInfo,μ,timesθ,nₜ,var)
    else
      snaps,row_idx = standard_MDEIM(FEMSpace,RBInfo,μ,timesθ,var)
    end
  end
  snaps,row_idx
end

function build_F_snapshots(
  FEMSpace::SteadyProblem,
  RBInfo::Info,
  μ::Matrix) ::Matrix

  F = zeros(FEMSpace.Nₛᵘ, RBInfo.nₛ_DEIM)

  for i_nₛ = 1:RBInfo.nₛ_DEIM
    @info "Snapshot number $i_nₛ, forcing"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    Param = get_ParamInfo(problem_ntuple, RBInfo, μ_i)
    F[:, i_nₛ] = assemble_forcing(FEMSpace, RBInfo, Param)[:]
  end
  F
end

function build_F_snapshots(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  μ::Vector,
  timesθ::Vector) ::Matrix

  Nₜ = length(timesθ)
  Param = get_ParamInfo(problem_ntuple, RBInfo, μ)
  F_t = assemble_forcing(FEMSpace, RBInfo, Param)
  F = zeros(FEMSpace.Nₛᵘ, Nₜ)

  for i_t = 1:Nₜ
    @info "Snapshot at time step $i_t, forcing"
    F[:,i_t] = F_t(timesθ[i_t])[:]
  end
  F
end

function build_H_snapshots(
  FEMSpace::SteadyProblem,
  RBInfo::Info,
  μ::Matrix) ::Matrix

  H = zeros(FEMSpace.Nₛᵘ, RBInfo.nₛ_DEIM)

  for i_nₛ = 1:RBInfo.nₛ_DEIM
    @info "Snapshot number $i_nₛ, neumann datum"
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    Param = get_ParamInfo(problem_ntuple, RBInfo, μ_i)
    H[:, i_nₛ] = assemble_neumann_datum(FEMSpace, RBInfo, Param)[:]
  end
  H
end

function build_H_snapshots(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  μ::Vector,
  timesθ::Vector) ::Matrix

  Nₜ = length(timesθ)
  Param = get_ParamInfo(problem_ntuple, RBInfo, μ)
  H_t = assemble_neumann_datum(FEMSpace, RBInfo, Param)
  H = zeros(FEMSpace.Nₛᵘ, Nₜ)

  for i_t = 1:Nₜ
    @info "Snapshot at time step $i_t, neumann datum"
    H[:, i_t] = H_t(timesθ[i_t])[:]
  end
  H
end

function get_snaps_DEIM(
  FEMSpace::SteadyProblem,
  RBInfo::Info,
  μ::Matrix,
  var="F") ::Matrix
  if var == "F"
    snaps = build_F_snapshots(FEMSpace, RBInfo, μ)
  elseif var == "H"
    snaps = build_H_snapshots(FEMSpace, RBInfo, μ)
  else
    error("Run DEIM on F or H only")
  end
  snaps
end

function standard_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  μ::Matrix,
  timesθ::Vector,
  var="F") ::Matrix

  for k = 1:RBInfo.nₛ_DEIM
    @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    if var == "F"
      snapsₖ = build_F_snapshots(FEMSpace,RBInfo,μₖ,timesθ)
    elseif var == "H"
      snapsₖ = build_H_snapshots(FEMSpace,RBInfo,μₖ,timesθ)
    else
      error("Run DEIM on F or H only")
    end
    compressed_snapsₖ, _ = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
    if k == 1
      global snaps = compressed_snapsₖ
    else
      global snaps = hcat(snaps, compressed_snapsₖ)
    end
  end
  snaps
end

function standard_DEIM_sampling(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  μ::Matrix,
  timesθ::Vector,
  nₜ::Int64,
  var="F") ::Matrix

  for k = 1:RBInfo.nₛ_DEIM
    @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    timesθₖ = timesθ[rand(1:length(timesθ),nₜ)]
    if var == "F"
      snapsₖ = build_F_snapshots(FEMSpace,RBInfo,μₖ,timesθₖ)
    elseif var == "H"
      snapsₖ = build_H_snapshots(FEMSpace,RBInfo,μₖ,timesθₖ)
    else
      error("Run DEIM on F or H only")
    end
    if k == 1
      global snaps = snapsₖ
    else
      global snaps = hcat(snaps, snapsₖ)
    end
  end
  snaps
end

function functional_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  μ::Matrix,
  timesθ::Vector,
  var="F") ::Matrix

  phys_quadp,ncells,nquad_cell,nquad,V₀_quad = get_LagrangianQuad_info(FEMSpace)
  for k = 1:RBInfo.nₛ_DEIM
    @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    Param = get_ParamInfo(problem_ntuple, RBInfo, μₖ)
    if var == "F"
      paramsₖ = [Param.f(phys_quadp[n][q],t_θ)
        for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
    elseif var == "H"
      paramsₖ = [Param.h(phys_quadp[n][q],t_θ)
        for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
    else
      error("Run DEIM on F or H only")
    end
    compressed_paramsₖ,_ = POD(reshape(paramsₖ,nquad,:), RBInfo.ϵₛ^2)
    if k == 1
      global params = compressed_paramsₖ
    else
      global params = hcat(params, compressed_paramsₖ)
    end
  end
  Θmat,_ = POD(params, RBInfo.ϵₛ^2)
  Q = size(Θmat)[2]
  snaps = zeros(FEMSpace.Nₛᵘ,Q)
  for q = 1:Q
    Θq = FEFunction(V₀_quad,Θmat[:,q])
    if var == "F"
      snaps[:,q] = assemble_vector(∫(FEMSpace.ϕᵥ*Θq)*FEMSpace.dΩ,FEMSpace.V₀)
    elseif var == "H"
      snaps[:,q] = assemble_vector(∫(FEMSpace.ϕᵥ*Θq)*FEMSpace.dΓn,FEMSpace.V₀)
    end
  end
  return snaps
end

function functional_DEIM_sampling(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  μ::Matrix,
  timesθ::Vector,
  nₜ::Int64,
  var="F") ::Matrix

  phys_quadp,ncells,nquad_cell,nquad,V₀_quad = get_LagrangianQuad_info(FEMSpace)
  for k = 1:RBInfo.nₛ_DEIM
    @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    Param = get_ParamInfo(problem_ntuple, RBInfo, μₖ)
    timesθₖ = timesθ[rand(1:length(timesθ),nₜ)]
    if var == "F"
      paramsₖ = [Param.f(phys_quadp[n][q],t_θ)
        for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
    elseif var == "H"
      paramsₖ = [Param.h(phys_quadp[n][q],t_θ)
        for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
    else
      error("Run DEIM on F or H only")
    end
    paramsₖ = reshape(paramsₖ,nquad,:)
    if k == 1
      global params = paramsₖ
    else
      global params = hcat(params, paramsₖ)
    end
  end
  Θmat,_ = POD(params, RBInfo.ϵₛ^2)
  Q = size(Θmat)[2]
  snaps = zeros(FEMSpace.Nₛᵘ,Q)
  for q = 1:Q
    Θq = FEFunction(V₀_quad,Θmat[:,q])
    if var == "F"
      snaps[:,q] = assemble_vector(∫(FEMSpace.ϕᵥ*Θq)*FEMSpace.dΩ,FEMSpace.V₀)
    elseif var == "H"
      snaps[:,q] = assemble_vector(∫(FEMSpace.ϕᵥ*Θq)*FEMSpace.dΓn,FEMSpace.V₀)
    end
  end
  snaps
end

function get_snaps_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  μ::Matrix,
  timesθ::Vector,
  var="F") ::Matrix

  if RBInfo.space_time_M_DEIM
    for k = 1:RBInfo.nₛ_MDEIM
      @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
      μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
      if var == "F"
        snapsₖ = build_F_snapshots(FEMSpace, RBInfo, μₖ)
      elseif var == "H"
        snapsₖ = build_H_snapshots(FEMSpace, RBInfo, μₖ)
      else
        error("Run DEIM on F or H only")
      end
      if k == 1
        global snaps = snapsₖ[:]
      else
        global snaps = hcat(snaps, snapsₖ[:])
      end
    end
  elseif RBInfo.functional_M_DEIM
    if RBInfo.sampling_MDEIM
      nₜ = floor(Int,length(timesθ)*RBInfo.sampling_percentage)
      snaps = functional_DEIM_sampling(FEMSpace,RBInfo,μ,timesθ,nₜ,var)
    else
      snaps = functional_DEIM(FEMSpace,RBInfo,μ,timesθ,var)
    end
  else
    if RBInfo.sampling_MDEIM
      nₜ = floor(Int,length(timesθ)*RBInfo.sampling_percentage)
      snaps = standard_DEIM_sampling(FEMSpace,RBInfo,μ,timesθ,nₜ,var)
    else
      snaps = standard_DEIM(FEMSpace,RBInfo,μ,timesθ,var)
    end
  end
  snaps
end

function assemble_parametric_FE_structure(
  FEMSpace::PoissonUnsteady,
  Θq::FEFunction,
  var::String)
  if var == "A"
    Matq = (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Θq*∇(FEMSpace.ϕᵤ(0.0))))*FEMSpace.dΩ,
      FEMSpace.V(0.0), FEMSpace.V₀))
  elseif var == "M"
    Matq = (assemble_matrix(∫(FEMSpace.ϕᵥ*(Θq*FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
      FEMSpace.V(0.0), FEMSpace.V₀))
  elseif var == "F"
    snaps[:,q] = assemble_vector(∫(FEMSpace.ϕᵥ*Θq)*FEMSpace.dΩ,FEMSpace.V₀)
  elseif var == "H"
    snaps[:,q] = assemble_vector(∫(FEMSpace.ϕᵥ*Θq)*FEMSpace.dΓn,FEMSpace.V₀)
  else
    error("Need to assemble an unrecognized FE structure")
  end
end

function assemble_parametric_FE_structure(
  FEMSpace::StokesUnsteady,
  Θq::FEFunction,
  var::String)
  if var == "A"
    Matq = (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Θq*∇(FEMSpace.ϕᵤ(0.0))))*FEMSpace.dΩ,
      FEMSpace.V(0.0), FEMSpace.V₀))
  elseif var == "M"
    Matq = (assemble_matrix(∫(FEMSpace.ϕᵥ⋅(Θq*FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
      FEMSpace.V(0.0), FEMSpace.V₀))
  elseif var == "F"
    snaps[:,q] = assemble_vector(∫(FEMSpace.ϕᵥ⋅Θq)*FEMSpace.dΩ,FEMSpace.V₀)
  elseif var == "H"
    snaps[:,q] = assemble_vector(∫(FEMSpace.ϕᵥ⋅Θq)*FEMSpace.dΓn,FEMSpace.V₀)
  else
    error("Need to assemble an unrecognized FE structure")
  end
end
