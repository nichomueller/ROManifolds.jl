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

  row_idx,Σ = Float64[],Float64[]
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
    compressed_snapsₖ,Σ = POD(snapsₖ, RBInfo.ϵₛ)
    if k == 1
      global snaps = compressed_snapsₖ
    else
      snaps_POD,_ = POD(hcat(snaps, compressed_snapsₖ), RBInfo.ϵₛ)
      global snaps = snaps_POD
    end
  end
  return snaps,row_idx,Σ
end

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

  Σ = Float64[]
  for k = 1:RBInfo.nₛ_MDEIM
    @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    Param = get_ParamInfo(problem_ntuple, RBInfo, μₖ)
    if var == "A"
      Θₖ = [Param.α(phys_quadp[n][q],t_θ)
        for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
    elseif var == "M"
      Θₖ = [Param.m(phys_quadp[n][q],t_θ)
        for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
    else
      error("Run MDEIM on A or M only")
    end
    compressed_Θₖ,Σ = POD(reshape(Θₖ,nquad,:), RBInfo.ϵₛ)
    if k == 1
      global Θmat = compressed_Θₖ
    else
      Θ_POD,_ = POD(hcat(Θmat, compressed_Θₖ), RBInfo.ϵₛ)
      global Θmat = Θ_POD
    end
  end
  Q = size(Θmat)[2]
  for q = 1:Q
    Θq = FEFunction(V₀_quad,Θmat[:,q])
    Matq = assemble_parametric_FE_structure(FEMSpace,Θq,var)
    i,v = findnz(Matq[:])
    if q == 1
      global row_idx = i
      global snaps = zeros(length(row_idx),Q)
    end
    global snaps[:,q] = v
  end
  return snaps,row_idx,Σ
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
    Matq = assemble_parametric_FE_structure(FEMSpace,Θq,var)
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
  Σ = Float64[]

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
      snaps,row_idx,Σ = functional_MDEIM(FEMSpace,RBInfo,μ,timesθ,var)
    end
  else
    if RBInfo.sampling_MDEIM
      nₜ = floor(Int,length(timesθ)*RBInfo.sampling_percentage)
      snaps,row_idx = standard_MDEIM_sampling(FEMSpace,RBInfo,μ,timesθ,nₜ,var)
    else
      snaps,row_idx,Σ = standard_MDEIM(FEMSpace,RBInfo,μ,timesθ,var)
    end
  end
  snaps,row_idx,Σ
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
    Θₖ = [Param.f(phys_quadp[n][q],t_θ)
        for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
    #= if var == "F"
      Θₖ = [Param.f(phys_quadp[n][q],t_θ)
        for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
    elseif var == "H"
      θₖ = [Param.h(phys_quadp[n][q],t_θ)
        for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
    else
      error("Run DEIM on F or H only")
    end =#
    compressed_Θₖ,_ = POD(reshape(Θₖ,nquad,:), RBInfo.ϵₛ)
    if k == 1
      global Θmat = compressed_Θₖ
    else
      Θ_POD,_ = POD(hcat(Θmat, compressed_Θₖ), RBInfo.ϵₛ)
      global Θmat = Θ_POD
    end
  end
  Q = size(Θmat)[2]
  snaps = zeros(FEMSpace.Nₛᵘ,Q)
  for q = 1:Q
    Θq = FEFunction(V₀_quad,Θmat[:,q])
    snaps[:,q] = assemble_parametric_FE_structure(FEMSpace,Θq,var)
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
    snaps[:,q] = assemble_parametric_FE_structure(FEMSpace,Θq,var)
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
  FEMSpace::FEMSpacePoissonUnsteady,
  Θq::FEFunction,
  var::String)
  if var == "A"
    return (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Θq*∇(FEMSpace.ϕᵤ(0.0))))*FEMSpace.dΩ,
      FEMSpace.V(0.0), FEMSpace.V₀))
  elseif var == "M"
    return (assemble_matrix(∫(FEMSpace.ϕᵥ*(Θq*FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
      FEMSpace.V(0.0), FEMSpace.V₀))
  elseif var == "F"
    return assemble_vector(∫(FEMSpace.ϕᵥ*Θq)*FEMSpace.dΩ,FEMSpace.V₀)
  elseif var == "H"
    return assemble_vector(∫(FEMSpace.ϕᵥ*Θq)*FEMSpace.dΓn,FEMSpace.V₀)
  else
    error("Need to assemble an unrecognized FE structure")
  end
end

function assemble_parametric_FE_structure(
  FEMSpace::FEMSpaceStokesUnsteady,
  Θq::FEFunction,
  var::String)
  if var == "A"
    return (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Θq*∇(FEMSpace.ϕᵤ(0.0))))*FEMSpace.dΩ,
      FEMSpace.V(0.0), FEMSpace.V₀))
  elseif var == "M"
    return (assemble_matrix(∫(FEMSpace.ϕᵥ⋅(Θq*FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
      FEMSpace.V(0.0), FEMSpace.V₀))
  elseif var == "F"
    return assemble_vector(∫(FEMSpace.ϕᵥ⋅Θq)*FEMSpace.dΩ,FEMSpace.V₀)
  elseif var == "H"
    return assemble_vector(∫(FEMSpace.ϕᵥ⋅Θq)*FEMSpace.dΓn,FEMSpace.V₀)
  else
    error("Need to assemble an unrecognized FE structure")
  end
end
