include("../FEM/LagrangianQuad.jl")

function build_M_snapshots(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  μ::Matrix{T}) where T

  for i_nₛ = 1:RBInfo.nₛ_MDEIM
    println("Snapshot number $i_nₛ, mass")
    μ_i = parse.(T, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    Param = get_ParamInfo(RBInfo, problem_id, μ_i)
    M_i = assemble_mass(FEMSpace, RBInfo, Param)
    i, v = findnz(M_i[:])
    if i_nₛ == 1
      global row_idx = i
      global M = zeros(T,length(row_idx),RBInfo.nₛ_MDEIM)
    else
      global M[:,i_nₛ] = v
    end
  end
  M,row_idx
end

function build_M_snapshots(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{T},
  timesθ::Vector{T}) where T

  Nₜ = length(timesθ)
  Param = get_ParamInfo(RBInfo, problem_id, μ)
  M_t = assemble_mass(FEMSpace, RBInfo, Param)

  for i_t = 1:Nₜ
    M_i = M_t(timesθ[i_t])
    i, v = findnz(M_i[:])
    if i_t == 1
      global row_idx = i
      global M = zeros(T,length(row_idx),Nₜ)
    end
    global M[:,i_t] = v
  end
  M,row_idx
end

function build_A_snapshots(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  μ::Matrix{T}) where T

  for i_nₛ = 1:RBInfo.nₛ_MDEIM
    println("Snapshot number $i_nₛ, stiffness")
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    Param = get_ParamInfo(RBInfo, problem_id, μ_i)
    A_i = assemble_stiffness(FEMSpace, RBInfo, Param)
    i, v = findnz(A_i[:])
    if i_nₛ == 1
      global row_idx = i
      global A = zeros(T,length(row_idx),RBInfo.nₛ_MDEIM)
    else
      global A[:,i_nₛ] = v
    end
  end
  A,row_idx
end

function build_A_snapshots(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{T},
  timesθ::Vector{T}) where T

  Nₜ = length(timesθ)
  Param = get_ParamInfo(RBInfo, problem_id, μ)
  A_t = assemble_stiffness(FEMSpace, RBInfo, Param)

  for i_t = 1:Nₜ
    A_i = A_t(timesθ[i_t])
    i, v = findnz(A_i[:])
    if i_t == 1
      global row_idx = i
      global A = zeros(T,length(row_idx),Nₜ)
    end
    global A[:,i_t] = v
  end
  A, row_idx
end

function get_snaps_MDEIM(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  μ::Matrix{T},
  var="A") where T

  snaps,row_idx = build_snapshots(FEMSpace, RBInfo, μ, var)
  M_DEIM_POD(snaps, RBInfo.ϵₛ)...,row_idx
end

function get_LagrangianQuad_info(FEMSpace::UnsteadyProblem)
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
  RBInfo::ROMInfoUnsteady{T},
  μ::Matrix{T},
  timesθ::Vector{T},
  var="A") where T

  snaps,row_idx,Σ = Matrix{T}(undef,0,0),Int64[],T[]
  @simd for k = 1:RBInfo.S.nₛ_MDEIM
    println("Considering Parameter number $k/$(RBInfo.S.nₛ_MDEIM)")
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    snapsₖ,row_idx = build_snapshots(FEMSpace,RBInfo,μₖ,timesθ,var)
    snapsₖ,Σ = M_DEIM_POD(snapsₖ, RBInfo.S.ϵₛ)
    if k == 1
      snaps = snapsₖ
    else
      snaps,Σ = M_DEIM_POD(hcat(snaps, snapsₖ), RBInfo.S.ϵₛ)
    end
  end
  return snaps,Σ,row_idx
end

function standard_MDEIM_sampling(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Matrix{T},
  timesθ::Vector{T},
  nₜ::Int64,
  var="A") where T

  snaps,row_idx = Matrix{T}(undef,0,0),Int64[]
  @simd for k = 1:RBInfo.S.nₛ_MDEIM
    println("Considering Parameter number $k/$(RBInfo.S.nₛ_MDEIM)")
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    timesθₖ = timesθ[rand(1:length(timesθ),nₜ)]
    snapsₖ,row_idx = build_snapshots(FEMSpace,RBInfo,μₖ,timesθₖ,var)
    if k == 1
      snaps = snapsₖ
    else
      snaps = hcat(snaps,snapsₖ)
    end
  end
  return M_DEIM_POD(snaps,RBInfo.S.ϵₛ)...,row_idx
end

function functional_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Matrix{T},
  timesθ::Vector{T},
  var="A") where T

  phys_quadp,ncells,nquad_cell,nquad,V₀_quad = get_LagrangianQuad_info(FEMSpace)

  Θmat = Matrix{T}(undef,0,0)
  @simd for k = 1:RBInfo.S.nₛ_MDEIM
    println("Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)")
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    Param = get_ParamInfo(RBInfo, problem_id, μₖ)
    Θₖ = build_parameter_on_phys_quadp(Param,phys_quadp,ncells,nquad_cell,
      timesθ,var)
    Θₖ,_ = M_DEIM_POD(reshape(Θₖ,nquad,:), RBInfo.S.ϵₛ)
    if k == 1
      Θmat = Θₖ
    else
      Θmat = hcat(Θmat,Θₖ)
    end
  end
  Θmat,_ = M_DEIM_POD(Θmat,RBInfo.S.ϵₛ)
  Q = size(Θmat)[2]
  snaps,row_idx = Matrix{T}(undef,0,0),Int64[]
  @simd for q = 1:Q
    Θq = FEFunction(V₀_quad,Θmat[:,q])
    Matq = assemble_parametric_FE_structure(FEMSpace,Θq,var)
    i,v = findnz(Matq[:])
    if q == 1
      row_idx = i
      snaps = zeros(T,length(row_idx),Q)
    end
    snaps[:,q] = v
  end
  return M_DEIM_POD(snaps,RBInfo.S.ϵₛ)...,row_idx
end

function functional_MDEIM_sampling(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Matrix{T},
  timesθ::Vector{T},
  nₜ::Int64,
  var="A") where T

  phys_quadp,ncells,nquad_cell,nquad,V₀_quad = get_LagrangianQuad_info(FEMSpace)
  Θmat = Matrix{T}(undef,0,0)
  @simd for k = 1:RBInfo.S.nₛ_MDEIM
    println("Considering Parameter number $k/$(RBInfo.S.nₛ_MDEIM)")
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    Param = get_ParamInfo(RBInfo, problem_id, μₖ)
    timesθₖ = timesθ[rand(1:length(timesθ),nₜ)]
    Θₖ = build_parameter_on_phys_quadp(Param,phys_quadp,ncells,nquad_cell,
      timesθₖ,var)
    Θₖ = reshape(Θₖ,nquad,:)
    if k == 1
      Θmat = Θₖ
    else
      Θmat = hcat(Θmat, Θₖ)
    end
  end
  Θmat,_ = M_DEIM_POD(Θmat,RBInfo.S.ϵₛ)
  Q = size(Θmat)[2]
  snaps,row_idx = Matrix{T}(undef,0,0),Int64[]
  @simd for q = 1:Q
    Θq = FEFunction(V₀_quad,Θmat[:,q])
    Matq = assemble_parametric_FE_structure(FEMSpace,Θq,var)
    i,v = findnz(Matq[:])
    if q == 1
      row_idx = i
      snaps = zeros(T,length(row_idx),Q)
    end
    snaps[:,q] = v
  end
  return M_DEIM_POD(snaps,RBInfo.S.ϵₛ)...,row_idx
end

function spacetime_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Matrix{T},
  timesθ::Vector{T},
  var="A") where T

  snaps,row_idx = Matrix{T}(undef,0,0),Int64[]
  @simd for k = 1:RBInfo.S.nₛ_MDEIM
    println("Considering Parameter number $k/$(RBInfo.S.nₛ_MDEIM)")
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    snapsₖ,row_idx = build_snapshots(FEMSpace,RBInfo,μₖ,timesθ,var)
    snapsₖ,_ = M_DEIM_POD(snapsₖ, RBInfo.S.ϵₛ)
    if k == 1
      snaps = snapsₖ
    else
      snaps = hcat(snaps,snapsₖ)
    end
    snaps,_ = M_DEIM_POD(snaps,RBInfo.S.ϵₛ)
  end
  return M_DEIM_POD(snaps,RBInfo.S.ϵₛ)...,row_idx
end

function get_snaps_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Matrix{T},
  var="A") where T

  timesθ = get_timesθ(RBInfo)

  if RBInfo.space_time_M_DEIM
    return spacetime_MDEIM(FEMSpace,RBInfo,μ,timesθ,var)
  elseif RBInfo.functional_M_DEIM
    if RBInfo.sampling_M_DEIM
      nₜ = floor(Int,length(timesθ)*RBInfo.sampling_percentage)
      return functional_MDEIM_sampling(FEMSpace,RBInfo,μ,timesθ,nₜ,var)
    else
      return functional_MDEIM(FEMSpace,RBInfo,μ,timesθ,var)
    end
  else
    if RBInfo.sampling_M_DEIM
      nₜ = floor(Int,length(timesθ)*RBInfo.sampling_percentage)
      return standard_MDEIM_sampling(FEMSpace,RBInfo,μ,timesθ,nₜ,var)
    else
      return standard_MDEIM(FEMSpace,RBInfo,μ,timesθ,var)
    end
  end
end

function build_F_snapshots(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  μ::Matrix{T}) where T

  F = zeros(T, FEMSpace.Nₛᵘ, RBInfo.nₛ_DEIM)

  for i_nₛ = 1:RBInfo.nₛ_DEIM
    println("Snapshot number $i_nₛ, forcing")
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    Param = get_ParamInfo(RBInfo, problem_id, μ_i)
    F[:,i_nₛ] = assemble_forcing(FEMSpace, RBInfo, Param)[:]
  end
  F
end

function build_F_snapshots(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{T},
  timesθ::Vector{T}) where T

  Nₜ = length(timesθ)
  Param = get_ParamInfo(RBInfo, problem_id, μ)
  F_t = assemble_forcing(FEMSpace, RBInfo, Param)
  F = zeros(T, FEMSpace.Nₛᵘ, Nₜ)

  for i_t = 1:Nₜ
    F[:,i_t] = F_t(timesθ[i_t])[:]
  end
  F
end

function build_H_snapshots(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  μ::Matrix{T}) where T

  H = zeros(T, FEMSpace.Nₛᵘ, RBInfo.nₛ_DEIM)

  for i_nₛ = 1:RBInfo.nₛ_DEIM
    println("Snapshot number $i_nₛ, neumann datum")
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    Param = get_ParamInfo(RBInfo, problem_id, μ_i)
    H[:,i_nₛ] = assemble_neumann_datum(FEMSpace, RBInfo, Param)[:]
  end
  H
end

function build_H_snapshots(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{T},
  timesθ::Vector{T}) where T

  Nₜ = length(timesθ)
  Param = get_ParamInfo(RBInfo, problem_id, μ)
  H_t = assemble_neumann_datum(FEMSpace, RBInfo, Param)
  H = zeros(T, FEMSpace.Nₛᵘ, Nₜ)

  for i_t = 1:Nₜ
    H[:,i_t] = H_t(timesθ[i_t])[:]
  end
  H
end

function get_snaps_DEIM(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  μ::Matrix{T},
  var="F") where T

  snaps = build_snapshots(FEMSpace,RBInfo,μ,var)
  return M_DEIM_POD(snaps, RBInfo.ϵₛ)
end

function standard_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Matrix{T},
  timesθ::Vector{T},
  var="F") where T

  snaps,Σ = Float64[],Float64[]
  for k = 1:RBInfo.S.nₛ_DEIM
    println("Considering Parameter number $k/$(RBInfo.S.nₛ_MDEIM)")
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    snapsₖ = build_snapshots(FEMSpace,RBInfo,μₖ,timesθ,var)
    snapsₖ,_ = M_DEIM_POD(snapsₖ, RBInfo.S.ϵₛ)
    if k == 1
      snaps = snapsₖ
    else
      snaps,Σ = M_DEIM_POD(hcat(snaps,snapsₖ), RBInfo.S.ϵₛ)
    end
  end
  return snaps,Σ
end

function standard_DEIM_sampling(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Matrix{T},
  timesθ::Vector{T},
  nₜ::Int64,
  var="F") where T

  for k = 1:RBInfo.S.nₛ_DEIM
    println("Considering Parameter number $k/$(RBInfo.S.nₛ_MDEIM)")
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    timesθₖ = timesθ[rand(1:length(timesθ),nₜ)]
    snapsₖ = build_snapshots(FEMSpace,RBInfo,μₖ,timesθ,var)
    if k == 1
      global snaps = snapsₖ
    else
      global snaps = hcat(snaps, snapsₖ)
    end
  end
  return M_DEIM_POD(snaps,RBInfo.S.ϵₛ)
end

function functional_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Matrix{T},
  timesθ::Vector{T},
  var="F") where T

  phys_quadp,ncells,nquad_cell,nquad,V₀_quad = get_LagrangianQuad_info(FEMSpace)
  Θmat = Float64[]
  for k = 1:RBInfo.S.nₛ_DEIM
    println("Considering Parameter number $k/$(RBInfo.S.nₛ_MDEIM)")
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    Param = get_ParamInfo(RBInfo, problem_id, μₖ)
    Θₖ = build_parameter_on_phys_quadp(Param,phys_quadp,ncells,nquad_cell,
      timesθ,var)
    Θₖ,_ = M_DEIM_POD(reshape(Θₖ,nquad,:), RBInfo.S.ϵₛ)
    if k == 1
      Θmat = Θₖ
    else
      Θmat = hcat(Θmat,Θₖ)
    end
  end
  Θmat,_ = POD(Θmat,RBInfo.S.ϵₛ)
  Q = size(Θmat)[2]
  snaps = zeros(T,FEMSpace.Nₛᵘ,Q)
  for q = 1:Q
    Θq = FEFunction(V₀_quad,Θmat[:,q])
    snaps[:,q] = assemble_parametric_FE_structure(FEMSpace,Θq,var)
  end
  return M_DEIM_POD(snaps,RBInfo.S.ϵₛ)
end

function functional_DEIM_sampling(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Matrix{T},
  timesθ::Vector{T},
  nₜ::Int64,
  var="F") where T

  phys_quadp,ncells,nquad_cell,nquad,V₀_quad = get_LagrangianQuad_info(FEMSpace)
  for k = 1:RBInfo.S.nₛ_DEIM
    println("Considering Parameter number $k/$(RBInfo.S.nₛ_MDEIM)")
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    Param = get_ParamInfo(RBInfo, problem_id, μₖ)
    timesθₖ = timesθ[rand(1:length(timesθ),nₜ)]
    Θₖ = build_parameter_on_phys_quadp(Param,phys_quadp,ncells,nquad_cell,
      timesθₖ,var)
    Θₖ = reshape(Θₖ,nquad,:)
    if k == 1
      global Θmat = Θₖ
    else
      global Θmat = hcat(Θmat, Θₖ)
    end
  end
  Θmat,_ = M_DEIM_POD(Θmat, RBInfo.S.ϵₛ)
  Q = size(Θmat)[2]
  snaps = zeros(T,FEMSpace.Nₛᵘ,Q)
  for q = 1:Q
    Θq = FEFunction(V₀_quad,Θmat[:,q])
    snaps[:,q] = assemble_parametric_FE_structure(FEMSpace,Θq,var)
  end
  return M_DEIM_POD(snaps,RBInfo.S.ϵₛ)
end

function spacetime_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Matrix{T},
  timesθ::Vector{T},
  var="A") where T

  for k = 1:RBInfo.S.nₛ_MDEIM
    println("Considering Parameter number $k/$(RBInfo.S.nₛ_MDEIM)")
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    snapsₖ = build_snapshots(FEMSpace,RBInfo,μₖ,timesθ,var)
    compressed_snapsₖ,_ = M_DEIM_POD(snapsₖ, RBInfo.S.ϵₛ)
    if k == 1
      snaps = compressed_snapsₖ
    else
      snaps = hcat(snaps, compressed_snapsₖ)
    end
    snaps,_ = M_DEIM_POD(snaps, RBInfo.S.ϵₛ)
  end
  return M_DEIM_POD(snaps,RBInfo.S.ϵₛ)
end

function get_snaps_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Matrix{T},
  var="F") where T

  timesθ = get_timesθ(RBInfo)
  if RBInfo.space_time_M_DEIM
    return spacetime_DEIM(FEMSpace,RBInfo,μ,timesθ,var)
  elseif RBInfo.functional_M_DEIM
    if RBInfo.sampling_M_DEIM
      nₜ = floor(Int,length(timesθ)*RBInfo.sampling_percentage)
      return functional_DEIM_sampling(FEMSpace,RBInfo,μ,timesθ,nₜ,var)
    else
      return functional_DEIM(FEMSpace,RBInfo,μ,timesθ,var)
    end
  else
    if RBInfo.sampling_M_DEIM
      nₜ = floor(Int,length(timesθ)*RBInfo.sampling_percentage)
      return standard_DEIM_sampling(FEMSpace,RBInfo,μ,timesθ,nₜ,var)
    else
      return standard_DEIM(FEMSpace,RBInfo,μ,timesθ,var)
    end
  end
end

function build_snapshots(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  μₖ::Matrix{T},
  var::String) where T

  if var == "A"
    return build_A_snapshots(FEMSpace,RBInfo,μₖ)
  elseif var == "M"
    return build_M_snapshots(FEMSpace,RBInfo,μₖ)
  elseif var == "F"
    return build_F_snapshots(FEMSpace,RBInfo,μₖ)
  else var == "H"
    return build_H_snapshots(FEMSpace,RBInfo,μₖ)
  end

end

function build_snapshots(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μₖ::Vector{T},
  timesθ::Vector{T},
  var::String) where T

  if var == "A"
    return build_A_snapshots(FEMSpace,RBInfo,μₖ,timesθ)
  elseif var == "M"
    return build_M_snapshots(FEMSpace,RBInfo,μₖ,timesθ)
  elseif var == "F"
    return build_F_snapshots(FEMSpace,RBInfo,μₖ,timesθ)
  else var == "H"
    return build_H_snapshots(FEMSpace,RBInfo,μₖ,timesθ)
  end

end

function build_parameter_on_phys_quadp(
  Param::ParametricInfoUnsteady{D,T},
  phys_quadp,
  ncells::Int64,
  nquad_cell::Int64,
  timesθ::Vector{T},
  var::String) where {D,T}

  if var == "A"
    return [Param.α(phys_quadp[n][q],t_θ)
      for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
  elseif var == "M"
    return [Param.m(phys_quadp[n][q],t_θ)
      for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
  elseif var == "F"
    return [Param.f(phys_quadp[n][q],t_θ)
      for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
  elseif var == "H"
    return [Param.h(phys_quadp[n][q],t_θ)
      for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
  end
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
