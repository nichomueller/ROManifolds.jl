function build_matrix_snapshots(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  μ::Vector{Vector{T}},
  var::String) where T

  Mat,row_idx = Matrix{T}(undef,0,0),Int64[]
  for k = 1:RBInfo.nₛ_MDEIM
    println("Snapshot number $k, $var")
    Param = get_ParamInfo(RBInfo, μ[i_nₛ])
    Mat_k = assemble_FEM_structure(FEMSpace, RBInfo, Param, var)
    i, v = findnz(Mat_k[:])::Tuple{Vector{Int},Vector{T}}
    if i_nₛ == 1
      row_idx = i
      Mat = zeros(T,length(row_idx),RBInfo.nₛ_MDEIM)
    else
      Mat[:,i_nₛ] = v
    end
  end

  Mat,row_idx

end

function build_matrix_snapshots(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector,
  timesθ::Vector,
  var::String) where T

  Nₜ = length(timesθ)
  Param = get_ParamInfo(RBInfo, μ)
  Mat_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, var)

  Mat,row_idx = Matrix{T}(undef,0,0),Int64[]
  for i_t = 1:Nₜ
    Mat_i = Mat_t(timesθ[i_t])
    i, v = findnz(Mat_i[:])::Tuple{Vector{Int},Vector{T}}
    if i_t == 1
      row_idx = i
      Mat = zeros(T,length(row_idx),Nₜ)
    end
    Mat[:,i_t] = v
  end

  Mat,row_idx

end

function get_snaps_MDEIM(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  μ::Vector{Vector{T}},
  var="A") where T

  snaps,row_idx = build_matrix_snapshots(FEMSpace, RBInfo, μ, var)
  snaps, Σ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  snaps, Σ, row_idx

end

function get_LagrangianQuad_info(FEMSpace::Problem)

  ncells = length(FEMSpace.phys_quadp)::Int
  nquad_cell = length(FEMSpace.phys_quadp[1])::Int

  ncells,nquad_cell

end

function standard_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var="A") where T

  snaps,row_idx,Σ = Matrix{T}(undef,0,0),Int64[],Vector{T}(undef,0)
  @simd for k = 1:RBInfo.nₛ_MDEIM
    println("Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)")
    snapsₖ,row_idx = build_matrix_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
    snapsₖ,Σ = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
    if k == 1
      snaps = snapsₖ
    else
      snaps,Σ = M_DEIM_POD(hcat(snaps, snapsₖ), RBInfo.ϵₛ)
    end
  end
  return snaps,Σ,row_idx
end

function standard_MDEIM_sampling(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  nₜ::Int64,
  var="A") where T

  snaps,row_idx,Σ = Matrix{T}(undef,0,0),Int64[],Vector{T}(undef,0)
  @simd for k = 1:RBInfo.nₛ_MDEIM
    println("Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)")
    timesθₖ = timesθ[rand(1:length(timesθ),nₜ)]
    snapsₖ,row_idx = build_matrix_snapshots(FEMSpace,RBInfo,μ[k],timesθₖ,var)
    if k == 1
      snaps = snapsₖ
    else
      snaps = hcat(snaps,snapsₖ)
    end
  end
  snaps, Σ = M_DEIM_POD(snaps,RBInfo.ϵₛ)
  return snaps, Σ, row_idx
end

function functional_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var="A") where T

  ncells,nquad_cell = get_LagrangianQuad_info(FEMSpace)

  Θmat = Matrix{T}(undef,0,0)
  @simd for k = 1:RBInfo.nₛ_MDEIM
    println("Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)")
    Param = get_ParamInfo(RBInfo, μ[k])
    Θₖ = build_parameter_on_phys_quadp(Param,FEMSpace.phys_quadp,ncells,nquad_cell,
      timesθ,var)
    Θₖ,_ = M_DEIM_POD(reshape(Θₖ,ncells*nquad_cell,:), RBInfo.ϵₛ)
    if k == 1
      Θmat = Θₖ
    else
      Θmat = hcat(Θmat,Θₖ)
    end
  end
  Θmat,_ = M_DEIM_POD(Θmat,RBInfo.ϵₛ)
  Q = size(Θmat)[2]
  snaps,row_idx = Matrix{T}(undef,0,0),Int64[]
  @simd for q = 1:Q
    Θq = FEFunction(FEMSpace.V₀_quad,Θmat[:,q])
    Matq = assemble_parametric_FE_matrix(FEMSpace,Θq,var)
    i,v = findnz(Matq[:])::Tuple{Vector{Int},Vector{T}}
    if q == 1
      row_idx = i
      snaps = zeros(T,length(row_idx),Q)
    end
    snaps[:,q] = v
  end
  snaps, Σ = M_DEIM_POD(snaps,RBInfo.ϵₛ)
  return snaps, Σ, row_idx
end

function functional_MDEIM_sampling(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  nₜ::Int64,
  var="A") where T

  ncells,nquad_cell = get_LagrangianQuad_info(FEMSpace)
  Θmat = Matrix{T}(undef,0,0)
  @simd for k = 1:RBInfo.nₛ_MDEIM
    println("Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)")
    Param = get_ParamInfo(RBInfo, μ[k])
    timesθₖ = timesθ[rand(1:length(timesθ),nₜ)]
    Θₖ = build_parameter_on_phys_quadp(Param,FEMSpace.phys_quadp,ncells,nquad_cell,
      timesθₖ,var)
    Θₖ = reshape(Θₖ,ncells*nquad_cell,:)
    if k == 1
      Θmat = Θₖ
    else
      Θmat = hcat(Θmat, Θₖ)
    end
  end
  Θmat,_ = M_DEIM_POD(Θmat,RBInfo.ϵₛ)
  Q = size(Θmat)[2]
  snaps,row_idx = Matrix{T}(undef,0,0),Int64[]
  @simd for q = 1:Q
    Θq = FEFunction(FEMSpace.V₀_quad,Θmat[:,q])
    Matq = assemble_parametric_FE_matrix(FEMSpace,Θq,var)
    i,v = findnz(Matq[:])::Tuple{Vector{Int},Vector{T}}
    if q == 1
      row_idx = i
      snaps = zeros(T,length(row_idx),Q)
    end
    snaps[:,q] = v
  end
  snaps, Σ = M_DEIM_POD(snaps,RBInfo.ϵₛ)
  return snaps, Σ, row_idx
end

function spacetime_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var="A") where T

  snaps, row_idx, Σ = Matrix{T}(undef,0,0), Int64[], Vector{T}(undef,0)
  @simd for k = 1:RBInfo.nₛ_MDEIM
    println("Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)")
    snapsₖ,row_idx = build_matrix_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
    snapsₖ,_ = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
    if k == 1
      snaps = snapsₖ
    else
      snaps = hcat(snaps,snapsₖ)
    end
    snaps,_ = M_DEIM_POD(snaps,RBInfo.ϵₛ)
  end
  snaps, Σ = M_DEIM_POD(snaps,RBInfo.ϵₛ)
  return snaps, Σ, row_idx
end

function get_snaps_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  μ::Vector{Vector{T}},
  var="A") where T

  timesθ = get_timesθ(RBInfo)

  if RBInfo.space_time_M_DEIM
    snaps, Σ, row_idx = spacetime_MDEIM(FEMSpace,RBInfo,μ,timesθ,var)
  elseif RBInfo.functional_M_DEIM
    if RBInfo.sampling_M_DEIM
      nₜ = floor(Int,length(timesθ)*RBInfo.sampling_percentage)
      snaps, Σ, row_idx = functional_MDEIM_sampling(FEMSpace,RBInfo,μ,timesθ,nₜ,var)
    else
      snaps, Σ, row_idx = functional_MDEIM(FEMSpace,RBInfo,μ,timesθ,var)
    end
  else
    if RBInfo.sampling_M_DEIM
      nₜ = floor(Int,length(timesθ)*RBInfo.sampling_percentage)
      snaps, Σ, row_idx = standard_MDEIM_sampling(FEMSpace,RBInfo,μ,timesθ,nₜ,var)
    else
      snaps, Σ, row_idx = standard_MDEIM(FEMSpace,RBInfo,μ,timesθ,var)
    end
  end

  return snaps, Σ, row_idx

end

function build_vector_snapshots(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  μ::Vector{Vector{T}},
  var::String) where T

  Vec = zeros(T, FEMSpace.Nₛᵘ, RBInfo.nₛ_DEIM)

  for i_nₛ = 1:RBInfo.nₛ_DEIM
    println("Snapshot number $i_nₛ, $var")
    Param = get_ParamInfo(RBInfo, μ[i_nₛ])
    Vec[:,i_nₛ] = assemble_FEM_structure(FEMSpace, RBInfo, Param, var)[:]
  end

  Vec

end

function build_vector_snapshots(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector,
  timesθ::Vector,
  var::String) where T

  Nₜ = length(timesθ)
  Param = get_ParamInfo(RBInfo, μ)
  Vec_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, var)
  Vec = zeros(T, FEMSpace.Nₛᵘ, Nₜ)

  for i_t = 1:Nₜ
    Vec[:,i_t] = Vec_t(timesθ[i_t])[:]
  end

  Vec

end

function get_snaps_DEIM(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  μ::Vector{Vector{T}},
  var="F") where T

  snaps = build_vector_snapshots(FEMSpace,RBInfo,μ,var)
  snaps, Σ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  return snaps, Σ

end

function standard_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var="F") where T

  snaps, Σ = Matrix{T}(undef,0,0), Vector{T}(undef,0)
  for k = 1:RBInfo.nₛ_DEIM
    println("Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)")
    snapsₖ = build_vector_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
    snapsₖ,_ = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
    if k == 1
      snaps = snapsₖ
    else
      snaps,Σ = M_DEIM_POD(hcat(snaps,snapsₖ), RBInfo.ϵₛ)
    end
  end

  return snaps, Σ

end

function standard_DEIM_sampling(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  nₜ::Int64,
  var="F") where T

  for k = 1:RBInfo.nₛ_DEIM
    println("Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)")
    timesθₖ = timesθ[rand(1:length(timesθ),nₜ)]
    snapsₖ = build_vector_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
    if k == 1
      global snaps = snapsₖ
    else
      global snaps = hcat(snaps, snapsₖ)
    end
  end

  snaps, Σ = M_DEIM_POD(snaps,RBInfo.ϵₛ)
  return snaps, Σ

end

function functional_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var="F") where T

  ncells,nquad_cell = get_LagrangianQuad_info(FEMSpace)
  Θmat = Matrix{T}(undef,0,0)
  for k = 1:RBInfo.nₛ_DEIM
    println("Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)")
    Param = get_ParamInfo(RBInfo, μ[k])
    Θₖ = build_parameter_on_phys_quadp(Param,FEMSpace.phys_quadp,ncells,nquad_cell,
      timesθ,var)
    Θₖ,_ = M_DEIM_POD(reshape(Θₖ,ncells*nquad_cell,:), RBInfo.ϵₛ)
    if k == 1
      Θmat = Θₖ
    else
      Θmat = hcat(Θmat,Θₖ)
    end
  end
  Θmat,_ = POD(Θmat,RBInfo.ϵₛ)
  Q = size(Θmat)[2]
  snaps = zeros(T,FEMSpace.Nₛᵘ,Q)
  for q = 1:Q
    Θq = FEFunction(FEMSpace.V₀_quad,Θmat[:,q])
    snaps[:,q] = assemble_parametric_FE_vector(FEMSpace,Θq,var)
  end
  snaps, Σ = M_DEIM_POD(snaps,RBInfo.ϵₛ)
  return snaps, Σ
end

function functional_DEIM_sampling(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  nₜ::Int64,
  var="F") where T

  ncells,nquad_cell = get_LagrangianQuad_info(FEMSpace)
  Θmat = Matrix{T}(undef,0,0)
  for k = 1:RBInfo.nₛ_DEIM
    println("Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)")
    Param = get_ParamInfo(RBInfo, μ[k])
    timesθₖ = timesθ[rand(1:length(timesθ),nₜ)]
    Θₖ = build_parameter_on_phys_quadp(Param,FEMSpace.phys_quadp,ncells,nquad_cell,
      timesθₖ,var)
    Θₖ = reshape(Θₖ,ncells*nquad_cell,:)
    if k == 1
      global Θmat = Θₖ
    else
      global Θmat = hcat(Θmat, Θₖ)
    end
  end
  Θmat,_ = M_DEIM_POD(Θmat, RBInfo.ϵₛ)
  Q = size(Θmat)[2]
  snaps = zeros(T,FEMSpace.Nₛᵘ,Q)
  for q = 1:Q
    Θq = FEFunction(FEMSpace.V₀_quad,Θmat[:,q])
    snaps[:,q] = assemble_parametric_FE_vector(FEMSpace,Θq,var)
  end
  snaps, Σ = M_DEIM_POD(snaps,RBInfo.ϵₛ)
  return snaps, Σ
end

function spacetime_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var="A") where T

  for k = 1:RBInfo.nₛ_MDEIM
    println("Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)")
    snapsₖ = build_vector_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
    compressed_snapsₖ,_ = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
    if k == 1
      snaps = compressed_snapsₖ
    else
      snaps = hcat(snaps, compressed_snapsₖ)
    end
    snaps,_ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  end
  snaps, Σ = M_DEIM_POD(snaps,RBInfo.ϵₛ)
  return snaps, Σ
end

function get_snaps_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  μ::Vector{Vector{T}},
  var="F") where T

  timesθ = get_timesθ(RBInfo)

  if RBInfo.space_time_M_DEIM
    snaps, Σ = spacetime_DEIM(FEMSpace,RBInfo,μ,timesθ,var)
  elseif RBInfo.functional_M_DEIM
    if RBInfo.sampling_M_DEIM
      nₜ = floor(Int,length(timesθ)*RBInfo.sampling_percentage)
      snaps, Σ = functional_DEIM_sampling(FEMSpace,RBInfo,μ,timesθ,nₜ,var)
    else
      snaps, Σ = functional_DEIM(FEMSpace,RBInfo,μ,timesθ,var)
    end
  else
    if RBInfo.sampling_M_DEIM
      nₜ = floor(Int,length(timesθ)*RBInfo.sampling_percentage)
      snaps, Σ = standard_DEIM_sampling(FEMSpace,RBInfo,μ,timesθ,nₜ,var)
    else
      snaps, Σ = standard_DEIM(FEMSpace,RBInfo,μ,timesθ,var)
    end
  end

  return snaps, Σ

end

function build_parameter_on_phys_quadp(
  Param::ParametricInfoUnsteady{T},
  phys_quadp::LazyArray,
  ncells::Int64,
  nquad_cell::Int64,
  timesθ::Vector,
  var::String) where T

  if var == "A"
    return [Param.α(phys_quadp[n][q],t_θ)
      for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]::Vector{T}
  elseif var == "M"
    return [Param.m(phys_quadp[n][q],t_θ)
      for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]::Vector{T}
  elseif var == "F"
    return [Param.f(phys_quadp[n][q],t_θ)
      for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]::Vector{T}
  elseif var == "H"
    return [Param.h(phys_quadp[n][q],t_θ)
      for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]::Vector{T}
  else
    error("not implemented")
  end

end

function assemble_parametric_FE_matrix(
  FEMSpace::FEMSpacePoissonUnsteady{D,T},
  Θq::FEFunction,
  var::String) where {D,T}

  if var == "A"
    return (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Θq*∇(FEMSpace.ϕᵤ(0.0))))*FEMSpace.dΩ,
      FEMSpace.V(0.0), FEMSpace.V₀))::SparseMatrixCSC{T, Int64}
  elseif var == "M"
    return (assemble_matrix(∫(FEMSpace.ϕᵥ*(Θq*FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
      FEMSpace.V(0.0), FEMSpace.V₀))::SparseMatrixCSC{T, Int64}
  else
    error("Need to assemble an unrecognized FE structure")
  end

end

function assemble_parametric_FE_vector(
  FEMSpace::FEMSpacePoissonUnsteady{D,T},
  Θq::FEFunction,
  var::String) where {D,T}

  if var == "F"
    return assemble_vector(∫(FEMSpace.ϕᵥ*Θq)*FEMSpace.dΩ,FEMSpace.V₀)::Vector{T}
  elseif var == "H"
    return assemble_vector(∫(FEMSpace.ϕᵥ*Θq)*FEMSpace.dΓn,FEMSpace.V₀)::Vector{T}
  else
    error("Need to assemble an unrecognized FE structure")
  end

end

function assemble_parametric_FE_matrix(
  FEMSpace::FEMSpaceStokesUnsteady{D,T},
  Θq::FEFunction,
  var::String) where {D,T}

  if var == "A"
    return (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Θq*∇(FEMSpace.ϕᵤ(0.0))))*FEMSpace.dΩ,
      FEMSpace.V(0.0), FEMSpace.V₀))::SparseMatrixCSC{T, Int64}
  elseif var == "M"
    return (assemble_matrix(∫(FEMSpace.ϕᵥ⋅(Θq*FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
      FEMSpace.V(0.0), FEMSpace.V₀))::SparseMatrixCSC{T, Int64}
  else
    error("Need to assemble an unrecognized FE structure")
  end

end

function assemble_parametric_FE_vector(
  FEMSpace::FEMSpaceStokesUnsteady{D,T},
  Θq::FEFunction,
  var::String) where {D,T}

  if var == "F"
    return assemble_vector(∫(FEMSpace.ϕᵥ⋅Θq)*FEMSpace.dΩ,FEMSpace.V₀)::Vector{T}
  elseif var == "H"
    return assemble_vector(∫(FEMSpace.ϕᵥ⋅Θq)*FEMSpace.dΓn,FEMSpace.V₀)::Vector{T}
  else
    error("Need to assemble an unrecognized FE structure")
  end

end
