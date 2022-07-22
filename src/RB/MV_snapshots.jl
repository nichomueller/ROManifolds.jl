function build_matrix_snapshots(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  μ::Vector{Vector{T}},
  var::String) where T

  Mat,row_idx = Matrix{T}(undef,0,0),Int[]
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

  Mat,row_idx = Matrix{T}(undef,0,0),Int[]
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

  snaps, row_idx = build_matrix_snapshots(FEMSpace, RBInfo, μ, var)
  snaps, _ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  snaps, row_idx

end

function get_LagrangianQuad_info(FEMSpace::Problem)

  ncells = length(FEMSpace.phys_quadp)::Int
  nquad_cell = length(FEMSpace.phys_quadp[1])::Int

  ncells,nquad_cell

end

function num_time_modes_M_DEIM(n::Int, nₛ_M_DEIM::Int, timesθ::Vector)
  min(n, ceil(Int, length(timesθ) / nₛ_M_DEIM))::Int
end

function standard_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var="A") where T

  (snaps_space, snaps_time, row_idx) =
    (Matrix{T}(undef,0,0), Matrix{T}(undef,0,0), Int[])
  @simd for k = 1:RBInfo.nₛ_MDEIM
    println("Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)")
    snapsₖ, row_idx = build_matrix_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
    snapsₖ_space, snapsₖ_time = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
    if k == 1
      snaps_space = snapsₖ_space
      snaps_time = snapsₖ_time
    else
      snaps_space, _ = M_DEIM_POD(hcat(snaps_space, snapsₖ_space), RBInfo.ϵₛ)
      snaps_time, _ = M_DEIM_POD(hcat(snaps_time, snapsₖ_time), RBInfo.ϵₜ)
    end
  end

  return snaps_space, snaps_time, row_idx

end

function functional_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var="A") where T

  ncells,nquad_cell = get_LagrangianQuad_info(FEMSpace)

  Θmat, Θmat_time = Matrix{Float}(undef,0,0), Matrix{Float}(undef,0,0)
  @simd for k = 1:RBInfo.nₛ_MDEIM
    println("Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)")
    Param = get_ParamInfo(RBInfo, μ[k])
    Θₖ = build_parameter_on_phys_quadp(Param,FEMSpace.phys_quadp,ncells,nquad_cell,
      timesθ,var)
    Θₖ, Θₖ_time = M_DEIM_POD(Θₖ, RBInfo.ϵₛ)
    if k == 1
      Θmat = Θₖ
      Θmat_time = Θₖ_time
    else
      Θmat = hcat(Θmat, Θₖ)
      Θmat_time = hcat(Θmat_time, Θₖ_time)
    end
  end

  Θmat, _ = M_DEIM_POD(Θmat,RBInfo.ϵₛ)

  Mat_Θ = assemble_parametric_FE_matrix(FEMSpace,var)
  Q = size(Θmat)[2]
  snaps,row_idx = Matrix{T}(undef,0,0),Int[]

  for q = 1:min(Q, Q_time)::Int
    Θq = FEFunction(FEMSpace.V₀_quad,Θmat[:,q])
    Matq = T.(Mat_Θ(Θq))::SparseMatrixCSC{T,Int}
    i,v = findnz(Matq[:])::Tuple{Vector{Int},Vector{T}}
    if q == 1
      row_idx = i
      snaps = zeros(T,length(row_idx),Q)
      snaps_time = zeros(T,length(row_idx),Q_time)
    end
    snaps[:,q] = v
  end
  snaps_space, _ = M_DEIM_POD(snaps,RBInfo.ϵₛ)
  snaps_time, _ = M_DEIM_POD(Θmat_time,RBInfo.ϵₜ)

  return snaps_space, snaps_time, row_idx

end

function get_snaps_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  μ::Vector{Vector{T}},
  var="A") where T

  timesθ = get_timesθ(RBInfo)

  if RBInfo.functional_M_DEIM
    return functional_MDEIM(FEMSpace,RBInfo,μ,timesθ,var)::Tuple{Matrix{T}, Matrix{T}, Vector{T}, Vector{Int}}
  else
    return standard_MDEIM(FEMSpace,RBInfo,μ,timesθ,var)::Tuple{Matrix{T}, Matrix{T}, Vector{T}, Vector{Int}}
  end

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
  snaps, _ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  return snaps

end

function standard_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var="F") where T

  snaps_space, snaps_time = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)
  for k = 1:RBInfo.nₛ_DEIM
    println("Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)")
    snapsₖ = build_vector_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
    snapsₖ_space, snapsₖ_time = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
    if k == 1
      snaps_space = snapsₖ_space
      snaps_time = snapsₖ_time
    else
      snaps_space, _ = M_DEIM_POD(hcat(snaps_space,snapsₖ_space), RBInfo.ϵₛ)
      snaps_time, _ = M_DEIM_POD(hcat(snaps_time,snapsₖ_time), RBInfo.ϵₜ)
    end
  end

  return snaps_space, snaps_time

end

function functional_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var="F") where T

  ncells,nquad_cell = get_LagrangianQuad_info(FEMSpace)

  Θmat, Θmat_time = Matrix{Float}(undef,0,0), Matrix{Float}(undef,0,0)
  @simd for k = 1:RBInfo.nₛ_MDEIM
    println("Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)")
    Param = get_ParamInfo(RBInfo, μ[k])
    Θₖ = build_parameter_on_phys_quadp(Param,FEMSpace.phys_quadp,ncells,nquad_cell,
      timesθ,var)
    Θₖ, Θₖ_time = M_DEIM_POD(Θₖ, RBInfo.ϵₛ)
    if k == 1
      Θmat = Θₖ
      Θmat_time = Θₖ_time
    else
      Θmat = hcat(Θmat, Θₖ)
      Θmat_time = hcat(Θmat_time, Θₖ_time)
    end
  end

  Θmat, _ = M_DEIM_POD(Θmat,RBInfo.ϵₛ)

  Vec_Θ = assemble_parametric_FE_vector(FEMSpace,var)
  Q = size(Θmat)[2]
  snaps = Matrix{T}(undef,0,0)

  for q = 1:Q
    Θq = FEFunction(FEMSpace.V₀_quad,Θmat[:,q])
    snaps[:,q] = T.(Vec_Θ(Θq))
  end
  snaps_space, _ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  snaps_time, _ = M_DEIM_POD(Θmat_time, RBInfo.ϵₜ)

  return snaps_space, snaps_time

end

function get_snaps_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  μ::Vector{Vector{T}},
  var="F") where T

  timesθ = get_timesθ(RBInfo)

  standard_DEIM(FEMSpace,RBInfo,μ,timesθ,var)::Tuple{Matrix{T}, Matrix{T}, Vector{T}}

end

function build_parameter_on_phys_quadp(
  Param::ParametricInfoUnsteady,
  phys_quadp::Vector{Vector{VectorValue{D,Float}}},
  ncells::Int,
  nquad_cell::Int,
  timesθ::Vector{T},
  var::String) where {D,T}

  if var == "A"
    Θ = [Param.α(phys_quadp[n][q],t_θ)
      for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
  elseif var == "M"
    Θ = [Param.m(phys_quadp[n][q],t_θ)
      for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
  elseif var == "F"
    Θ = [Param.f(phys_quadp[n][q],t_θ)
      for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
  elseif var == "H"
    Θ = [Param.h(phys_quadp[n][q],t_θ)
      for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
  else
    error("not implemented")
  end

  reshape(Θ, ncells*nquad_cell, :)::Matrix{Float}

end

function assemble_parametric_FE_matrix(
  FEMSpace::FEMSpacePoissonUnsteady,
  var::String)

  function Mat_θ(Θ)
    if var == "A"
      (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Θ*∇(FEMSpace.ϕᵤ(0.0))))*FEMSpace.dΩ,
        FEMSpace.V(0.0), FEMSpace.V₀))
    elseif var == "M"
      (assemble_matrix(∫(FEMSpace.ϕᵥ*(Θ*FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
        FEMSpace.V(0.0), FEMSpace.V₀))
    else
      error("Need to assemble an unrecognized FE structure")
    end
  end

  Mat_θ

end

function assemble_parametric_FE_vector(
  FEMSpace::FEMSpacePoissonUnsteady,
  var::String)

  function Vec_θ(Θ)
    if var == "F"
      assemble_vector(∫(FEMSpace.ϕᵥ*Θ)*FEMSpace.dΩ,FEMSpace.V₀)
    elseif var == "H"
      assemble_vector(∫(FEMSpace.ϕᵥ*Θ)*FEMSpace.dΓn,FEMSpace.V₀)
    else
      error("Need to assemble an unrecognized FE structure")
    end
  end

  Vec_θ

end

function assemble_parametric_FE_matrix(
  FEMSpace::FEMSpaceStokesUnsteady,
  var::String)

  function Mat_θ(Θ)
    if var == "A"
      (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Θ*∇(FEMSpace.ϕᵤ(0.0))))*FEMSpace.dΩ,
        FEMSpace.V(0.0), FEMSpace.V₀))
    elseif var == "M"
      (assemble_matrix(∫(FEMSpace.ϕᵥ⋅(Θ*FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
        FEMSpace.V(0.0), FEMSpace.V₀))
    else
      error("Need to assemble an unrecognized FE structure")
    end
  end

  Mat_θ

end

function assemble_parametric_FE_vector(
  FEMSpace::FEMSpaceStokesUnsteady,
  var::String)

  function Vec_θ(Θ)
    if var == "F"
      assemble_vector(∫(FEMSpace.ϕᵥ⋅Θ)*FEMSpace.dΩ,FEMSpace.V₀)
    elseif var == "H"
      assemble_vector(∫(FEMSpace.ϕᵥ⋅Θ)*FEMSpace.dΓn,FEMSpace.V₀)
    else
      error("Need to assemble an unrecognized FE structure")
    end
  end

  Vec_θ

end
