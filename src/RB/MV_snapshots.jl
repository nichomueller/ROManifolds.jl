function build_matrix_snapshots(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  μ::Vector{Vector{T}},
  var::String) where T

  Mat, row_idx = Matrix{T}(undef,0,0), Int[]
  for k = 1:RBInfo.nₛ_MDEIM
    println("Snapshot number $k, $var")
    Param = get_ParamInfo(RBInfo, μ[k])
    Mat_k = assemble_FEM_structure(FEMSpace, RBInfo, Param, var)
    i, v = findnz(Mat_k[:])::Tuple{Vector{Int},Vector{T}}
    if k == 1
      row_idx = i
      Mat = zeros(T,length(row_idx),RBInfo.nₛ_MDEIM)
    else
      Mat[:,k] = v
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

  Mat, row_idx = Matrix{T}(undef,0,0), Int[]
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

function build_matrix_snapshots(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  RBVars::RBSteadyProblem,
  μ::Vector{Vector{T}},
  var::String) where T

  Matᵩ, row_idx = Matrix{T}(undef,0,0), Int[]
  for k = 1:RBInfo.nₛ_MDEIM
    Param = get_ParamInfo(RBInfo, μ[k])
    Matᵤ = assemble_FEM_structure(FEMSpace, RBInfo, Param, var)
    i, v = findnz(Matᵤ(RBVars.Sᵘ_quad[:, k])[:])::Tuple{Vector{Int},Vector{T}}
    if k == 1
      row_idx = i
      Matᵩ = zeros(T, length(row_idx), RBInfo.nₛ_MDEIM)
    end
    Matᵩ[:, k] = v
  end

  Matᵩ, row_idx

end

function build_matrix_snapshots(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  RBVars::RBUnsteadyProblem,
  μ::Vector,
  k::Int,
  timesθ::Vector,
  var::String) where T

  Param = get_ParamInfo(RBInfo, μ)
  Matᵤ = assemble_FEM_structure(FEMSpace, RBInfo, Param, var)

  Mat, row_idx = Array{T}(undef,0,0,0), Int[]
  for i_t = 1:RBVars.Nₜ
    # this is not precise: Sᵘ_quad is not computed at timesθ
    Mat_i = Matᵤ(RBVars.Sᵘ_quad[:, (k-1)*RBVars.Nₜ+i_t], timesθ[i_t])
    i, v = findnz(Mat_i[:])::Tuple{Vector{Int},Vector{T}}
    if i_t == 1
      row_idx = i
      Mat = zeros(T,length(row_idx),Nₜ)
    end
    Mat[:,i_t] = v
  end

  reshape(Mat, length(row_idx), :)::Matrix{T}, row_idx

end

function call_matrix_snapshots(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::RBSteadyProblem,
  μ::Vector{Vector{T}},
  var::String) where T

  if var ∈ ["C"]
    build_matrix_snapshots(FEMSpace, RBInfo, RBVars, μ, var)
  else
    build_matrix_snapshots(FEMSpace, RBInfo, μ, var)
  end

end

function call_matrix_snapshots(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  RBVars::RBUnsteadyProblem,
  μ::Vector{Vector{T}},
  k::Int,
  timesθ::Vector,
  var::String) where T

  if var ∈ ["C"]
    build_matrix_snapshots(FEMSpace, RBInfo, RBVars, μ[k], k, timesθ, var)
  else
    build_matrix_snapshots(FEMSpace, RBInfo, μ[k], timesθ, var)
  end

end

function get_snaps_MDEIM(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::RBSteadyProblem,
  μ::Vector{Vector{T}},
  var="A") where T

  snaps, row_idx = call_matrix_snapshots(FEMSpace, RBInfo, RBVars, μ, var)

  snaps, _ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  snaps, row_idx

end

function get_LagrangianQuad_info(FEMSpace::Problem)

  ncells = length(FEMSpace.phys_quadp)::Int
  nquad_cell = length(FEMSpace.phys_quadp[1])::Int

  ncells,nquad_cell

end

function standard_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  RBVars::RBUnsteadyProblem,
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var="A") where T

  (nₛ_min, nₛ_max) = sort([RBInfo.nₛ_MDEIM, RBInfo.nₛ_MDEIM_time])
  snaps_space, snaps_time = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)
  row_idx = Int[]

  @simd for k = 1:nₛ_min
    println("Considering parameter number $k/$nₛ_max")
    snapsₖ, row_idx = call_matrix_snapshots(
      FEMSpace,RBInfo,RBVars,μ[k],k,timesθ,var)
    snapsₖ_space, snapsₖ_time = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
    if k == 1
      snaps_space = snapsₖ_space
      snaps_time = snapsₖ_time
    else
      snaps_space = hcat(snaps_space, snapsₖ_space)
      snaps_time = hcat(snaps_time, snapsₖ_time)
    end
  end

  if nₛ_min == RBInfo.nₛ_MDEIM
    @simd for k = nₛ_min+1:nₛ_max
      println("Considering parameter number $k/$nₛ_max")
      snapsₖ, row_idx = call_matrix_snapshots(
        FEMSpace,RBInfo,RBVars,μ[k],k,timesθ,var)
      _, snapsₖ_time = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
      if k == 1
        snaps_time = snapsₖ_time
      else
        snaps_time = hcat(snaps_time, snapsₖ_time)
      end
    end
  else
    @simd for k = nₛ_min+1:nₛ_max
      println("Considering parameter number $k/$nₛ_max")
      snapsₖ, row_idx = call_matrix_snapshots(
        FEMSpace,RBInfo,RBVars,μ[k],k,timesθ,var)
      snapsₖ_space, _ = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
      if k == 1
        snaps_space = snapsₖ_space
      else
        snaps_space = hcat(snaps_space, snapsₖ_space)
      end
    end
  end

  snaps_space, _ = M_DEIM_POD(snaps_space, RBInfo.ϵₛ)
  snaps_time, _ = M_DEIM_POD(snaps_time, RBInfo.ϵₜ)

  return snaps_space, snaps_time, row_idx

end

function functional_MDEIM_linear(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var="A") where T

  Θmat_space, Θmat_time = build_θmat_snapshots(FEMSpace, RBInfo, μ, timesθ, var)

  # space
  Θmat_space, _ = M_DEIM_POD(Θmat_space,RBInfo.ϵₛ)
  Mat_Θ = assemble_parametric_FE_matrix(FEMSpace,var)
  Q = size(Θmat_space)[2]
  snaps_space,row_idx = Matrix{T}(undef,0,0),Int[]

  for q = 1:Q
    Θq = FEFunction(FEMSpace.V₀_quad,Θmat_space[:,q])
    Matq = T.(Mat_Θ(Θq))::SparseMatrixCSC{T,Int}
    i,v = findnz(Matq[:])::Tuple{Vector{Int},Vector{T}}
    if q == 1
      row_idx = i
      snaps_space = zeros(T,length(row_idx),Q)
    end
    snaps_space[:,q] = v
  end

  snaps_space, _ = M_DEIM_POD(snaps_space,RBInfo.ϵₛ)

  #time
  snaps_time, _ = M_DEIM_POD(Θmat_time,RBInfo.ϵₜ)

  return snaps_space, snaps_time, row_idx

end

function functional_MDEIM_nonlinear(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  RBVars::RBUnsteadyProblem,
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var="C") where T

  function index_mapping_inverse_quad(i::Int)
    iₛ = 1+Int(floor((i-1)/RBVars.nₜᵘ_quad))
    iₜ = i-(iₛ-1)*RBVars.nₜᵘ_quad
    iₛ, iₜ
  end

  Param = get_ParamInfo(RBInfo, μ)
  Matᵤ = assemble_FEM_structure(FEMSpace, RBInfo, Param, var)

  Mat = Array{T}(undef,0,0,0)
  for i_t = 1:RBVars.Nₜ
    for k = 1:RBVars.nᵘ_quad
      kₛ, kₜ = index_mapping_inverse(k)
      Φₛᵘ_fun = FEFunction(FEMSpace.V₀_quad,
        RBVars.Φₛᵘ_quad[:, kₛ] * RBVars.Φₜᵘ_quad[i_t, kₜ])
      # this is wrong: Φₛᵘ_fun is not at time timesθ[i_t]
      i, v = findnz(Matᵤ(Φₛᵘ_fun, timesθ[i_t])[:])::Tuple{Vector{Int},Vector{T}}
      if i_t == 1 && k == 1
        row_idx = i
        Mat = zeros(T, length(row_idx), RBVars.Nₜ, RBVars.nᵘ_quad)
      end
      Mat[:, i_t, k] = v
    end
  end

  #space
  snaps_space = reshape(Mat, length(row_idx), :)::Matrix{T}
  snaps_space, _ = M_DEIM_POD(snaps_space, RBInfo.ϵₛ)

  #time
  snaps_time = reshape(permutedims(Mat, (2,1,3)), RBVars.Nₜ, :)::Matrix{T}
  snaps_time, _ = M_DEIM_POD(snaps_time, RBInfo.ϵₜ)

  return snaps_space, snaps_time, row_idx

end

function call_functional_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  RBVars::RBUnsteadyProblem,
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var) where T

  if var ∈ ["C"]
    functional_MDEIM_nonlinear(FEMSpace, RBInfo, RBVars, μ, timesθ, var)
  else
    functional_MDEIM_linear(FEMSpace, RBInfo, μ, timesθ, var)
  end

end

function get_snaps_MDEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::RBUnsteadyProblem,
  μ::Vector{Vector{T}},
  var="A") where T

  timesθ = get_timesθ(RBInfo)

  if RBInfo.functional_M_DEIM
    return call_functional_MDEIM(FEMSpace,RBInfo,RBVars,μ,timesθ,var)::Tuple{Matrix{T}, Matrix{T}, Vector{Int}}
  else
    return standard_MDEIM(FEMSpace,RBInfo,μ,timesθ,var)::Tuple{Matrix{T}, Matrix{T}, Vector{Int}}
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

  (nₛ_min, nₛ_max) = sort([RBInfo.nₛ_DEIM, RBInfo.nₛ_DEIM_time])
  snaps_space, snaps_time = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)

  @simd for k = 1:nₛ_min
    println("Considering parameter number $k/$(RBInfo.nₛ_MDEIM)")
    snapsₖ = build_vector_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
    snapsₖ_space, snapsₖ_time = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
    if k == 1
      snaps_space = snapsₖ_space
      snaps_time = snapsₖ_time
    else
      snaps_space = hcat(snaps_space, snapsₖ_space)
      snaps_time = hcat(snaps_time, snapsₖ_time)
    end
  end

  if nₛ_min == RBInfo.nₛ_DEIM
    @simd for k = nₛ_min+1:nₛ_max
      println("Considering parameter number $k/$nₛ_max")
      snapsₖ = build_vector_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
      _, snapsₖ_time = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
      if k == 1
        snaps_time = snapsₖ_time
      else
        snaps_time = hcat(snaps_time, snapsₖ_time)
      end
    end
  else
    @simd for k = nₛ_min+1:nₛ_max
      println("Considering parameter number $k/$nₛ_max")
      snapsₖ = build_vector_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
      snapsₖ_space, _ = M_DEIM_POD(snapsₖ, RBInfo.ϵₛ)
      if k == 1
        snaps_space = snapsₖ_space
      else
        snaps_space = hcat(snaps_space, snapsₖ_space)
      end
    end
  end

  snaps_space, _ = M_DEIM_POD(snaps_space, RBInfo.ϵₛ)
  snaps_time, _ = M_DEIM_POD(snaps_time, RBInfo.ϵₜ)

  return snaps_space, snaps_time

end

function functional_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var="F") where T

  Θmat_space, Θmat_time = build_θmat_snapshots(FEMSpace, RBInfo, μ, timesθ, var)

  # space
  Θmat_space, _ = M_DEIM_POD(Θmat_space,RBInfo.ϵₛ)
  Vec_Θ = assemble_parametric_FE_vector(FEMSpace,var)
  Q = size(Θmat_space)[2]
  snaps_space = zeros(FEMSpace.Nₛᵘ, Q)

  for q = 1:Q
    Θq = FEFunction(FEMSpace.V₀_quad, Θmat_space[:,q])
    snaps_space[:,q] = T.(Vec_Θ(Θq))
  end

  snaps_space, _ = M_DEIM_POD(snaps_space,RBInfo.ϵₛ)

  #time
  snaps_time, _ = M_DEIM_POD(Θmat_time,RBInfo.ϵₜ)

  return snaps_space, snaps_time

end

function get_snaps_DEIM(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  μ::Vector{Vector{T}},
  var="F") where T

  timesθ = get_timesθ(RBInfo)
  return standard_DEIM(FEMSpace,RBInfo,μ,timesθ,var)::Tuple{Matrix{T}, Matrix{T}}

end

function build_θmat_snapshots(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var::String) where T

  if var ∈ ["f", "h"]
    nₛ, nₛ_time = RBInfo.nₛ_DEIM, RBInfo.nₛ_DEIM_time
  else
    nₛ, nₛ_time = RBInfo.nₛ_MDEIM, RBInfo.nₛ_MDEIM_time
  end

  (nₛ_min, nₛ_max) = sort([nₛ, nₛ_time])
  ncells,nquad_cell = get_LagrangianQuad_info(FEMSpace)
  Θmat_space, Θmat_time = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)

  @simd for k = 1:nₛ_min
    println("Considering parameter number $k/$nₛ_max")
    Param = get_ParamInfo(RBInfo, μ[k])
    Θₖ = build_parameter_on_phys_quadp(Param,FEMSpace.phys_quadp,ncells,nquad_cell,
      timesθ,var)
    Θₖ_space, Θₖ_time = M_DEIM_POD(Θₖ, RBInfo.ϵₛ)
    if k == 1
      Θmat_space = Θₖ_space
      Θmat_time = Θₖ_time
    else
      Θmat_space = hcat(Θmat_space, Θₖ_space)
      Θmat_time = hcat(Θmat_time, Θₖ_time)
    end
  end

  if nₛ_min == RBInfo.nₛ
    @simd for k = nₛ_min+1:nₛ_max
      println("Considering parameter number $k/$nₛ_max")
      Param = get_ParamInfo(RBInfo, μ[k])
      Θₖ = build_parameter_on_phys_quadp(Param,FEMSpace.phys_quadp,ncells,nquad_cell,
        timesθ,var)
      _, Θₖ_time = M_DEIM_POD(Θₖ, RBInfo.ϵₛ)
      if k == 1
        Θmat_time = Θₖ_time
      else
        Θmat_time = hcat(Θmat_time, Θₖ_time)
      end
    end
  else
    @simd for k = nₛ_min+1:nₛ_max
      println("Considering parameter number $k/$nₛ_max")
      Param = get_ParamInfo(RBInfo, μ[k])
      Θₖ = build_parameter_on_phys_quadp(Param,FEMSpace.phys_quadp,ncells,nquad_cell,
        timesθ,var)
      Θₖ_space, _ = M_DEIM_POD(Θₖ, RBInfo.ϵₛ)
      if k == 1
        Θmat_space = Θₖ_space
      else
        Θmat_space = hcat(Θmat_space, Θₖ_space)
      end
    end
  end

  Θmat_space, Θmat_time

end

function build_θmat_snapshots(RBVars::RBUnsteadyProblem)

  RBVars.Φₛᵘ_quad, RBVars.Φₜᵘ_quad

end

function call_θmat_snapshots(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  RBVars::RBUnsteadyProblem,
  μ::Vector,
  timesθ::Vector,
  var::String) where T

  if var ∈ ["C"]
    build_θmat_snapshots(RBVars)
  else
    build_θmat_snapshots(FEMSpace, RBInfo, μ, timesθ, var)
  end

end

function build_parameter_on_phys_quadp(
  Param::UnsteadyParametricInfo,
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
  elseif var == "B"
    Θ = [Param.b(phys_quadp[n][q],t_θ)
      for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]
  elseif var == "D"
    Θ = [Param.σ(phys_quadp[n][q],t_θ)
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
  FEMSpace::FEMSpaceADRUnsteady,
  var::String)

  function Mat_θ(Θ)
    if var == "A"
      (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Θ*∇(FEMSpace.ϕᵤ(0.0))))*FEMSpace.dΩ,
        FEMSpace.V(0.0), FEMSpace.V₀))
    elseif var == "M" || var == "D"
      (assemble_matrix(∫(FEMSpace.ϕᵥ*(Θ*FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
        FEMSpace.V(0.0), FEMSpace.V₀))
    elseif var == "B"
      (assemble_matrix(∫(FEMSpace.ϕᵥ * (Θ⋅∇(FEMSpace.ϕᵤ(0.0))))*FEMSpace.dΩ,
        FEMSpace.V(0.0), FEMSpace.V₀))
    else
      error("Need to assemble an unrecognized FE structure")
    end
  end

  Mat_θ

end

function assemble_parametric_FE_vector(
  FEMSpace::FEMSpaceADRUnsteady,
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

function assemble_parametric_FE_matrix(
  FEMSpace::FEMSpaceNavierStokesUnsteady,
  var::String)

  function Mat_θ(Θ)
    if var == "A"
      (assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Θ*∇(FEMSpace.ϕᵤ(0.0))))*FEMSpace.dΩ,
        FEMSpace.V(0.0), FEMSpace.V₀))
    elseif var == "M"
      (assemble_matrix(∫(FEMSpace.ϕᵥ⋅(Θ*FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
        FEMSpace.V(0.0), FEMSpace.V₀))
    elseif var == "C"
      (assemble_matrix(∫(FEMSpace.ϕᵥ⋅(Θ*FEMSpace.ϕᵤ(0.0)))*FEMSpace.dΩ,
        FEMSpace.V(0.0), FEMSpace.V₀))
    else
      error("Need to assemble an unrecognized FE structure")
    end
  end

  Mat_θ

end

function assemble_parametric_FE_vector(
  FEMSpace::FEMSpaceNavierStokesUnsteady,
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
