function assemble_matrix_snapshots(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
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
    end
    Mat[:,k] = v
  end

  Mat,row_idx

end

function assemble_matrix_snapshots(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
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

function assemble_matrix_snapshots(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
  RBVars::RBProblemS,
  μ::Vector{Vector{T}},
  var::String) where T

  Matᵩ, row_idx = Matrix{T}(undef,0,0), Int[]
  Param = get_ParamInfo(RBInfo, μ[1])
  Matᵤ = assemble_FEM_structure(FEMSpace, RBInfo, Param, var)

  for nᵩ = 1:RBVars.nₛᵘ
    println("Snapshot number $nᵩ, $var")
    Φₛ_fun = FEFunction(FEMSpace.V₀, RBVars.Φₛ[:, nᵩ])
    i, v = findnz(Matᵤ(Φₛ_fun)[:])::Tuple{Vector{Int},Vector{T}}
    if nᵩ == 1
      row_idx = i
      Matᵩ = zeros(T, length(row_idx), RBVars.nₛᵘ)
    end
    Matᵩ[:, nᵩ] = v
  end

  Matᵩ, row_idx

end

function assemble_matrix_snapshots(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  RBVars::RBProblemST,
  μ::Vector,
  timesθ::Vector,
  var::String) where T

  error("Not implemented yet")

end

function call_matrix_snapshots(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  RBVars::RBProblemST,
  μ::Vector,
  timesθ::Vector,
  var::String) where T

  if var ∈ ["C"]
    assemble_matrix_snapshots(FEMSpace, RBInfo, RBVars, μ, timesθ, var)
  else
    assemble_matrix_snapshots(FEMSpace, RBInfo, μ, timesθ, var)
  end

end

function get_snaps_MDEIM(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  RBVars::RBProblemS,
  μ::Vector{Vector{T}},
  var::String) where T

  if var ∈ ("C", "D")
    snaps, row_idx = assemble_matrix_snapshots(FEMSpace, RBInfo, RBVars, μ, var)
  else
    snaps, row_idx = assemble_matrix_snapshots(FEMSpace, RBInfo, μ, var)
  end

  snaps, _ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  snaps, row_idx

end

function get_LagrangianQuad_info(FEMSpace::Problem)

  ncells = length(FEMSpace.phys_quadp)::Int
  nquad_cell = length(FEMSpace.phys_quadp[1])::Int

  ncells,nquad_cell

end

function standard_MDEIM(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  RBVars::RBProblemST,
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var::String) where T

  (nₛ_min, nₛ_max) = sort([RBInfo.nₛ_MDEIM, RBInfo.nₛ_MDEIM_time])
  snaps_space, snaps_time = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)
  row_idx = Int[]

  @simd for k = 1:nₛ_min
    println("Considering parameter number $k/$nₛ_max")
    snapsₖ, row_idx = call_matrix_snapshots(
      FEMSpace,RBInfo,RBVars,μ[k],timesθ,var)
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
        FEMSpace,RBInfo,RBVars,μ[k],timesθ,var)
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
        FEMSpace,RBInfo,RBVars,μ[k],timesθ,var)
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
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var::String) where T

  Θmat_space, Θmat_time = assemble_θmat_snapshots(FEMSpace, RBInfo, μ, timesθ, var)

  # space
  Θmat_space, _ = M_DEIM_POD(Θmat_space,RBInfo.ϵₛ)
  Mat_Θ = assemble_parametric_FE_matrix(RBInfo.FEMInfo.problem_id,FEMSpace,var)
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
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  RBVars::RBProblemST,
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var::String) where T

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
      Φₛ_fun = FEFunction(FEMSpace.V₀_quad,
        RBVars.Φₛ_quad[:, kₛ] * RBVars.Φₜᵘ_quad[i_t, kₜ])
      # this is wrong: Φₛ_fun is not at time timesθ[i_t]
      i, v = findnz(Matᵤ(Φₛ_fun, timesθ[i_t])[:])::Tuple{Vector{Int},Vector{T}}
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

function functional_MDEIM(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  RBVars::RBProblemST,
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
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::RBProblemST,
  μ::Vector{Vector{T}},
  var::String) where T

  timesθ = get_timesθ(RBInfo)

  if RBInfo.functional_M_DEIM
    return functional_MDEIM(FEMSpace,RBInfo,RBVars,μ,timesθ,var)::Tuple{Matrix{T}, Matrix{T}, Vector{Int}}
  else
    return standard_MDEIM(FEMSpace,RBInfo,RBVars,μ,timesθ,var)::Tuple{Matrix{T}, Matrix{T}, Vector{Int}}
  end

end

function assemble_vector_snapshots(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
  μ::Vector{Vector{T}},
  var::String) where T

  Vec = Matrix{T}(undef,0,0)

  for i_nₛ = 1:RBInfo.nₛ_MDEIM
    println("Snapshot number $i_nₛ, $var")
    Param = get_ParamInfo(RBInfo, μ[i_nₛ])
    v = assemble_FEM_structure(FEMSpace, RBInfo, Param, var)[:]
    if i_nₛ == 1
      Vec = zeros(T, length(v), RBInfo.nₛ_MDEIM)
    end
    Vec[:, i_nₛ] = v
  end

  Vec

end

function assemble_vector_snapshots(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  μ::Vector,
  timesθ::Vector,
  var::String) where T

  Nₜ = length(timesθ)
  Param = get_ParamInfo(RBInfo, μ)
  Vec_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, var)
  Vec = Matrix{T}(undef,0,0)

  for i_t = 1:Nₜ
    v = Vec_t(timesθ[i_t])[:]
    if i_t == 1
      Vec = zeros(T, length(v), Nₜ)
    end
    Vec[:, i_t] = v
  end

  Vec

end

function get_snaps_DEIM(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  μ::Vector{Vector{T}},
  var::String) where T

  snaps = assemble_vector_snapshots(FEMSpace,RBInfo,μ,var)
  snaps, _ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  return snaps

end

function standard_DEIM(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var::String) where T

  (nₛ_min, nₛ_max) = sort([RBInfo.nₛ_MDEIM, RBInfo.nₛ_MDEIM_time])
  snaps_space, snaps_time = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)

  @simd for k = 1:nₛ_min
    println("Considering parameter number $k/$(RBInfo.nₛ_MDEIM)")
    snapsₖ = assemble_vector_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
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
      snapsₖ = assemble_vector_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
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
      snapsₖ = assemble_vector_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
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

function get_snaps_DEIM(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  μ::Vector{Vector{T}},
  var::String) where T

  timesθ = get_timesθ(RBInfo)
  return standard_DEIM(FEMSpace,RBInfo,μ,timesθ,var)::Tuple{Matrix{T}, Matrix{T}}

end

function assemble_θmat_snapshots(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  μ::Vector{Vector{T}},
  timesθ::Vector,
  var::String) where T

  nₛ, nₛ_time = RBInfo.nₛ_MDEIM, RBInfo.nₛ_MDEIM_time

  (nₛ_min, nₛ_max) = sort([nₛ, nₛ_time])
  ncells,nquad_cell = get_LagrangianQuad_info(FEMSpace)
  Θmat_space, Θmat_time = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)

  @simd for k = 1:nₛ_min
    println("Considering parameter number $k/$nₛ_max")
    Param = get_ParamInfo(RBInfo, μ[k])
    Θₖ = assemble_parameter_on_phys_quadp(Param,FEMSpace.phys_quadp,ncells,nquad_cell,
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

  if nₛ_min == nₛ
    @simd for k = nₛ_min+1:nₛ_max
      println("Considering parameter number $k/$nₛ_max")
      Param = get_ParamInfo(RBInfo, μ[k])
      Θₖ = assemble_parameter_on_phys_quadp(Param,FEMSpace.phys_quadp,ncells,nquad_cell,
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
      Θₖ = assemble_parameter_on_phys_quadp(Param,FEMSpace.phys_quadp,ncells,nquad_cell,
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

function assemble_θmat_snapshots(RBVars::RBProblemST)

  RBVars.Φₛ_quad, RBVars.Φₜᵘ_quad

end

function call_θmat_snapshots(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  RBVars::RBProblemST,
  μ::Vector,
  timesθ::Vector,
  var::String) where T

  if var ∈ ["C"]
    assemble_θmat_snapshots(RBVars)
  else
    assemble_θmat_snapshots(FEMSpace, RBInfo, μ, timesθ, var)
  end

end

function assemble_parameter_on_phys_quadp(
  Param::ParamInfoST,
  phys_quadp::Vector{Vector{VectorValue{D,Float}}},
  ncells::Int,
  nquad_cell::Int,
  timesθ::Vector{T},
  var::String) where {D,T}

  if var == "A"
    Θfun = Param.α
  elseif var == "M"
    Θfun = Param.m
  elseif var == "B"
    Θfun = Param.b
  elseif var == "D"
    Θfun = Param.σ
  else
    error("not implemented")
  end

  Θ = [Θfun(phys_quadp[n][q],t_θ)
      for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]

  reshape(Θ, ncells*nquad_cell, :)::Matrix{Float}

end

function assemble_parametric_FE_matrix(
  ::NTuple{1,Int},
  FEMSpace::FEMProblemST,
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

function assemble_parametric_FE_matrix(
  ::NTuple{2,Int},
  FEMSpace::FEMProblemST,
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

function assemble_parametric_FE_matrix(
  ::NTuple{3,Int},
  FEMSpace::FEMProblemST,
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

function correct_structures(
  Matᵩ::Vector{Vector{T}},
  row_idx::Vector{Vector{Int}}) where T

  idx = 1
  for i = 1 .+ eachindex(row_idx[2:end])
    if length(row_idx[i]) > length(row_idx[idx])
      idx = i
    end
  end

  row_idx_new = row_idx[idx]
  Matᵩ_new = zeros(T, length(row_idx_new), length(Matᵩ))
  Matᵩ_new[:, idx] = Matᵩ[idx]

  for i = setdiff(eachindex(row_idx), idx)
    missing_idx_sparse = setdiff(row_idx_new, row_idx[i])
    missing_idx_full = zeros(Int, length(missing_idx_sparse))
    for j = eachindex(missing_idx_full)
      missing_idx_full[j] = findall(x -> x == missing_idx_sparse[j], row_idx_new)[1]
    end
    same_idx_full = setdiff(eachindex(row_idx_new), missing_idx_full)

    Matᵩ_new[same_idx_full, i] = Matᵩ[i]
  end

  Matᵩ_new, row_idx_new

end
