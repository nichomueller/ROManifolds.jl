function Mat_snapshots(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
  RBVars::RBProblemS,
  μ::Vector{Vector{T}},
  var::String) where T

  function Mat_linear(k::Int)
    println("Snapshot number $k, $var")
    Mat = assemble_FEM_structure(FEMSpace, RBInfo, μ[k], var)
    findnz(Mat[:])::Tuple{Vector{Int},Vector{T}}
  end

  function Mat_nonlinear(k::Int)
    println("Snapshot number $k, $var")
    Φₛ_fun = FEFunction(FEMSpace.V₀, RBVars.Φₛ[1][:, k])
    Mat = assemble_FEM_structure(FEMSpace, RBInfo, Φₛ_fun, var)
    findnz(Mat[:])::Tuple{Vector{Int},Vector{T}}
  end

  var ∈ ("C", "D") ? Mat(k) = Mat_nonlinear(k) : Mat(k) = Mat_linear(k)
  var ∈ ("C", "D") ? nₛ = RBVars.nₛᵘ : nₛ = RBInfo.nₛ_MDEIM

  Mat_block, row_idx_block = Broadcasting(Mat)(1:nₛ)
  blocks_to_matrix(Mat_block), row_idx_block[1]

end

function Vec_snapshots(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
  μ::Vector{Vector{T}},
  var::String) where T

  function Vec(k::Int)
    println("Snapshot number $k, $var")
    assemble_FEM_structure(FEMSpace, RBInfo, μ[k], var)::Vector{T}
  end

  Vec_block = Broadcasting(Vec)(1:RBInfo.nₛ_MDEIM)

  blocks_to_matrix(Vec_block)

end

function snaps_MDEIM(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  RBVars::RBProblemS,
  μ::Vector{Vector{T}},
  var::String) where T

  snaps, row_idx = M_snapshots(FEMSpace, RBInfo, RBVars, μ, var)
  snaps_POD, _ = MDEIM_POD(snaps, RBInfo.ϵₛ)
  snaps_POD, row_idx

end

function snaps_MDEIM(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  μ::Vector{Vector{T}},
  var::String) where T

  snaps = MV_snapshots(FEMSpace, RBInfo, μ, var)
  snaps_POD, _ = MDEIM_POD(snaps, RBInfo.ϵₛ)
  snaps_POD

end

function MV_snapshots(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  μ::Vector,
  timesθ::Vector,
  var::String) where T

  Nₜ = length(timesθ)
  Param = ParamInfo(RBInfo, μ, var)
  Mat_t = assemble_FEM_structure(FEMSpace, RBInfo, Param)

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

function MV_snapshots(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  RBVars::RBProblemST,
  μ::Vector,
  timesθ::Vector,
  var::String) where T

  error("Not implemented yet")

end

function call_MV_snapshots(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  RBVars::RBProblemST,
  μ::Vector,
  timesθ::Vector,
  var::String) where T

  if var ∈ ["C"]
    MV_snapshots(FEMSpace, RBInfo, RBVars, μ, timesθ, var)
  else
    MV_snapshots(FEMSpace, RBInfo, μ, timesθ, var)
  end

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
    snapsₖ, row_idx = call_MV_snapshots(
      FEMSpace,RBInfo,RBVars,μ[k],timesθ,var)
    snapsₖ_space, snapsₖ_time = MDEIM_POD(snapsₖ, RBInfo.ϵₛ)
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
      snapsₖ, row_idx = call_MV_snapshots(
        FEMSpace,RBInfo,RBVars,μ[k],timesθ,var)
      _, snapsₖ_time = MDEIM_POD(snapsₖ, RBInfo.ϵₛ)
      if k == 1
        snaps_time = snapsₖ_time
      else
        snaps_time = hcat(snaps_time, snapsₖ_time)
      end
    end
  else
    @simd for k = nₛ_min+1:nₛ_max
      println("Considering parameter number $k/$nₛ_max")
      snapsₖ, row_idx = call_MV_snapshots(
        FEMSpace,RBInfo,RBVars,μ[k],timesθ,var)
      snapsₖ_space, _ = MDEIM_POD(snapsₖ, RBInfo.ϵₛ)
      if k == 1
        snaps_space = snapsₖ_space
      else
        snaps_space = hcat(snaps_space, snapsₖ_space)
      end
    end
  end

  snaps_space, _ = MDEIM_POD(snaps_space, RBInfo.ϵₛ)
  snaps_time, _ = MDEIM_POD(snaps_time, RBInfo.ϵₜ)

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
  Θmat_space, _ = MDEIM_POD(Θmat_space,RBInfo.ϵₛ)
  Mat(Θ) = assemble_FEM_structure(FEMSpace, RBInfo, Θ, var)
  Q = size(Θmat_space)[2]
  snaps_space,row_idx = Matrix{T}(undef,0,0),Int[]

  for q = 1:Q
    Θq = FEFunction(FEMSpace.V₀_quad,Θmat_space[:,q])
    Matq = T.(Mat(Θq))::SparseMatrixCSC{T,Int}
    i,v = findnz(Matq[:])::Tuple{Vector{Int},Vector{T}}
    if q == 1
      row_idx = i
      snaps_space = zeros(T,length(row_idx),Q)
    end
    snaps_space[:,q] = v
  end

  snaps_space, _ = MDEIM_POD(snaps_space,RBInfo.ϵₛ)

  #time
  snaps_time, _ = MDEIM_POD(Θmat_time,RBInfo.ϵₜ)

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

  Param = ParamInfo(RBInfo, μ)
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
  snaps_space, _ = MDEIM_POD(snaps_space, RBInfo.ϵₛ)

  #time
  snaps_time = reshape(permutedims(Mat, (2,1,3)), RBVars.Nₜ, :)::Matrix{T}
  snaps_time, _ = MDEIM_POD(snaps_time, RBInfo.ϵₜ)

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

function snaps_MDEIM(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::RBProblemST,
  μ::Vector{Vector{T}},
  var::String) where T

  timesθ = get_timesθ(RBInfo)

  if RBInfo.functional_MDEIM
    return functional_MDEIM(FEMSpace,RBInfo,RBVars,μ,timesθ,var)::Tuple{Matrix{T}, Matrix{T}, Vector{Int}}
  else
    return standard_MDEIM(FEMSpace,RBInfo,RBVars,μ,timesθ,var)::Tuple{Matrix{T}, Matrix{T}, Vector{Int}}
  end

end

function MV_snapshots(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
  μ::Vector{Vector{T}},
  var::String) where T

  Vec = Matrix{T}(undef,0,0)

  for i_nₛ = 1:RBInfo.nₛ_MDEIM
    println("Snapshot number $i_nₛ, $var")
    Param = ParamInfo(RBInfo, μ[i_nₛ])
    v = assemble_FEM_structure(FEMSpace, RBInfo, Param, var)[:]
    if i_nₛ == 1
      Vec = zeros(T, length(v), RBInfo.nₛ_MDEIM)
    end
    Vec[:, i_nₛ] = v
  end

  Vec

end

function MV_snapshots(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  μ::Vector,
  timesθ::Vector,
  var::String) where T

  Nₜ = length(timesθ)
  Param = ParamInfo(RBInfo, μ)
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
    snapsₖ = MV_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
    snapsₖ_space, snapsₖ_time = MDEIM_POD(snapsₖ, RBInfo.ϵₛ)
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
      snapsₖ = MV_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
      _, snapsₖ_time = MDEIM_POD(snapsₖ, RBInfo.ϵₛ)
      if k == 1
        snaps_time = snapsₖ_time
      else
        snaps_time = hcat(snaps_time, snapsₖ_time)
      end
    end
  else
    @simd for k = nₛ_min+1:nₛ_max
      println("Considering parameter number $k/$nₛ_max")
      snapsₖ = MV_snapshots(FEMSpace,RBInfo,μ[k],timesθ,var)
      snapsₖ_space, _ = MDEIM_POD(snapsₖ, RBInfo.ϵₛ)
      if k == 1
        snaps_space = snapsₖ_space
      else
        snaps_space = hcat(snaps_space, snapsₖ_space)
      end
    end
  end

  snaps_space, _ = MDEIM_POD(snaps_space, RBInfo.ϵₛ)
  snaps_time, _ = MDEIM_POD(snaps_time, RBInfo.ϵₜ)

  return snaps_space, snaps_time

end

function snaps_DEIM(
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
    Param = ParamInfo(RBInfo, μ[k])
    Θₖ = assemble_parameter_on_phys_quadp(Param,FEMSpace.phys_quadp,ncells,nquad_cell,
      timesθ)
    Θₖ_space, Θₖ_time = MDEIM_POD(Θₖ, RBInfo.ϵₛ)
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
      Param = ParamInfo(RBInfo, μ[k])
      Θₖ = assemble_parameter_on_phys_quadp(Param,FEMSpace.phys_quadp,ncells,nquad_cell,
        timesθ)
      _, Θₖ_time = MDEIM_POD(Θₖ, RBInfo.ϵₛ)
      if k == 1
        Θmat_time = Θₖ_time
      else
        Θmat_time = hcat(Θmat_time, Θₖ_time)
      end
    end
  else
    @simd for k = nₛ_min+1:nₛ_max
      println("Considering parameter number $k/$nₛ_max")
      Param = ParamInfo(RBInfo, μ[k])
      Θₖ = assemble_parameter_on_phys_quadp(Param,FEMSpace.phys_quadp,ncells,nquad_cell,
        timesθ)
      Θₖ_space, _ = MDEIM_POD(Θₖ, RBInfo.ϵₛ)
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
  timesθ::Vector{T}) where {D,T}

  Θ = [Param.fun(phys_quadp[n][q],t_θ)
      for t_θ = timesθ for n = 1:ncells for q = 1:nquad_cell]

  reshape(Θ, ncells*nquad_cell, :)::Matrix{Float}

end
