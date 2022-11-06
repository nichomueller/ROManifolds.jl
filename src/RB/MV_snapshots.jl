function Mat_snapshots(
  FEMSpace::FOMS{ID,D},
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  function Mat_linear(k::Int)::Tuple{Vector{Int},Vector{Float}}
    println("Snapshot number $k, $var")
    Mat = assemble_FEM_matrix(FEMSpace, RBInfo, μ[k], var)
    findnz(Mat[:])
  end

  function Mat_nonlinear(k::Int)::Tuple{Vector{Int},Vector{Float}}
    println("Snapshot number $k, $var")
    Mat = assemble_FEM_nonlinear_matrix(FEMSpace, RBInfo, μ[k],
      RBVars.Φₛ[1][:, k], var)
    findnz(Mat[:])
  end

  Mat(k) = isnonlinear(RBInfo, var) ? Mat_nonlinear(k) : Mat_linear(k)
  nₛ = isnonlinear(RBInfo, var) ? RBVars.nₛ[1] : RBInfo.nₛ_MDEIM

  i_v_block = Broadcasting(Mat)(1:nₛ)
  correct_structures(last.(i_v_block), first.(i_v_block))::Tuple{Matrix{Float}, Vector{Int}}

end

function Vec_snapshots(
  FEMSpace::FOMS{ID,D},
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  function Vec_linear(k::Int)::Vector{Float}
    println("Snapshot number $k, $var")
    assemble_FEM_vector(FEMSpace, RBInfo, μ[k], var)
  end

  function Vec_nonlinear(k::Int)::Vector{Float}
    println("Snapshot number $k, $var")
    assemble_FEM_nonlinear_vector(FEMSpace, RBInfo, μ[k],
      RBVars.Φₛ[1][:, k], var)
  end

  Vec(k) = isnonlinear(RBInfo, var) ? Vec_nonlinear(k) : Vec_linear(k)
  nₛ = isnonlinear(RBInfo, var) ? RBVars.nₛ[1] : RBInfo.nₛ_MDEIM

  blocks_to_matrix(Broadcasting(Vec)(1:nₛ))::Matrix{T}

end

function assemble_Mat_snapshots(
  FEMSpace::FOMS{ID,D},
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  snaps, row_idx = Mat_snapshots(FEMSpace, RBInfo, RBVars, μ, var)
  snaps_POD = MDEIM_POD(snaps, RBInfo.ϵₛ)
  snaps_POD, row_idx

end

function assemble_Vec_snapshots(
  FEMSpace::FOMS{ID,D},
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  snaps = Vec_snapshots(FEMSpace, RBInfo, RBVars, μ, var)
  snaps_POD = MDEIM_POD(snaps, RBInfo.ϵₛ)
  snaps_POD

end

function Mat_snapshots(
  FEMSpace::FOMST{ID,D},
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  timesθ = get_timesθ(RBInfo)

  function Mat_linear(k::Int)::Tuple{Vector{Int},Matrix{Float}}
    println("Snapshot number $k, $var")
    Mats = assemble_FEM_matrix(FEMSpace, RBInfo, μ[k], var, timesθ)
    iv = Broadcasting(Mat -> findnz(Mat[:]))(Mats)
    i, v = first.(iv), last.(iv)
    @assert all([length(i[1]) == length(i[j]) for j = eachindex(i)])
    i[1], blocks_to_matrix(v)
  end

  function Mat_nonlinear(k::Int)::Tuple{Vector{Int},Matrix{Float}}
    println("Snapshot number $k, $var")
    Φₛ_fun = FEFunction(FEMSpace.V₀[1], RBVars.Φₛ[1][:, k])
    Mats = assemble_FEM_nonlinear_matrix(FEMSpace, RBInfo, μ[k], var, timesθ)(Φₛ_fun)
    i, v = Broadcasting(Mat -> findnz(Mat[:]))(Mats)
    first.(i), first(v)
  end

  Mat(k) = isnonlinear(RBInfo, var) ? Mat_nonlinear(k) : Mat_linear(k)
  Mat

end

function Vec_snapshots(
  FEMSpace::FOMST{ID,D},
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  timesθ = get_timesθ(RBInfo)

  function Vec_linear(k::Int)::Matrix{Float}
    println("Snapshot number $k, $var")
    Vec_block = assemble_FEM_vector(FEMSpace, RBInfo, μ[k], var, timesθ)
    blocks_to_matrix(Vec_block)
  end

  function Vec_nonlinear(k::Int)::Matrix{Float}
    println("Snapshot number $k, $var")
    Vec_block = assemble_FEM_nonlinear_vector(FEMSpace, RBInfo, μ[k],
      RBVars.Φₛ[1][:, k], var)#, timesθ)
    blocks_to_matrix(Vec_block)
  end

  Vec(k) = isnonlinear(RBInfo, var) ? Vec_nonlinear(k) : Vec_linear(k)
  Vec

end

function assemble_Mat_snapshots(
  FEMSpace::FOMST{ID,D},
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  if RBInfo.functional_MDEIM
    functional_MDEIM(FEMSpace, RBInfo, RBVars, μ, var)::Tuple{Matrix{T}, Matrix{T}, Vector{Int}}
  else
    standard_MMDEIM(FEMSpace, RBInfo, RBVars, μ, var)::Tuple{Matrix{T}, Matrix{T}, Vector{Int}}
  end

end

function assemble_Vec_snapshots(
  FEMSpace::FOMST{ID,D},
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  standard_VMDEIM(FEMSpace, RBInfo, RBVars, μ, var)::Tuple{Matrix{T}, Matrix{T}}

end

function standard_MMDEIM(
  FEMSpace::FOMST{ID,D},
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  nₛ = isnonlinear(RBInfo, var) ? RBVars.nₛ[1] : RBInfo.nₛ_MDEIM
  Mat = Mat_snapshots(FEMSpace, RBInfo, RBVars, μ, var)

  ivals = Broadcasting(Mat)(1:nₛ)
  row_idx = first.(ivals)[1]
  vals = last.(ivals)
  vals_space = blocks_to_matrix(vals)
  vals_time = mode₂_unfolding(vals_space, nₛ)

  snaps_space = MDEIM_POD(vals_space, RBInfo.ϵₛ)
  snaps_time = MDEIM_POD(vals_time, RBInfo.ϵₜ*1e-2)

  snaps_space, snaps_time, row_idx

end

function standard_VMDEIM(
  FEMSpace::FOMST{ID,D},
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  nₛ = isnonlinear(RBInfo, var) ? RBVars.nₛ[1] : RBInfo.nₛ_MDEIM
  Vec = Vec_snapshots(FEMSpace, RBInfo, RBVars, μ, var)

  vals = Broadcasting(Vec)(1:nₛ)
  vals_space = blocks_to_matrix(vals)
  vals_time = mode₂_unfolding(vals_space, nₛ)

  snaps_space = MDEIM_POD(vals_space, RBInfo.ϵₛ)
  snaps_time = MDEIM_POD(vals_time, RBInfo.ϵₜ*1e-2)

  snaps_space, snaps_time

end

function functional_MDEIM(
  FEMSpace::FOMST{ID,D},
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  θ_space, θ_time = θ_snapshots(FEMSpace, RBInfo, RBVars, μ, var)

  #time
  snaps_time = MDEIM_POD(θ_time, RBInfo.ϵₜ*1e-2)

  # space
  θ_space, _ = MDEIM_POD(θ_space, RBInfo.ϵₛ)
  Paramθ = ParamInfo(FEMSpace, θ_space, var)
  Mats = assemble_FEM_matrix(FEMSpace, RBInfo, Paramθ)
  iv = Broadcasting(Mat -> findnz(Mat[:]))(Mats)
  i, v = first.(iv), last.(iv)
  @assert all([length(i[1]) == length(i[j]) for j = eachindex(i)])
  row_idx, vals_space = i[1], blocks_to_matrix(v)
  snaps_space = MDEIM_POD(vals_space, RBInfo.ϵₛ)

  snaps_space, snaps_time, row_idx

end

function θ_snapshots(
  FEMSpace::FOMST{ID,D},
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  timesθ = get_timesθ(RBInfo)
  nₛ = isnonlinear(RBInfo, var) ? RBVars.nₛ[1] : RBInfo.nₛ_MDEIM

  function θ_st_linear()
    Param = ParamInfo(RBInfo, μ[1:nₛ], var)
    θinfo = θ_phys_quadp(Param, FEMSpace.phys_quadp, timesθ)
    θblock_space, θblock_time = first.(θinfo), last.(θinfo)
    θ_space, θ_time = blocks_to_matrix(θblock_space), blocks_to_matrix(θblock_time)
    MDEIM_POD(θ_space, RBInfo.ϵₛ), MDEIM_POD(θ_time, RBInfo.ϵₜ*1e-2)
  end

  function θ_st_nonlinear()
    RBVars.Φₛ[1], RBVars.Φₜ[1]
  end

  isnonlinear(RBInfo, var) ? θ_st_nonlinear() : θ_st_linear()

end

function θ_phys_quadp_snapshot(
  Param::ParamInfoST,
  phys_quadp::Vector{Vector{VectorValue{D,Float}}},
  timesθ::Vector{T}) where {D,T}

  ncells = length(phys_quadp)
  nquad_cell = length(phys_quadp[1])

  θfun(tθ,n,q) = Param.fun(phys_quadp[n][q], tθ)
  θfun(tθ,n) = Broadcasting(q -> θfun(tθ,n,q))(1:nquad_cell)
  θfun(tθ) = blocks_to_matrix(Broadcasting(n -> θfun(tθ,n))(1:ncells))[:]

  θ_space = blocks_to_matrix(Broadcasting(θfun)(timesθ))
  θ_time = mode₂_unfolding(θ_space, nₛ)

  θ_space, θ_time

end

function θ_phys_quadp(
  Param::Vector{ParamInfoST},
  phys_quadp::Vector{Vector{VectorValue{D,Float}}},
  timesθ::Vector{T}) where {D,T}

  function loop_k(k::Int)
    println("Parametric snapshot number $k, $(Param[k].var)")
    θ_phys_quadp_snapshot(Param[k], phys_quadp, timesθ)
  end

  Broadcasting(loop_k)(eachindex(Param))

end

function correct_structures(
  Mat::Vector{Vector{Float}},
  row_idx::Vector{Vector{Int}})

  if all(Broadcasting(a->isequal(a, row_idx[1]))(row_idx))

    return blocks_to_matrix(Mat), row_idx[1]

  else

    println("Advanced version of findnz(⋅) is applied: correcting structures")
    row_idx_new = blocks_to_matrix(row_idx)
    sort!(unique!(row_idx_new))

    vec_new = zeros(Float, length(row_idx_new))
    function fix_ith_vector(i::Int)
      missing_idx_sparse = setdiff(row_idx_new, row_idx[i])
      missing_idx_full = from_sparse_idx_to_full_idx(missing_idx_sparse, row_idx_new)
      same_idx_full = setdiff(eachindex(row_idx_new), missing_idx_full)

      vec_new[same_idx_full] = Mat[i]
      vec_new
    end

    Mat_new = Broadcasting(fix_ith_vector)(eachindex(row_idx))

    return blocks_to_matrix(Mat_new), row_idx_new

  end

end
