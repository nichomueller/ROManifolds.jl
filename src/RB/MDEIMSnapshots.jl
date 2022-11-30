basis_as_fefun(::RBLinOperator) = error("Not implemented")

function basis_as_fefun(
  op::RBBilinOperator{OT,ParamTrialFESpace,RBSpaceSteady}) where OT

  bspace = get_basis_space(op)
  ns = get_ns(bspace)
  trial = get_trial(op)
  fefuns(k::Int) = FEFunction(trial,bspace[:,k])
  eval.(fefuns,1:ns)
end

function basis_as_fefun(
  op::RBBilinOperator{OT,ParamTransientTrialFESpace,RBSpaceUnsteady}) where OT

  bspace = get_basis_space(op)
  ns = get_ns(bspace)
  trial = get_trial(op)
  fefuns(t::Real,k::Int) = FEFunction(trial(t),bspace[:,k])
  fefuns(t::Real) = k->fefuns(t,k)
  t -> eval.(fefuns(t),1:ns)
end

mdeim_snapshots(op::RBLinOperator,args...) = vector_snapshots(op,args...)
mdeim_snapshots(op::RBLinOperator,args...) = matrix_snapshots(op,args...)

function vector_snapshots(
  op::RBLinOperator{Nonaffine,RBSpaceSteady},
  μ::Snapshots)

  id = get_id(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    assemble_vector(op)(μ[k])
  end

  values = blocks_to_matrix(snapshot.(eachindex(μ)))
  Snapshots(id,values)
end

function vector_snapshots(
  op::RBLinOperator{Nonlinear,RBSpaceSteady},
  ::Snapshots)

  id = get_id(op)
  bfun = basis_as_fefun(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    assemble_vector(op)(bfun[k])
  end

  values = blocks_to_matrix(snapshot.(eachindex(bfun)))
  Snapshots(id,values)
end

function vector_snapshots(
  op::RBLinOperator{Nonaffine,RBSpaceUnsteady},
  μ::Snapshots)

  id = get_id(op)
  timesθ = get_timesθ(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    assemble_vector(op)(μ[k],timesθ)
  end

  values = blocks_to_matrix(snapshot.(eachindex(μ)))
  Snapshots(id,values)
end

function vector_snapshots(
  op::RBLinOperator{Nonlinear,RBSpaceUnsteady},
  ::Snapshots)

  id = get_id(op)
  bfun = basis_as_fefun(op)
  timesθ = get_timesθ(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    assemble_vector(op)(bfun[k],timesθ)
  end

  values = blocks_to_matrix(snapshot.(eachindex(bfun)))
  Snapshots(id,values)
end

function matrix_snapshots(
  op::RBBilinOperator{Nonaffine,ParamTrialFESpace,RBSpaceSteady},
  μ::Snapshots)

  id = get_id(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    M = assemble_matrix(op)(μ[k])
    i,v = findnz(M[:])
    i,v
  end

  iv = blocks_to_matrix(snapshot.(eachindex(μ)))
  row_idx,values = first.(iv),last.(iv)
  check_row_idx(row_idx)
  Snapshots(id,values)
end

function matrix_snapshots(
  op::RBLinOperator{Nonlinear,ParamTrialFESpace,RBSpaceSteady},
  ::Snapshots)

  id = get_id(op)
  bfun = basis_as_fefun(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    M = assemble_matrix(op)(bfun[k])
    i,v = findnz(M[:])
    i,v
  end

  iv = blocks_to_matrix(snapshot.(eachindex(bfun)))
  row_idx,values = first.(iv),last.(iv)
  check_row_idx(row_idx)
  Snapshots(id,values)
end

function matrix_snapshots(
  op::RBBilinOperator{Nonaffine,ParamTrialFESpace,RBSpaceUnsteady},
  μ::Snapshots)

  id = get_id(op)
  timesθ = get_timesθ(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    M = assemble_matrix(op)(μ[k],timesθ)
    i,v = findnz(M[:])
    i,v
  end

  iv = blocks_to_matrix(snapshot.(eachindex(μ)))
  row_idx,values = first.(iv),last.(iv)
  check_row_idx(row_idx)
  Snapshots(id,values)
end

function matrix_snapshots(
  op::RBLinOperator{Nonlinear,ParamTransientTrialFESpace,RBSpaceUnsteady},
  ::Snapshots)

  id = get_id(op)
  bfun = basis_as_fefun(op)
  timesθ = get_timesθ(op)

  function snapshot(k::Int)
    println("Snapshot number $k, $id")
    M = assemble_matrix(op)(bfun[k],timesθ)
    i,v = findnz(M[:])
    i,v
  end

  iv = snapshot.(eachindex(bfun))
  row_idx,values = first.(iv),last.(iv)
  check_row_idx(row_idx)
  Snapshots(id,values)
end

function check_row_idx(row_idx::Vector{Vector{Int}})
  @assert all(Broadcasting(a->isequal(a,row_idx[1]))(row_idx)) "Need to correct snaps"
end



























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
  nₛ = isnonlinear(RBInfo, var) ? RBVars.nₛ[1] : RBInfo.mdeim_nsnap

  i_v_block = Broadcasting(Mat)(1:nₛ)
  correct_structures(last.(i_v_block), first.(i_v_block))::Tuple{Matrix{Float}, Vector{Int}}

end

function assemble_Mat_snapshots(
  FEMSpace::FOMS{ID,D},
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  snaps, row_idx = Mat_snapshots(FEMSpace, RBInfo, RBVars, μ, var)
  snaps_POD = POD_for_MDEIM(snaps, RBInfo.ϵₛ)
  snaps_POD, row_idx

end

function assemble_Vec_snapshots(
  FEMSpace::FOMS{ID,D},
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  snaps = Vec_snapshots(FEMSpace, RBInfo, RBVars, μ, var)
  snaps_POD = POD_for_MDEIM(snaps, RBInfo.ϵₛ)
  snaps_POD

end

function Mat_snapshots(
  FEMSpace::FOMST{ID,D},
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  timesθ = get_timesθ(RBInfo)

  function Mat_linear(k::Int)
    println("Snapshot number $k, $var")
    Mats = assemble_FEM_matrix(FEMSpace, RBInfo, μ[k], var, timesθ)
    iv = Broadcasting(Mat -> findnz(Mat[:]))(Mats)
    i, v = first.(iv), last.(iv)
    @assert all([length(i[1]) == length(i[j]) for j = eachindex(i)])
    i[1], blocks_to_matrix(v)
  end

  function Mat_nonlinear(k::Int)
    println("Snapshot number $k, $var")
    Mat = assemble_FEM_nonlinear_matrix(FEMSpace, RBInfo, μ[k],
      RBVars.Φₛ[1][:, k], var)
    typeof(Mat)
    findnz(Mat[:])
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

  function Vec_linear(k::Int)
    println("Snapshot number $k, $var")
    Vec_block = assemble_FEM_vector(FEMSpace, RBInfo, μ[k], var, timesθ)
    blocks_to_matrix(Vec_block)
  end

  function Vec_nonlinear(k::Int)
    println("Snapshot number $k, $var")
    assemble_FEM_nonlinear_vector(FEMSpace, RBInfo, μ[k],
      RBVars.Φₛ[1][:, k], var)
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

  if RBInfo.fun_mdeim
    fun_mdeim(FEMSpace, RBInfo, RBVars, μ, var)::Tuple{Matrix{T}, Matrix{T}, Vector{Int}}
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

  nₛ = isnonlinear(RBInfo, var) ? RBVars.nₛ[1] : RBInfo.mdeim_nsnap
  Mat = Mat_snapshots(FEMSpace, RBInfo, RBVars, μ, var)

  ivals = Broadcasting(Mat)(1:nₛ)
  row_idx = first.(ivals)[1]
  vals = last.(ivals)
  vals_space = blocks_to_matrix(vals)
  vals_time = mode2_unfolding(vals_space, nₛ)

  snaps_space = POD_for_MDEIM(vals_space, RBInfo.ϵₛ)
  snaps_time = POD_for_MDEIM(vals_time, RBInfo.ϵₜ*1e-5)

  snaps_space, snaps_time, row_idx

end

function standard_VMDEIM(
  FEMSpace::FOMST{ID,D},
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  nₛ = isnonlinear(RBInfo, var) ? RBVars.nₛ[1] : RBInfo.mdeim_nsnap
  Vec = Vec_snapshots(FEMSpace, RBInfo, RBVars, μ, var)

  vals = Broadcasting(Vec)(1:nₛ)
  vals_space = blocks_to_matrix(vals)
  vals_time = mode2_unfolding(vals_space, nₛ)

  snaps_space = POD_for_MDEIM(vals_space, RBInfo.ϵₛ)
  snaps_time = POD_for_MDEIM(vals_time, RBInfo.ϵₜ*1e-5)

  snaps_space, snaps_time

end

function fun_mdeim(
  FEMSpace::FOMST{ID,D},
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  θ_space, θ_time = θ_snapshots(FEMSpace, RBInfo, RBVars, μ, var)

  #time
  snaps_time = POD_for_MDEIM(θ_time, RBInfo.ϵₜ*1e-5)

  # space
  θ_space, _ = POD_for_MDEIM(θ_space, RBInfo.ϵₛ)
  Paramθ = ParamInfo(FEMSpace, θ_space, var)
  Mats = assemble_FEM_matrix(FEMSpace, RBInfo, Paramθ)
  iv = Broadcasting(Mat -> findnz(Mat[:]))(Mats)
  i, v = first.(iv), last.(iv)
  @assert all([length(i[1]) == length(i[j]) for j = eachindex(i)])
  row_idx, vals_space = i[1], blocks_to_matrix(v)
  snaps_space = POD_for_MDEIM(vals_space, RBInfo.ϵₛ)

  snaps_space, snaps_time, row_idx

end

function θ_snapshots(
  FEMSpace::FOMST{ID,D},
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  μ::Vector{Vector{Float}},
  var::String) where {ID,D,T}

  timesθ = get_timesθ(RBInfo)
  nₛ = isnonlinear(RBInfo, var) ? RBVars.nₛ[1] : RBInfo.mdeim_nsnap

  function θ_st_linear()
    Param = ParamInfo(RBInfo, μ[1:nₛ], var)
    θinfo = θ_phys_quadp(Param, FEMSpace.phys_quadp, timesθ)
    θblock_space, θblock_time = first.(θinfo), last.(θinfo)
    θ_space, θ_time = blocks_to_matrix(θblock_space), blocks_to_matrix(θblock_time)
    POD_for_MDEIM(θ_space, RBInfo.ϵₛ), POD_for_MDEIM(θ_time, RBInfo.ϵₜ*1e-5)
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
  θ_time = mode2_unfolding(θ_space, nₛ)

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
  row_idx::Vector{Vector{Int}},
  Mat::Vector{Vector{Float}})

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
