include("MDEIMSnapshots.jl")

abstract type MDEIM{T} end

mutable struct MDEIMSteady{T} <: MDEIM{T}
  rbspace::RBSpaceSteady{T}
  idx_space::Vector{Int}

  function MDEIMSteady(rbspace::RBSpaceSteady{T}) where T
    new{T}(rbspace,MDEIM_idx(rbspace))
  end
end

mutable struct MDEIMUnsteady{T} <: MDEIM{T}
  rbspace::RBSpaceUnsteady{T}
  idx_space::Vector{Int}
  idx_time::Vector{Int}

  function MDEIMUnsteady(rbspace::RBSpaceUnsteady{T}) where T
    new{T}(rbspace,MDEIM_idx(rbspace)...)
  end
end

function MDEIM(rbspace::RBSpaceSteady{T}) where T
  MDEIMSteady{T}(rbspace,idx_space)
end

function MDEIM(rbspace::RBSpaceUnsteady{T}) where T
  MDEIMUnsteady{T}(rbspace,idx_space,idx_time)
end

function MDEIM(info::RBInfo,op::RBVarOperator,μ::Snapshots,args...)
  μ_mdeim = Snapshots(μ,1:info.mdeim_nsnap)
  snaps = mdeim_snapshots(op,μ_mdeim)
  rbspace = mdeim_basis(snaps...)
  MDEIM(rbspace)
end

function mdeim_basis(snaps_space::Snapshots)
  basis_space = POD(snaps_space)
  RBSpaceSteady(snaps,basis_space)
  #idx = compute_mdeim_idx(basis)
  #el = find_fe_elements(op,idx)
end

function mdeim_basis(snaps_space::Snapshots,snaps_time::Snapshots)
  basis_space = POD(snaps_space)
  basis_time = POD(snaps_time)
  RBSpaceUnsteady(snaps,basis_space,basis_time)
  #idx_space = compute_mdeim_idx(basis_space)
  #idx_time = compute_mdeim_idx(basis_time)
  #el = find_fe_elements(op,idx_space)
end

function mdeim_basis(
  info::RBInfoSteady,
  op::RBBilinOperator{Top,TT,RBSpaceSteady},
  μ::Snapshots) where {Top,TT}

  μ_mdeim = cut_snapshots(μ,1:info.mdeim_nsnap)
  snaps = matrix_snapshots(op,μ_mdeim)
  POD(snaps)
  #idx = compute_mdeim_idx(basis)
  #el = find_fe_elements(op,idx)
end

function mdeim_basis(
  info::RBInfoUnsteady,
  op::RBLinOperator{Top,RBSpaceUnsteady},
  μ::Snapshots) where Top

  μ_mdeim = cut_snapshots(μ,1:info.mdeim_nsnap)
  snaps = vector_snapshots(op,μ_mdeim)
  Nt = get_Nt(snaps)
  snaps2 = mode2_unfolding(snaps,Nt)
  POD(snaps),POD(snaps2)
  #idx_space = compute_mdeim_idx(basis_space)
  #idx_time = compute_mdeim_idx(basis_time)
  #el = find_fe_elements(op,idx_space)
end






function MDEIM_offline(
  MDEIM::MMDEIM{T},
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  var::String) where {ID,T}

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))
  Nₛ = get_Nₛ(RBVars, var)

  Mat, row_idx = assemble_Mat_snapshots(FEMSpace, RBInfo, RBVars, μ, var)
  idx_full, Matᵢ = MDEIM_offline(Mat)
  idx = from_full_idx_to_sparse_idx(idx_full, row_idx, Nₛ)
  idx_space, _ = from_vec_to_mat_idx(idx, Nₛ)
  el = find_FE_elements(FEMSpace, idx_space, var)

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.row_idx, MDEIM.el =
    Mat, Matᵢ, idx, row_idx, el

end

function MDEIM_offline(
  MDEIM::VMDEIM{T},
  RBInfo::ROMInfoS{ID},
  RBVars::ROMS{ID,T},
  var::String) where {ID,T}

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))

  Mat = assemble_Vec_snapshots(FEMSpace, RBInfo, RBVars, μ, var)
  idx, Matᵢ = MDEIM_offline(Mat)
  el = find_FE_elements(FEMSpace, idx, var)

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.el = Mat, Matᵢ, idx, el

end

function MDEIM_offline(
  MDEIM::MMDEIM{T},
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  var::String) where {ID,T}

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))
  Nₛ = get_Nₛ(RBVars, var)

  Mat, Mat_time, row_idx = assemble_Mat_snapshots(FEMSpace, RBInfo, RBVars, μ, var)
  idx_full, Matᵢ = MDEIM_offline(Mat)
  idx = from_full_idx_to_sparse_idx(idx_full, row_idx, Nₛ)
  idx_space, _ = from_vec_to_mat_idx(idx, Nₛ)
  el = find_FE_elements(FEMSpace, idx_space, var)

  time_idx, _ = MDEIM_offline(Mat_time)
  unique!(sort!(time_idx))

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.time_idx, MDEIM.row_idx, MDEIM.el =
    Mat, Matᵢ, idx, time_idx, row_idx, el

end

function MDEIM_offline(
  MDEIM::VMDEIM{T},
  RBInfo::ROMInfoST{ID},
  RBVars::ROMST{ID,T},
  var::String) where {ID,T}

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))

  Mat, Mat_time = assemble_Vec_snapshots(FEMSpace, RBInfo, RBVars, μ, var)
  idx, Matᵢ = MDEIM_offline(Mat)
  el = find_FE_elements(FEMSpace, idx, var)

  time_idx, _ = MDEIM_offline(Mat_time)
  unique!(sort!(time_idx))

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.time_idx, MDEIM.el =
    Mat, Matᵢ, idx, time_idx, el

end

function mdeim_idx(M::Matrix)
  n = size(M)[2]
  idx = Int[]
  append!(idx,Int(argmax(abs.(M[:, 1]))))

  @inbounds for i = 2:n
    res = (M[:,i] - M[:,1:i-1] *
      (M[idx[1:i-1],1:i-1] \ M[idx[1:i-1],i]))
    append!(idx,Int(argmax(abs.(res))[1]))
  end

  unique!(idx)
end

function mdeim_idx(rb::RBSpaceSteady)
  mdeim_idx(get_basis_space(rb))
end

function mdeim_idx(rb::RBSpaceUnsteady)
  mdeim_idx(get_basis_space(rb)),mdeim_idx(get_basis_time(rb))
end
