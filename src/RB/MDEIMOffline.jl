include("MDEIMSnapshots.jl")

abstract type MDEIM{T} end

mutable struct MDEIMSteady{T} <: MDEIM{T}
  rbspace::RBSpaceSteady{T}
  idx_space::Vector{Int}

  function MDEIMSteady(
    op::RBVarOperator,
    rbspace::RBSpaceSteady{T}) where T

    new{T}(rbspace,mdeim_idx(op,rbspace))
  end
end

mutable struct MDEIMUnsteady{T} <: MDEIM{T}
  rbspace::RBSpaceUnsteady{T}
  idx_space::Vector{Int}
  idx_time::Vector{Int}

  function MDEIMUnsteady(
    op::RBVarOperator,
    rbspace::RBSpaceUnsteady{T}) where T

    new{T}(rbspace,mdeim_idx(op,rbspace)...)
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
  rbspace = mdeim_basis(info,snaps)
  MDEIM(rbspace)
end

get_basis_space(mdeim::MDEIM) = get_basis_space(mdeim.rbspace)
get_basis_time(mdeim::MDEIMUnsteady) = get_basis_time(mdeim.rbspace)

mdeim_basis(info::RBInfoSteady,snaps::Snapshots) = RBSpaceSteady(snaps;info.ϵ)
mdeim_basis(info::RBInfoUnsteady,snaps::Snapshots) = RBSpaceUnsteady(snaps;info.ϵ)

function mdeim_idx(op::RBVarOperator,rb::RBSpaceSteady)
  idx_space_tmp = mdeim_idx(get_basis_space(rb))
  fix_idx(op,idx_space_tmp)
end

function mdeim_idx(op::RBVarOperator,rb::RBSpaceUnsteady)
  idx_space_tmp = mdeim_idx(get_basis_space(rb))
  idx_space = fix_idx(op,idx_space_tmp)
  idx_time = mdeim_idx(get_basis_time(rb))
  idx_space,idx_time
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

function fix_idx(::RBLinOperator,idx_tmp::Vector{Int})
  idx_tmp
end

function fix_idx(op::RBBilinOperator,idx_tmp::Vector{Int})
  findnz_map = get_findnz_mapping(op)
  findnz_map[idx_tmp]
end

function find_mesh_elements(
  op::RBVarOperator,
  trian::Triangulation,
  idx::Vector)

  connectivity = get_cell_dof_ids(op,trian)

  el = Int[]
  for i = eachindex(idx)
    for j = axes(connectivity,1)
      if idx[i] in abs.(connectivity[j])
        append!(el,j)
      end
    end
  end

  unique(el)
end
