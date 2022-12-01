include("MDEIMSnapshots.jl")

abstract type MDEIM{T} end

mutable struct MDEIMSteady{T} <: MDEIM{T}
  rbspace::RBSpaceSteady{T}
  rbspace_idx::RBSpaceSteady{T}
  idx_space::Vector{Int}

  function MDEIMSteady(
    op::RBVarOperator,
    rbspace::RBSpaceSteady{T}) where T

    idx_space = mdeim_idx(op,rbspace)
    rbspace_idx = get_rbspace_idx(rbspace,idx_space)
    new{T}(rbspace,rbspace_idx,idx_space)
  end
end

mutable struct MDEIMUnsteady{T} <: MDEIM{T}
  rbspace::RBSpaceUnsteady{T}
  rbspace_idx::RBSpaceUnsteady{T}
  idx_space::Vector{Int}
  idx_time::Vector{Int}

  function MDEIMUnsteady(
    op::RBVarOperator,
    rbspace::RBSpaceUnsteady{T}) where T

    idx_space,idx_time = mdeim_idx(op,rbspace)...
    rbspace_idx = get_rbspace_idx(rbspace,idx_space,idx_time)
    new{T}(rbspace,rbspace_idx,idx_space,idx_time)
  end
end

function mdeim_offline(info::RBInfo,op::RBVarOperator,μ::Snapshots,args...)
  μ_mdeim = Snapshots(μ,1:info.mdeim_nsnap)
  snaps = mdeim_snapshots(op,info,μ_mdeim)
  rbspace = mdeim_basis(info,snaps)
  mdeim = MDEIM(op,rbspace)
  project_mdeim_space!(mdeim,op)
end

get_rbspace(mdeim::MDEIM) = mdeim.rbspace
get_id(mdeim::MDEIM) = get_id(mdeim::MDEIM)
get_basis_space(mdeim::MDEIM) = get_basis_space(mdeim.rbspace)
get_basis_time(mdeim::MDEIMUnsteady) = get_basis_time(mdeim.rbspace)
get_idx_space(mdeim::MDEIM) = mdeim.idx_space
get_idx_time(mdeim::MDEIMUnsteady) = mdeim.idx_time

function get_rbspace_idx(
  rbspace::RBSpaceSteady,
  idx_space::Vector{Int})

  rbspace_idx = allocate_rbspace(rbspace)

  bs = get_basis_space(rbspace)
  bs_idx = bs[idx_space,:]
  rbspace_idx.basis_space = bs_idx

  rbspace_idx
end

function get_rbspace_idx(
  rbspace::RBSpaceUnsteady,
  idx_space::Vector{Int},
  idx_time::Vector{Int})

  rbspace_idx = allocate_rbspace(rbspace)

  bs = get_basis_space(rbspace)
  bt = get_basis_time(rbspace)
  bs_idx = bs[idx_space,:]
  bt_idx = bt[idx_time,:]
  rbspace_idx.basis_space = bs_idx
  rbspace_idx.basis_time = bt_idx

  rbspace_idx
end

function allocate_mdeim(
  op::RBVarOperator{Top,TT,RBSpaceUnsteady},T=Float) where {Top,TT}
  MDEIM(allocate_rbspace(get_id(op),T))
end

correct_path(path::String,mdeim::MDEIM) = correct_path(joinpath(path,"$(get_id(mdeim))"))

function save(path::String,mdeim::MDEIMSteady)
  save(correct_path(joinpath(path,"basis_space"),mdeim),get_basis_space(mdeim))
  save(correct_path(joinpath(path,"idx_space"),mdeim),get_idx_space(mdeim))
end

function save(path::String,mdeim::MDEIMUnsteady)
  save(correct_path(joinpath(path,"basis_space"),mdeim),get_basis_space(mdeim))
  save(correct_path(joinpath(path,"idx_space"),mdeim),get_idx_space(mdeim))
  save(correct_path(joinpath(path,"basis_time"),mdeim),get_basis_time(mdeim))
  save(correct_path(joinpath(path,"idx_time"),mdeim),get_idx_time(mdeim))
end

function load!(mdeim::RBSpaceUnsteady,path::String)
  basis_space = load(correct_path(joinpath(path,"basis_space"),mdeim))
  basis_time = load(correct_path(joinpath(path,"basis_time"),mdeim))
  idx_space = load(correct_path(joinpath(path,"idx_space"),mdeim))
  idx_time = load(correct_path(joinpath(path,"idx_time"),mdeim))

  mdeim.rbspace = RBSpace(basis_space,basis_time)
  mdeim.idx_space = idx_space
  mdeim.idx_time = idx_time
  mdeim
end

function load!(mdeim::RBSpaceUnsteady,path::String)
  basis_space = load(correct_path(joinpath(path,"basis_space"),mdeim))
  idx_space = load(correct_path(joinpath(path,"idx_space"),mdeim))

  mdeim.rbspace = RBSpace(get_id(mdeim),basis_space)
  mdeim.idx_space = idx_space
  mdeim
end

function load(path::String,T=Float)
  mdeim = allocate_mdeim(op,T)
  load!(mdeim,path)
end

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

function project_mdeim_space!(op::RBLinOperator,mdeim::MDEIM)
  basis_space = get_basis_space(mdeim)
  findnz_map = get_findnz_mapping(op)
  full_basis_space = fill_rows_with_zeros(basis_space,findnz_map)
  rbspace_row = get_rbspace_row(op)

  mdeim.rbspace.basis_space = rbspace_row'*full_basis_space
  mdeim
end

function project_mdeim_space!(op::RBBilinOperator,mdeim::MDEIM)
  basis_space = get_basis_space(mdeim)
  findnz_map = get_findnz_mapping(op)
  full_basis_space = fill_rows_with_zeros(basis_space,findnz_map)

  rbspace_row = get_rbspace_row(op)
  rbspace_col = get_rbspace_col(op)

  mdeim.rbspace.basis_space = rbspace_row'*full_basis_space*rbspace_col
  mdeim
end

function get_reduced_measure(
  op::RBVarOperator,
  mdeim::MDEIM,
  meas::Measure)

  get_reduced_measure(op,meas,get_idx_space(mdeim))
end

function get_reduced_measure(
  op::RBVarOperator,
  meas::Measure,
  idx::Vector{Int})

  get_reduced_measure(op,get_triangulation(meas),idx)
end

function get_reduced_measure(
  op::RBVarOperator,
  trian::Triangulation,
  idx::Vector{Int})

  el = find_mesh_elements(op,trian,idx)
  red_trian = view(trian,el)
  Measure(red_trian,get_degree(get_test(op)))
end

function find_mesh_elements(
  op::RBVarOperator,
  trian::Triangulation,
  idx::Vector{Int})

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
