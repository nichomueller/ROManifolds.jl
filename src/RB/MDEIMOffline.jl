include("MDEIMSnapshots.jl")

abstract type MDEIM{T} end

mutable struct MDEIMSteady{T} <: MDEIM{T}
  rbspace::RBSpaceSteady{T}
  rbspace_idx::RBSpaceSteady{T}
  idx_space::Vector{Int}
  red_measure::Measure

  function MDEIMSteady(
    op::RBVarOperator,
    rbspace::RBSpaceSteady{T},
    meas::ProblemMeasures,
    field=:dΩ) where T

    idx_space = mdeim_idx(op,rbspace)
    rbspace_idx = get_rbspace_idx(rbspace,idx_space)
    red_meas = get_reduced_measure(op,idx_space,meas,field)
    new{T}(rbspace,rbspace_idx,idx_space,red_meas)
  end
end

mutable struct MDEIMUnsteady{T} <: MDEIM{T}
  rbspace::RBSpaceUnsteady{T}
  rbspace_idx::RBSpaceUnsteady{T}
  idx_space::Vector{Int}
  idx_time::Vector{Int}

  function MDEIMUnsteady(
    op::RBVarOperator,
    rbspace::RBSpaceUnsteady{T},
    meas::ProblemMeasures,
    field=:dΩ) where T

    idx_space,idx_time = mdeim_idx(op,rbspace)
    rbspace_idx = get_rbspace_idx(rbspace,idx_space,idx_time)
    red_meas = get_reduced_measure(op,idx_space,meas,field)
    new{T}(rbspace,rbspace_idx,idx_space,idx_time,red_meas)
  end
end

function mdeim_offline(
  info::RBInfo,
  op::RBVarOperator{Top,UnconstrainedFESpace,Tsp},
  μ::Vector{Param},
  meas::ProblemMeasures,
  field=:dΩ) where {Top,Tsp}

  μ_mdeim = μ[1:info.mdeim_nsnap]
  snaps = mdeim_snapshots(op,info,μ_mdeim)
  rbspace = mdeim_basis(info,snaps)
  mdeim = MDEIM(op,rbspace,meas,field)
  project_mdeim_space!(mdeim,op)
end

function mdeim_offline(
  info::RBInfo,
  op::RBVarOperator,
  μ::Vector{Param},
  meas::ProblemMeasures,
  field=:dΩ)

  μ_mdeim = μ[1:info.mdeim_nsnap]
  snaps,snaps_lift = mdeim_snapshots(op,info,μ_mdeim)
  rbspace = mdeim_basis(info,snaps)
  rbspace_lift = mdeim_basis(info,snaps_lift)
  mdeim = MDEIM(op,rbspace,meas,field)
  mdeim_lift = MDEIM(op,rbspace_lift,meas,field)
  project_mdeim_space!(mdeim,op),project_mdeim_space!(mdeim_lift,op)
end

get_rbspace(mdeim::MDEIM) = mdeim.rbspace
get_rbspace_idx(mdeim::MDEIM) = mdeim.rbspace_idx
get_id(mdeim::MDEIM) = get_id(mdeim::MDEIM)
get_basis_space(mdeim::MDEIM) = get_basis_space(mdeim.rbspace)
get_basis_time(mdeim::MDEIMUnsteady) = get_basis_time(mdeim.rbspace)
get_idx_space(mdeim::MDEIM) = mdeim.idx_space
get_idx_time(mdeim::MDEIMUnsteady) = mdeim.idx_time
get_reduced_measure(mdeim::MDEIM) = mdeim.red_measure

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

correct_path(path::String,mdeim::MDEIM) = path*"$(get_id(mdeim))"

function save(path::String,mdeim::MDEIMSteady)
  save(joinpath(path,"basis_space"*"$(get_id(mdeim))"),get_basis_space(mdeim))
  save(joinpath(path,"idx_space"*"$(get_id(mdeim))"),get_idx_space(mdeim))
end

function save(path::String,mdeim::MDEIMUnsteady)
  save(joinpath(path,"basis_space"*"$(get_id(mdeim))"),get_basis_space(mdeim))
  save(joinpath(path,"idx_space"*"$(get_id(mdeim))"),get_idx_space(mdeim))
  save(joinpath(path,"basis_time"*"$(get_id(mdeim))"),get_basis_time(mdeim))
  save(joinpath(path,"idx_time"*"$(get_id(mdeim))"),get_idx_time(mdeim))
end

function save(path::String,mdeim::NTuple{2,<:MDEIM})
  m,mlift = mdeim
  save(path,m)
  save(joinpath(path,"_lift"),mlift)
end

function load!(mdeim::MDEIMSteady,path::String)
  basis_space = load(correct_path(joinpath(path,"basis_space"),mdeim))
  idx_space = load(correct_path(joinpath(path,"idx_space"),mdeim))

  mdeim.rbspace = RBSpace(get_id(mdeim),basis_space)
  mdeim.idx_space = idx_space
  mdeim
end

function load!(mdeim::MDEIMUnsteady,path::String)
  basis_space = load(correct_path(joinpath(path,"basis_space"),mdeim))
  idx_space = load(correct_path(joinpath(path,"idx_space"),mdeim))
  basis_time = load(correct_path(joinpath(path,"basis_time"),mdeim))
  idx_time = load(correct_path(joinpath(path,"idx_time"),mdeim))

  mdeim.rbspace = RBSpace(basis_space,basis_time)
  mdeim.idx_space = idx_space
  mdeim.idx_time = idx_time
  mdeim
end

load_mdeim(info::RBInfo,args...) = if info.load_offline load_mdeim(info.offline_path,args...) end

function load_mdeim(path::String,T=Float)
  mdeim = allocate_mdeim(op,T)
  load!(mdeim,path)
end

function blocks(mdeim::MDEIMSteady)
  mdeim.rbspace.basis_space = blocks(mdeim.rbspace.basis_space)
end

function blocks(mdeim::MDEIMUnsteady)
  mdeim.rbspace.basis_space = blocks(mdeim.rbspace.basis_space)
  mdeim.rbspace.basis_time = blocks(mdeim.rbspace.basis_time)
  mdeim
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

  Ns,Q = get_Ns(op),size(full_basis_space)[2]
  full_basis_space_resh = reshape(full_basis_space,Ns,Ns,Q)

  mdeim.rbspace.basis_space =
    [rbspace_row'*full_basis_space_resh[:,:,q]*rbspace_col for q=1:Q]
  mdeim
end

function get_reduced_measure(
  op::RBVarOperator,
  idx::Vector{Int},
  meas::ProblemMeasures,
  field=:dΩ)

  m = getproperty(meas,field)
  get_reduced_measure(op,m,idx)
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
