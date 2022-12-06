include("MDEIMSnapshots.jl")

abstract type MDEIM{T} end

mutable struct MDEIMSteady{T} <: MDEIM{T}
  rbspace::RBSpaceSteady{T}
  rbspace_idx::RBSpaceSteady{T}
  idx_space::Vector{Int}
  red_measure::Measure
end

mutable struct MDEIMUnsteady{T} <: MDEIM{T}
  rbspace::RBSpaceUnsteady{T}
  rbspace_idx::RBSpaceUnsteady{T}
  idx_space::Vector{Int}
  idx_time::Vector{Int}
  red_measure::Measure
end

function MDEIM(
  red_rbspace::RBSpaceSteady{T},
  rbspace_idx::RBSpaceSteady{T},
  idx_space::Vector{Int},
  red_meas::Measure) where T

  MDEIMSteady{T}(red_rbspace,rbspace_idx,idx_space,red_meas)
end

function MDEIM(
  red_rbspace::RBSpaceUnsteady{T},
  rbspace_idx::RBSpaceUnsteady{T},
  idx_st::NTuple{2,Vector{Int}},
  red_meas::Measure) where T

  MDEIMUnsteady{T}(red_rbspace,rbspace_idx,idx_st...,red_meas)
end

function MDEIM(
  red_rbspace::NTuple{2,<:RBSpace},
  rbspace_idx::NTuple{2,<:RBSpace},
  idx_st,
  red_meas::NTuple{2,Measure})

  MDEIM.(red_rbspace,rbspace_idx,idx_st,red_meas)
end

function mdeim_offline(
  info::RBInfo,
  op::RBVarOperator,
  μ::Vector{Param},
  meas::ProblemMeasures,
  field=:dΩ)

  μ_mdeim = μ[1:info.mdeim_nsnap]
  snaps = mdeim_snapshots(op,info,μ_mdeim)
  rbspace = mdeim_basis(info,snaps)
  red_rbspace = project_mdeim_basis(op,rbspace)
  idx = mdeim_idx(rbspace)
  rbspace_idx = get_rbspace_idx(rbspace,idx)
  red_meas = get_reduced_measure(op,idx,meas,field)
  idx = fix_idx_space(op,idx)

  MDEIM(red_rbspace,rbspace_idx,idx,red_meas)
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

  id = get_id(rbspace)
  bs = get_basis_space(rbspace)
  bs_idx = bs[idx_space,:]

  RBSpaceSteady(id,bs_idx)
end

function get_rbspace_idx(
  rbspace::NTuple{2,RBSpaceSteady},
  idx_space::NTuple{2,Vector{Int}})
  get_rbspace_idx.(rbspace,idx_space)
end

function get_rbspace_idx(
  rbspace::RBSpaceUnsteady,
  idx_st::NTuple{2,Vector{Int}})

  idx_space,idx_time = idx_st

  id = get_id(rbspace)
  bs = get_basis_space(rbspace)
  bt = get_basis_time(rbspace)
  bs_idx = bs[idx_space,:]
  bt_idx = bt[idx_time,:]

  RBSpaceUnsteady(id,bs_idx,bt_idx)
end

function get_rbspace_idx(
  rbspace::NTuple{2,RBSpaceUnsteady},
  idx_st::NTuple{2,NTuple{2,Vector{Int}}})
  get_rbspace_idx.(rbspace,idx_st)
end

function allocate_mdeim(
  op::RBVarOperator{Top,TT,<:RBSpaceUnsteady},T=Float) where {Top,TT}
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

mdeim_basis(info::RBInfoSteady,snaps) = RBSpaceSteady(snaps;info.ϵ)
mdeim_basis(info::RBInfoUnsteady,snaps) = RBSpaceUnsteady(snaps;info.ϵ)

function project_mdeim_basis(
  op::RBVarOperator{Top,TT,<:RBSpaceSteady},
  rbspace) where {Top,TT}

  id = get_id(op)
  bs = project_mdeim_basis_space(op,rbspace...)
  RBSpaceSteady(id,bs)
end

function project_mdeim_basis(
  op::RBVarOperator{Top,TT,<:RBSpaceUnsteady},
  rbspace) where {Top,TT}

  id = get_id(op)
  bs = project_mdeim_basis_space(op,rbspace...)
  bt = project_mdeim_basis_time(op,rbspace...)
  RBSpaceUnsteady(id,bs,bt)
end

function project_mdeim_basis_space(
  op::RBLinOperator,
  rbspace::RBSpace)

  basis_space = get_basis_space(rbspace)
  rbspace_row = get_rbspace_row(op)
  brow = get_basis_space(rbspace_row)
  brow'*basis_space
end

function project_mdeim_basis_space(
  op::RBBilinOperator,
  rbspace::RBSpace)

  basis_space = get_basis_space(rbspace)
  findnz_map = get_findnz_mapping(op)
  sparse_basis_space = sparsevec(basis_space,findnz_map)
  rbspace_row = get_rbspace_row(op)
  brow = get_basis_space(rbspace_row)
  rbspace_col = get_rbspace_col(op)
  bcol = get_basis_space(rbspace_col)

  Q = length(sparse_basis_space)
  red_basis_space = zeros(get_ns(rbspace_row)*get_ns(rbspace_col),Q)
  for q = 1:Q
    smat = sparsevec_to_sparsemat(sparse_basis_space[q],get_Ns(rbspace_col))
    red_basis_space[:,q] = (brow'*smat*bcol)[:]
  end

  red_basis_space
end

function project_mdeim_basis_space(
  op::RBBilinOperator,
  rbspace::RBSpace,
  rbspace_lift::RBSpace)

  red_basis_space = project_mdeim_basis_space(op,rbspace)

  basis_space_lift = get_basis_space(rbspace_lift)
  rbspace_row_lift = get_rbspace_row(op)
  brow_lift = get_basis_space(rbspace_row_lift)
  red_basis_space_lift = brow_lift'*basis_space_lift

  red_basis_space,red_basis_space_lift
end

# CORRECT THIS!
function project_mdeim_basis_time(
  ::RBLinOperator,
  rbspace::RBSpace)
  get_basis_time(rbspace)
end

# CORRECT THIS!
function project_mdeim_basis_time(
  ::RBBilinOperator,
  rbspace::RBSpace)
  get_basis_time(rbspace)
end

# CORRECT THIS!
function project_mdeim_basis_time(
  ::RBBilinOperator,
  rbspace::RBSpace,
  rbspace_lift::RBSpace)
  get_basis_time(rbspace),get_basis_time(rbspace_lift)
end

#= function mdeim_idx(op::RBVarOperator,rbspace::RBSpaceSteady)
  idx_space = mdeim_idx(get_basis_space(rbspace))
  rbspace_idx = get_rbspace_idx(rbspace,idx_space)
  rbspace_idx,fix_idx_space(op,idx_space)
end

function mdeim_idx(op::RBVarOperator,rbspace::RBSpaceUnsteady)
  idx_space = mdeim_idx(get_basis_space(rbspace))
  idx_time = mdeim_idx(get_basis_time(rbspace))
  rbspace_idx = get_rbspace_idx(rbspace,idx_space,idx_time)
  rbspace_idx,fix_idx_space(op,idx_space),idx_time
end =#

function mdeim_idx(rbspace::RBSpaceSteady)
  idx_space = mdeim_idx(get_basis_space(rbspace))
  idx_space
end

function mdeim_idx(rbspace::RBSpaceUnsteady)
  idx_space = mdeim_idx(get_basis_space(rbspace))
  idx_time = mdeim_idx(get_basis_time(rbspace))
  idx_space,idx_time
end

mdeim_idx(rbspace::NTuple{2,<:RBSpace}) = mdeim_idx.(rbspace)

function mdeim_idx(M::Matrix)
  n = size(M)[2]
  idx = Int[]
  append!(idx,Int(argmax(abs.(M[:,1]))))

  @inbounds for i = 2:n
    res = (M[:,i] - M[:,1:i-1] *
      (M[idx[1:i-1],1:i-1] \ M[idx[1:i-1],i]))
    append!(idx,Int(argmax(abs.(res))[1]))
  end

  unique!(idx)
end

fix_idx_space(::RBLinOperator,idx_tmp::Vector{Int}) = idx_tmp

fix_idx_space(op::RBBilinOperator,idx_tmp::Vector{Int}) = get_findnz_mapping(op)[idx_tmp]

function fix_idx_space(
  op::RBBilinOperator{Top,TT,<:RBSpaceSteady},
  idx_tmp::NTuple{2,Vector{Int}}) where {Top,TT}
  Broadcasting(i->fix_idx_space(op,i))(idx_tmp)
end

function fix_idx_space(
  op::RBBilinOperator{Top,TT,<:RBSpaceUnsteady},
  idx_tmp::NTuple{2,Vector{Int}}) where {Top,TT}
  fix_idx_space(op,first(idx_tmp)),last(idx_tmp)
end

function fix_idx_space(
  op::RBBilinOperator{Top,TT,<:RBSpaceUnsteady},
  idx_tmp::NTuple{2,NTuple{2,Vector{Int}}}) where {Top,TT}
  Broadcasting(i->fix_idx_space(op,i))(idx_tmp)
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

function get_reduced_measure(
  op::RBVarOperator{Top,TT,<:RBSpaceSteady},
  idx::NTuple{2,Vector{Int}},
  meas::ProblemMeasures,
  field=:dΩ) where {Top,TT}
  Broadcasting(i->get_reduced_measure(op,i,meas,field))(idx)
end

function get_reduced_measure(
  op::RBVarOperator{Top,TT,<:RBSpaceUnsteady},
  idx::NTuple{2,Vector{Int}},
  meas::ProblemMeasures,
  field=:dΩ) where {Top,TT}
  get_reduced_measure(op,first(idx),meas,field)
end

function get_reduced_measure(
  op::RBVarOperator{Top,TT,<:RBSpaceUnsteady},
  idx::NTuple{2,NTuple{2,Vector{Int}}},
  meas::ProblemMeasures,
  field=:dΩ) where {Top,TT}
  get_reduced_measure(op,first.(idx),meas,field)
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
