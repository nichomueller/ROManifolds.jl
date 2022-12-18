include("MDEIMSnapshots.jl")

abstract type MDEIM end

mutable struct MDEIMSteady <: MDEIM
  rbspace::RBSpaceSteady
  idx_lu_factors::LU
  idx::Vector{Int}
  red_measure::Measure
end

mutable struct MDEIMUnsteady <: MDEIM
  rbspace::RBSpaceUnsteady
  idx_lu_factors::LU
  idx::NTuple{2,Vector{Int}}
  red_measure::Measure
end

function MDEIM(
  red_rbspace::RBSpaceSteady,
  idx_lu_factors::LU,
  idx::Vector{Int},
  red_measure::Measure)

  MDEIMSteady(red_rbspace,idx_lu_factors,idx,red_measure)
end

function MDEIM(
  red_rbspace::RBSpaceUnsteady,
  idx_lu_factors::LU,
  idx::NTuple{2,Vector{Int}},
  red_measure::Measure)

  MDEIMUnsteady(red_rbspace,idx_lu_factors,idx,red_measure)
end

function MDEIM(
  red_rbspace::NTuple{2,<:RBSpace},
  idx_lu_factors::NTuple{2,LU},
  idx::NTuple{2,T},
  red_measure::NTuple{2,Measure}) where T

  MDEIM.(red_rbspace,idx_lu_factors,idx,red_measure)
end

#= function load_mdeim(
  path::String,
  op::RBLinOperator,
  meas::Measure)

  load_mdeim(path,op,meas)
end

function load_mdeim(
  path::String,
  op::RBBilinOperator,
  meas::Measure)

  load_mdeim(path,op,meas),load_mdeim(path*"_lift",op,meas)
end

function load_mdeim(
  path::String,
  op::RBBilinOperator{Top,UnconstrainedFESpace},
  meas::Measure) where Top

  load_mdeim(path,op,meas)
end =#

function load_mdeim(
  path::String,
  op::Union{RBSteadyLinOperator,RBSteadyBilinOperator},
  meas::Measure)

  id = Symbol(last(split(path,'/')))

  basis_space = load(joinpath(path,"basis_space"))
  rbspace = RBSpace(id,basis_space)
  idx_space = Int.(load(joinpath(path,"idx_space"))[:,1])

  factors = load(joinpath(path,"LU"))
  ipiv = Int.(load(joinpath(path,"p"))[:,1])
  idx_lu_factors = LU(factors,ipiv,0)

  red_measure = get_reduced_measure(op,meas,idx_space)

  MDEIM(rbspace,idx_lu_factors,idx_space,red_measure)
end

function load_mdeim(
  path::String,
  op::Union{RBUnsteadyLinOperator,RBUnsteadyBilinOperator},
  meas::Measure)

  id = Symbol(last(split(path,'/')))

  basis_space = load(joinpath(path,"basis_space"))
  basis_time = load(joinpath(path,"basis_time"))
  rbspace = RBSpace(id,basis_space,basis_time)

  idx_space = load(joinpath(path,"idx_space"))
  idx_time = load(joinpath(path,"idx_time"))
  idx = [idx_space,idx_time]

  factors = load(joinpath(path,"LU"))
  ipiv = Int.(load(joinpath(path,"p"))[:,1])
  idx_lu_factors = LU(factors,ipiv,0)

  red_measure = get_reduced_measure(op,meas,idx_space)

  MDEIM(rbspace,idx_lu_factors,idx,red_measure)
end

get_rbspace(mdeim::MDEIM) = mdeim.rbspace
get_idx_lu_factors(mdeim::MDEIM) = mdeim.idx_lu_factors
get_id(mdeim::MDEIM) = get_id(mdeim.rbspace)
get_basis_space(mdeim::MDEIM) = get_basis_space(mdeim.rbspace)
get_basis_time(mdeim::MDEIMUnsteady) = get_basis_time(mdeim.rbspace)
get_idx(mdeim::MDEIM) = mdeim.idx
get_idx_space(mdeim::MDEIMSteady) = mdeim.idx
get_idx_space(mdeim::MDEIMUnsteady) = first(mdeim.idx)
get_idx_time(mdeim::MDEIMUnsteady) = last(mdeim.idx)
get_reduced_measure(mdeim::MDEIM) = mdeim.red_measure

get_basis_space(mdeim::NTuple{2,MDEIM}) = get_basis_space.(mdeim)
get_basis_time(mdeim::NTuple{2,MDEIMUnsteady}) = get_basis_time.(mdeim)
get_idx_lu_factors(mdeim::NTuple{2,MDEIM}) = get_idx_lu_factors.(mdeim)
get_idx_space(mdeim::NTuple{2,MDEIM}) = get_idx_space.(mdeim)
get_idx_time(mdeim::NTuple{2,MDEIMUnsteady}) = get_idx_time.(mdeim)
get_reduced_measure(mdeim::NTuple{2,MDEIM}) = get_reduced_measure.(mdeim)

function mdeim_offline(
  info::RBInfo,
  op::RBVarOperator,
  μ::Vector{Param},
  meas::ProblemMeasures,
  field=:dΩ,
  args...)

  μ_mdeim = μ[1:info.mdeim_nsnap]
  snaps = mdeim_snapshots(op,info,μ_mdeim,args...)
  rbspace = mdeim_basis(info,snaps)
  red_rbspace = project_mdeim_basis(op,rbspace)
  idx = mdeim_idx(rbspace)
  idx_lu_factors = get_idx_lu_factors(rbspace,idx)
  idx = recast_in_full_dim(op,idx)
  red_meas = get_reduced_measure(op,idx,meas,field)

  MDEIM(red_rbspace,idx_lu_factors,idx,red_meas)
end

mdeim_basis(info::RBInfoSteady,snaps) = RBSpaceSteady(snaps;ismdeim=Val(true),ϵ=info.ϵ)
mdeim_basis(info::RBInfoUnsteady,snaps) = RBSpaceUnsteady(snaps;ismdeim=Val(true),ϵ=info.ϵ)

function project_mdeim_basis(
  op::Union{RBSteadyLinOperator,RBSteadyBilinOperator,RBSteadyLiftingOperator},
  rbspace)

  id = get_id(rbspace)
  bs = project_mdeim_basis_space(op,rbspace)
  RBSpaceSteady(id,bs)
end

function project_mdeim_basis(
  op::Union{RBUnsteadyLinOperator,RBUnsteadyBilinOperator,RBUnsteadyLiftingOperator},
  rbspace)

  id = get_id(rbspace)
  bs = project_mdeim_basis_space(op,rbspace)
  bt = project_mdeim_basis_time(op,rbspace)
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
  rb::NTuple{2,<:RBSpace})

  rbspace,rbspace_lift = rb

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
  rb::NTuple{2,<:RBSpace})
  rbspace,rbspace_lift = rb
  get_basis_time(rbspace),get_basis_time(rbspace_lift)
end

mdeim_idx(rbspace::NTuple{2,<:RBSpace}) = mdeim_idx.(rbspace)

function mdeim_idx(rbspace::RBSpaceSteady)
  idx_space = mdeim_idx(get_basis_space(rbspace))
  idx_space
end

function mdeim_idx(rbspace::RBSpaceUnsteady)
  idx_space = mdeim_idx(get_basis_space(rbspace))
  idx_time = mdeim_idx(get_basis_time(rbspace))
  idx_space,idx_time
end

function mdeim_idx(M::Matrix{Float})
  n = size(M)[2]
  idx = Int[]
  append!(idx,Int(argmax(abs.(M[:,1]))))

  @inbounds for i = 2:n
    res = (M[:,i] - M[:,1:i-1] *
      (M[idx[1:i-1],1:i-1] \ M[idx[1:i-1],i]))
    append!(idx,Int(argmax(abs.(res))))
  end

  unique(idx)
end

function get_idx_lu_factors(
  rbspace::RBSpaceSteady,
  idx_space::Vector{Int})

  bs = get_basis_space(rbspace)
  bs_idx = bs[idx_space,:]
  lu(bs_idx)
end

function get_idx_lu_factors(
  rbspace::NTuple{2,RBSpaceSteady},
  idx_space::NTuple{2,Vector{Int}})
  get_idx_lu_factors.(rbspace,idx_space)
end

# CORRECT THIS!
function get_idx_lu_factors(
  rbspace::RBSpaceUnsteady,
  idx_st::NTuple{2,Vector{Int}})

  idx_space,idx_time = idx_st

  bs = get_basis_space(rbspace)
  bt = get_basis_time(rbspace)
  bs_idx = bs[idx_space,:]
  bt_idx = bt[idx_time,:]

  lu(bs_idx)
end

function get_idx_lu_factors(
  rbspace::NTuple{2,RBSpaceUnsteady},
  idx_st::NTuple{2,NTuple{2,Vector{Int}}})
  get_idx_lu_factors.(rbspace,idx_st)
end

recast_in_full_dim(::RBLinOperator,idx_tmp) = idx_tmp

recast_in_full_dim(op::RBBilinOperator,idx_tmp::Vector{Int}) =
  get_findnz_mapping(op)[idx_tmp]

recast_in_full_dim(op::RBBilinOperator,idx_tmp::NTuple{2,Vector{Int}}) =
  recast_in_full_dim(op,first(idx_tmp)),last(idx_tmp)

function recast_in_full_dim(
  op::RBBilinOperator,
  idx_tmp::NTuple{2,NTuple{2,Vector{Int}}})

  idx,idx_lift = idx_tmp
  (recast_in_full_dim(op,first(idx)),last(idx)),idx_lift
end

function get_reduced_measure(
  op::Union{RBSteadyLinOperator,RBSteadyBilinOperator,RBSteadyLiftingOperator},
  idx::Vector{Int},
  meas::ProblemMeasures,
  field=:dΩ)

  m = getproperty(meas,field)
  get_reduced_measure(op,idx,m)
end

function get_reduced_measure(
  op::Union{RBUnsteadyLinOperator,RBUnsteadyBilinOperator,RBUnsteadyLiftingOperator},
  idx::NTuple{2,Vector{Int}},
  meas::ProblemMeasures,
  field=:dΩ)

  m = getproperty(meas,field)
  get_reduced_measure(op,first(idx),m)
end

function get_reduced_measure(
  op::RBSteadyBilinOperator,
  idx::NTuple{2,Vector{Int}},
  meas::ProblemMeasures,
  field=:dΩ)

  idx_space,idx_space_lift = idx
  m = get_reduced_measure(op,idx_space,meas,field)
  m_lift = get_reduced_measure(op,idx_space_lift,meas,field)
  m,m_lift
end

function get_reduced_measure(
  op::RBUnsteadyBilinOperator,
  idx::NTuple{2,NTuple{2,Vector{Int}}},
  meas::ProblemMeasures,
  field=:dΩ)

  idx_space,idx_space_lift = idx
  m = get_reduced_measure(op,idx_space,meas,field)
  m_lift = get_reduced_measure(op,idx_space_lift,meas,field)
  m,m_lift
end

function get_reduced_measure(
  op::RBVarOperator,
  idx::Vector{Int},
  meas::Measure)

  get_reduced_measure(op,idx,get_triangulation(meas))
end

function get_reduced_measure(
  op::RBVarOperator,
  idx::Vector{Int},
  trian::Triangulation)

  el = find_mesh_elements(op,idx,trian)
  red_trian = view(trian,el)
  Measure(red_trian,get_degree(get_test(op)))
end

function find_mesh_elements(
  op::RBVarOperator,
  idx_tmp::Vector{Int},
  trian::Triangulation)

  idx = recast_in_mat_form(op,idx_tmp)
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

recast_in_mat_form(::RBLinOperator,idx_tmp::Vector{Int}) = idx_tmp

function recast_in_mat_form(op::RBBilinOperator,idx_tmp::Vector{Int})
  Ns = get_Ns(get_rbspace_row(op))
  idx_space,_ = from_vec_to_mat_idx(idx_tmp,Ns)
  idx_space
end
