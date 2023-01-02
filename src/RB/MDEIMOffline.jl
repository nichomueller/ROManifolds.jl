include("MDEIMSnapshots.jl")

abstract type MDEIM end

mutable struct MDEIMSteady <: MDEIM
  rbspace::RBSpaceSteady
  idx_lu_factors::LU
  idx::Vector{Int}
  red_measure::Measure
end

mutable struct MDEIMUnsteady{N} <: MDEIM
  rbspace::NTuple{N,RBSpaceUnsteady}
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

  MDEIMUnsteady{1}((red_rbspace,),idx_lu_factors,idx,red_measure)
end

function MDEIM(
  red_rbspace::NTuple{N,RBSpaceUnsteady},
  idx_lu_factors::LU,
  idx::NTuple{2,Vector{Int}},
  red_measure::Measure) where N

  MDEIMUnsteady{N}(red_rbspace,idx_lu_factors,idx,red_measure)
end

function MDEIM(
  red_rbspace::Tuple,
  idx_lu_factors::NTuple{2,LU},
  idx::NTuple{2,T},
  red_measure::NTuple{2,Measure}) where T

  MDEIM.(red_rbspace,idx_lu_factors,idx,red_measure)
end

function load_mdeim(
  path::String,
  op::Union{RBSteadyLinOperator,RBSteadyBilinOperator,RBSteadyLiftingOperator},
  meas::Measure)

  id = Symbol(last(split(path,'/')))

  basis_space = load(joinpath(path,"basis_space"))
  rbspace = RBSpace(id,basis_space)
  idx_space = Int.(load(joinpath(path,"idx_space"))[:,1])

  factors = load(joinpath(path,"LU"))
  ipiv = Int.(load(joinpath(path,"p"))[:,1])
  idx_lu_factors = LU(factors,ipiv,0)

  red_measure = get_reduced_measure(op,idx_space,meas)

  MDEIM(rbspace,idx_lu_factors,idx_space,red_measure)
end

function load_mdeim(
  path::String,
  op::Union{RBUnsteadyLinOperator,RBUnsteadyBilinOperator,RBUnsteadyLiftingOperator},
  meas::Measure)

  id = Symbol(last(split(path,'/')))

  basis_space = load(joinpath(path,"basis_space"))
  basis_time = load(joinpath(path,"basis_time"))
  rbspace = (RBSpace(id,basis_space,basis_time),)
  if isfile(joinpath(path,"basis_time_shift.csv"))
    basis_time_shift = load(joinpath(path,"basis_time_shift"))
    rbspace_shift = RBSpace(id,basis_space,basis_time_shift)
    rbspace = (rbspace...,rbspace_shift)
  end

  idx_space = load(joinpath(path,"idx_space"))
  idx_time = load(joinpath(path,"idx_time"))
  idx = [idx_space,idx_time]

  factors = load(joinpath(path,"LU"))
  ipiv = Int.(load(joinpath(path,"p"))[:,1])
  idx_lu_factors = LU(factors,ipiv,0)

  red_measure = get_reduced_measure(op,idx_space,meas)

  MDEIM(rbspace,idx_lu_factors,idx,red_measure)
end

get_rbspace(mdeim::MDEIM) = mdeim.rbspace
get_idx_lu_factors(mdeim::MDEIM) = mdeim.idx_lu_factors
get_id(mdeim::MDEIM) = get_id(mdeim.rbspace)
get_basis_space(mdeim::MDEIMSteady) = get_basis_space(mdeim.rbspace)
get_basis_space(mdeim::MDEIMUnsteady) = get_basis_space(first(mdeim.rbspace))
get_basis_time(mdeim::MDEIMUnsteady) = get_basis_time.(mdeim.rbspace)
get_idx_space(mdeim::MDEIMSteady) = mdeim.idx
get_idx_space(mdeim::MDEIMUnsteady) = first(mdeim.idx)
get_idx_time(mdeim::MDEIMUnsteady) = last(mdeim.idx)
get_reduced_measure(mdeim::MDEIM) = mdeim.red_measure

get_basis_space(mdeim::NTuple{2,MDEIM}) = get_basis_space.(mdeim)
get_basis_time(mdeim::NTuple{2,MDEIM}) = get_basis_time.(mdeim)
get_idx_lu_factors(mdeim::NTuple{2,MDEIM}) = get_idx_lu_factors.(mdeim)
get_idx_space(mdeim::NTuple{2,MDEIM}) = get_idx_space.(mdeim)
get_idx_time(mdeim::NTuple{2,MDEIM}) = get_idx_time.(mdeim)
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
  bs = rb_space_projection(op,rbspace)
  RBSpaceSteady(id,bs)
end

function project_mdeim_basis(
  op::Union{RBUnsteadyLinOperator,RBUnsteadyBilinOperator,RBUnsteadyLiftingOperator},
  rbspace)

  id = get_id(rbspace)
  bs = rb_space_projection(op,rbspace)
  bt = rb_time_projection(op,rbspace)
  RBSpaceUnsteady(id,bs,bt)
end

function project_mdeim_basis(
  op::RBUnsteadyBilinOperator,
  rbspace)

  id = get_id(rbspace)
  bs = rb_space_projection(op,rbspace)
  bt = rb_time_projection(op,rbspace)
  RBSpaceUnsteady(id,bs,bt)
end

function rb_space_projection(
  op::RBLinOperator,
  rbspace::RBSpace)

  basis_space = get_basis_space(rbspace)
  rbspace_row = get_rbspace_row(op)
  brow = get_basis_space(rbspace_row)
  brow'*basis_space
end

function rb_space_projection(
  op::RBBilinOperator,
  rbspace::RBSpace)

  basis_space = get_basis_space(rbspace)
  findnz_map = get_findnz_mapping(op)
  sparse_basis_space = sparsevec(basis_space,findnz_map)
  rbspace_row = get_rbspace_row(op)
  brow = get_basis_space(rbspace_row)
  rbspace_col = get_rbspace_col(op)
  bcol = get_basis_space(rbspace_col)

  Qs = length(sparse_basis_space)
  Ns = get_Ns(rbspace_col)
  red_basis_space = zeros(get_ns(rbspace_row)*get_ns(rbspace_col),Qs)
  for q = 1:Qs
    smat = sparsevec_to_sparsemat(sparse_basis_space[q],Ns)
    red_basis_space[:,q] = (brow'*smat*bcol)[:]
  end

  red_basis_space
end

function rb_space_projection(
  op::RBBilinOperator,
  rb::NTuple{2,<:RBSpace})

  rbspace,rbspace_lift = rb
  op_lift = RBLiftingOperator(op)

  red_basis_space = rb_space_projection(op,rbspace)
  red_basis_space_lift = rb_space_projection(op_lift,rbspace_lift)

  red_basis_space,red_basis_space_lift
end

function rb_time_projection(
  op::RBLinOperator,
  rbspace::RBSpace)

  rbrow = get_rbspace_row(op)
  bt = get_basis_time(rbspace)
  Matrix(rb_time_projection(rbrow,bt))
end

function rb_time_projection(
  op::RBBilinOperator,
  rbspace::RBSpace)

  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)
  bt = get_basis_time(rbspace)
  time_proj(idx1,idx2) = rb_time_projection(rbrow,rbcol,bt,idx1,idx2)

  Nt = get_Nt(op)
  idx = 1:Nt
  idx_backwards,idx_forwards = 1:Nt-1,2:Nt

  red_basis_time = Matrix(time_proj(idx,idx))
  red_basis_time_shift = Matrix(time_proj(idx_forwards,idx_backwards))
  red_basis_time,red_basis_time_shift
end

function rb_time_projection(
  op::RBBilinOperator,
  rb::NTuple{2,<:RBSpace})

  rbspace,rbspace_lift = rb
  op_lift = RBLiftingOperator(op)

  red_basis_time = rb_time_projection(op,rbspace)
  red_basis_time_lift = rb_time_projection(op_lift,rbspace_lift)

  red_basis_time,red_basis_time_lift
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

function get_idx_lu_factors(
  rbspace::RBSpaceUnsteady,
  idx_st::NTuple{2,Vector{Int}})

  idx_space,idx_time = idx_st

  bs = get_basis_space(rbspace)
  bt = get_basis_time(rbspace)
  bs_idx = bs[idx_space,:]
  bt_idx = bt[idx_time,:]
  bst_idx = kron(bs_idx,bt_idx)

  lu(bst_idx)
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
