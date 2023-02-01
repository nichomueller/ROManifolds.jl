include("MDEIMSnapshots.jl")

abstract type MDEIM end

mutable struct MDEIMSteady <: MDEIM
  rbspace::RBSpaceSteady
  red_lu_factors::LU
  idx::Vector{Int}
  red_measure::Measure
end

mutable struct MDEIMUnsteady <: MDEIM
  rbspace::RBSpaceUnsteady
  red_lu_factors::LU
  idx::NTuple{2,Vector{Int}}
  red_measure::Measure
end

function MDEIM(
  red_rbspace::RBSpaceSteady,
  red_lu_factors::LU,
  idx::Vector{Int},
  red_measure::Measure)

  MDEIMSteady(red_rbspace,red_lu_factors,idx,red_measure)
end

function MDEIM(
  red_rbspace::RBSpaceUnsteady,
  red_lu_factors::LU,
  idx::NTuple{2,Vector{Int}},
  red_measure::Measure)

  MDEIMUnsteady(red_rbspace,red_lu_factors,idx,red_measure)
end

function MDEIM(
  info::RBInfo,
  op::RBLinVariable,
  μ::Vector{Param},
  meas::ProblemMeasures,
  field::Symbol,
  args...)

  μ_mdeim = μ[1:info.mdeim_nsnap]
  snaps = mdeim_snapshots(info,op,μ_mdeim,args...)
  rbspace = mdeim_basis(info,snaps)
  red_rbspace = project_mdeim_basis(op,rbspace)
  idx = mdeim_idx(rbspace)
  red_lu_factors = get_red_lu_factors(info,rbspace,idx)
  red_meas = get_red_measure(op,idx,meas,field)

  MDEIM(red_rbspace,red_lu_factors,idx,red_meas)
end

function MDEIM(
  info::RBInfo,
  op::RBBilinVariable,
  μ::Vector{Param},
  meas::ProblemMeasures,
  field::Symbol,
  args...)

  μ_mdeim = μ[1:info.mdeim_nsnap]
  findnz_map,snaps = mdeim_snapshots(info,op,μ_mdeim,args...)
  rbspace = mdeim_basis(info,snaps)
  red_rbspace = project_mdeim_basis(op,rbspace,findnz_map)
  idx = mdeim_idx(rbspace)
  red_lu_factors = get_red_lu_factors(info,rbspace,idx)
  idx = recast_in_full_dim(idx,findnz_map)
  red_meas = get_red_measure(op,idx,meas,field)

  MDEIM(red_rbspace,red_lu_factors,idx,red_meas)
end

function save(path::String,mdeim::MDEIMSteady)
  save(joinpath(path,"basis_space"),get_basis_space(mdeim))
  save(joinpath(path,"idx_space"),get_idx_space(mdeim))
  red_lu = get_red_lu_factors(mdeim)
  save(joinpath(path,"LU"),red_lu.factors)
  save(joinpath(path,"p"),red_lu.ipiv)
end

function save(path::String,mdeim::MDEIMUnsteady)
  save(joinpath(path,"basis_space"),get_basis_space(mdeim))
  save(joinpath(path,"idx_space"),get_idx_space(mdeim))
  save(joinpath(path,"basis_time"),get_basis_time(mdeim))
  save(joinpath(path,"idx_time"),get_idx_time(mdeim))
  red_lu = get_red_lu_factors(mdeim)
  save(joinpath(path,"LU"),red_lu.factors)
  save(joinpath(path,"p"),red_lu.ipiv)
end

function load(
  path::String,
  op::RBSteadyVariable,
  meas::Measure)

  id = Symbol(last(split(path,'/')))

  basis_space = load(joinpath(path,"basis_space"))
  rbspace = RBSpace(id,basis_space)
  idx_space = Int.(load(joinpath(path,"idx_space"))[:,1])

  factors = load(joinpath(path,"LU"))
  ipiv = Int.(load(joinpath(path,"p"))[:,1])
  red_lu_factors = LU(factors,ipiv,0)

  red_measure = get_red_measure(op,idx_space,meas)

  MDEIM(rbspace,red_lu_factors,idx_space,red_measure)
end

function load(
  path::String,
  op::RBUnsteadyVariable,
  meas::Measure)

  id = Symbol(last(split(path,'/')))

  basis_space = load(joinpath(path,"basis_space"))
  basis_time = load(joinpath(path,"basis_time"))
  rbspace = RBSpace(id,basis_space,basis_time)

  idx_space = Int.(load(joinpath(path,"idx_space"))[:,1])
  idx_time = Int.(load(joinpath(path,"idx_time"))[:,1])
  idx = (idx_space,idx_time)

  factors = load(joinpath(path,"LU"))
  ipiv = Int.(load(joinpath(path,"p"))[:,1])
  red_lu_factors = LU(factors,ipiv,0)

  red_measure = get_red_measure(op,idx_space,meas)

  MDEIM(rbspace,red_lu_factors,idx,red_measure)
end

get_rbspace(mdeim::MDEIM) = mdeim.rbspace
get_red_lu_factors(mdeim::MDEIM) = mdeim.red_lu_factors
get_id(mdeim::MDEIM) = get_id(mdeim.rbspace)
get_basis_space(mdeim::MDEIM) = get_basis_space(mdeim.rbspace)
get_basis_time(mdeim::MDEIMUnsteady) = get_basis_time(mdeim.rbspace)
get_idx_space(mdeim::MDEIMSteady) = mdeim.idx
get_idx_space(mdeim::MDEIMUnsteady) = first(mdeim.idx)
get_idx_time(mdeim::MDEIMUnsteady) = last(mdeim.idx)
get_red_measure(mdeim::MDEIM) = mdeim.red_measure

mdeim_basis(info::RBInfoSteady,snaps) = RBSpaceSteady(snaps;ismdeim=Val(true),ϵ=info.ϵ)
mdeim_basis(info::RBInfoUnsteady,snaps) = RBSpaceUnsteady(snaps;ismdeim=Val(true),ϵ=info.ϵ)

function project_mdeim_basis(
  op::RBSteadyVariable,
  rbspace,
  args...)

  id = get_id(rbspace)
  bs = rb_space_projection(op,rbspace,args...)
  RBSpaceSteady(id,bs)
end

function project_mdeim_basis(
  op::RBUnsteadyVariable,
  rbspace,
  args...)

  id = get_id(rbspace)
  bs = rb_space_projection(op,rbspace,args...)
  bt = get_basis_time(rbspace)
  RBSpaceUnsteady(id,bs,bt)
end

function rb_space_projection(
  op::RBVariable,
  rbspace::RBSpace,
  args...)

  rb_space_projection(op,get_basis_space(rbspace),args...)
end

function rb_space_projection(
  op::RBLinVariable,
  basis_space::Matrix{Float},
  args...)

  rbspace_row = get_rbspace_row(op)
  brow = get_basis_space(rbspace_row)
  brow'*basis_space
end

function rb_space_projection(
  op::RBBilinVariable,
  basis_space::Matrix{Float},
  findnz_map::Vector{Int})

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

function get_red_lu_factors(
  ::RBInfoSteady,
  rbspace::RBSpaceSteady,
  idx_space::Vector{Int})

  basis = get_basis_space(rbspace)
  get_red_lu_factors(basis,idx_space)
end

function get_red_lu_factors(info::RBInfoUnsteady,args...)
  get_red_lu_factors(Val(info.st_mdeim),args...)
end

function get_red_lu_factors(
  ::Val{false},
  rbspace::RBSpaceUnsteady,
  idx_st::NTuple{2,Vector{Int}})

  basis = get_basis_space(rbspace)
  get_red_lu_factors(basis,first(idx_st))
end

#= function get_red_lu_factors(
  ::Val{false},
  basis_st::NTuple{2,Matrix{Float}},
  idx_st::NTuple{2,Vector{Int}})

  get_red_lu_factors(first(basis_st),first(idx_st))
end =#

function get_red_lu_factors(
  ::Val{true},
  rbspace::RBSpaceUnsteady,
  idx_st::NTuple{2,Vector{Int}})

  basis = get_basis_space(rbspace),get_basis_time(rbspace)
  get_red_lu_factors(basis,idx_st)
end

#= function get_red_lu_factors(
  ::Val{true},
  basis_st::NTuple{2,Matrix{Float}},
  idx_st::NTuple{2,Vector{Int}})

  get_red_lu_factors(basis_st,idx_st)
end =#

function get_red_lu_factors(
  basis::Matrix{Float},
  idx::Vector{Int})

  basis_idx = basis[idx,:]
  lu(basis_idx)
end

function get_red_lu_factors(
  basis::NTuple{2,Matrix{Float}},
  idx::NTuple{2,Vector{Int}})

  bs,bt = basis
  idx_space,idx_time = idx
  bs_idx = bs[idx_space,:]
  bt_idx = bt[idx_time,:]
  bst_idx = kron(bt_idx,bs_idx)

  lu(bst_idx)
end

recast_in_full_dim(idx_tmp::Vector{Int},findnz_map::Vector{Int}) =
  findnz_map[idx_tmp]

recast_in_full_dim(idx_tmp::NTuple{2,Vector{Int}},findnz_map::Vector{Int}) =
  recast_in_full_dim(first(idx_tmp),findnz_map),last(idx_tmp)

function get_red_measure(
  op::RBSteadyVariable,
  idx::Vector{Int},
  meas::ProblemMeasures,
  field=:dΩ)

  m = getproperty(meas,field)
  get_red_measure(op,idx,m)
end

function get_red_measure(
  op::RBUnsteadyVariable,
  idx::NTuple{2,Vector{Int}},
  meas::ProblemMeasures,
  field=:dΩ)

  m = getproperty(meas,field)
  get_red_measure(op,first(idx),m)
end

function get_red_measure(
  op::RBVariable,
  idx::Vector{Int},
  meas::Measure)

  get_red_measure(op,idx,get_triangulation(meas))
end

function get_red_measure(
  op::RBVariable,
  idx::Vector{Int},
  trian::Triangulation)

  el = find_mesh_elements(op,idx,trian)
  red_trian = view(trian,el)
  Measure(red_trian,get_degree(get_test(op)))
end

function find_mesh_elements(
  op::RBVariable,
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

recast_in_mat_form(::RBLinVariable,idx_tmp::Vector{Int}) = idx_tmp

function recast_in_mat_form(op::RBBilinVariable,idx_tmp::Vector{Int})
  Ns = get_Ns(get_rbspace_row(op))
  idx_space,_ = from_vec_to_mat_idx(idx_tmp,Ns)
  idx_space
end
