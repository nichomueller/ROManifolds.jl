function allocate_algebraic_cache(
  ::Val{1},
  odeop::PODEOperator,
  μ::Table,
  t::Vector{<:Real},
  x::NonaffinePTArray,
  ode_cache,
  meas::Measure)

  allocate_residual(odeop,μ,t,x,ode_cache)
end

function allocate_algebraic_cache(
  ::Val{2},
  odeop::PODEOperator,
  μ::Table,
  t::Real,
  x::NonaffinePTArray,
  ode_cache)

  allocate_jacobian(odeop,μ,t,x,ode_cache)
end

struct RBIntegrationDomain{T,N}
  op::PTAlgebraicOperator
  meas::Measure
  idx_space::Vector{Int}
  idx_time::Vector{Int}
  cache::PTArray{<:AbstractArray{T,N}}
  function RBIntegrationDomain(
    op::PTAlgebraicOperator,
    meas::Measure,
    idx_space::Vector{Int},
    idx_time::Vector{Int},
    cache::PTArray{<:AbstractArray{T,N}}) where {T,N}
    new{T,N}(op,meas,idx_space,idx_time,cache)
  end
end

function RBIntegrationDomain(
  op::PTAlgebraicOperator,
  nzm::NnzMatrix,
  trian::TriangulationWithTags,
  idx_space::Vector{Int},
  idx_time::Vector{Int},
  ::NTuple{N,RBSpace{T}}) where {T,N}

  feop = op.odeop.feop
  model = get_background_model(trian)
  times = op.tθ

  recast_idx_space = recast_idx(nzm,idx_space)
  recast_idx_space_rows,_ = vec_to_mat_idx(recast_idx_space,nzm.nrows)
  cell_dof_ids = get_cell_dof_ids(test,trian)
  red_cells = get_reduced_cells(recast_idx_space_rows,cell_dof_ids)

  red_model = DiscreteModelPortion(model,red_cells)
  red_trian = if isa(trian,BoundaryTriangulationWithTags)
    BoundaryTriangulationWithTags(red_model,tags=get_tags(trian),id=get_id(trian))
  else
    TriangulationWithTags(red_model,tags=get_tags(trian),id=get_id(trian))
  end
  red_meas = Measure(red_trian,2*get_order(test))
  red_feop = reduce_fe_operator(feop,red_model)
  red_test = get_test(red_feop)
  red_nfree = num_free_dofs(red_test)
  red_odeop = get_algebraic_operator(red_feop)
  red_times = times[idx_time]
  time_to_parent_time = findall(x->x in red_times,times)

  x = NonaffinePTArray([zeros(T,red_nfree) for _ = 1:length(op.μ)*length(times)])
  red_ode_cache = allocate_cache(red_odeop,op.μ,times)
  red_cache = allocate_algebraic_cache(Val(N),red_odeop,op.μ,times,x,red_ode_cache,red_meas)

  red_op = get_ptoperator(red_odeop,op.μ,red_times,op.dtθ,u0,ode_cache,vθ)

  RBIntegrationDomain(red_op,red_meas,recast_idx_space,time_to_parent_time,red_cache)
end

function get_space_row_idx(op::PTAlgebraicOperator,rbintd::RBIntegrationDomain)
  feop = op.odeop.feop
  test = get_test(feop)
  nrows = num_free_dofs(test)
  idx_space_row, = vec_to_mat_idx(rbintd.idx_space,nrows)
  return idx_space_row
end

function selectidx(a::PTArray,rbintd::RBIntegrationDomain;kwargs...)
  idx_space = rbintd.idx_space
  idx_time = rbintd.idx_time
  selectidx(a,idx_space,idx_time;kwargs...)
end

function _update_solution!(x::NonaffinePTArray,x0::NonaffinePTArray)
  @. x = x0
  x
end

function _update_parameter!(μ::Table,μ0::Table)
  @. μ = μ0
  μ
end

function _update_cache!(cache::NonaffinePTArray,n::Int)
  cache0 = NonaffinePTArray(cache[1:n])
  @. cache = cache0
  cache
end

function _update_cache!(cache::AffinePTArray,n::Int)
  cache0 = AffinePTArray(cache.array,n)
  @. cache = cache0
  cache
end

function update_reduced_operator!(rbintd::RBIntegrationDomain,x::NonaffinePTArray,μ::Table)
  op = rbintd.op

  xidx = selectidx(x,rbintd;nparams=length(μ))
  _update_solution!(op.u0,xidx)

  if op.μ != μ
    _update_parameter!(op.μ,μ)
  end

  N = length(rbintd.cache)
  n = length(μ)*length(op.tθ)
  @assert N ≥ n
  if length(rbintd.cache) < N
    _update_cache!(op.cache,n)
  end

  rbintd
end
