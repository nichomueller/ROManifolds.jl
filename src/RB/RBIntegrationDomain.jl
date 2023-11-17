struct RBIntegrationCache{T,N}
  x::NonaffinePTArray{Vector{T}}
  q::PTArray{Array{T,N}}
  ode_cache::PTODECacheType
  function RBIntegrationCache(
    x::NonaffinePTArray{Vector{T}},
    q::PTArray{Array{T,N}},
    ode_cache::PTODECacheType) where {T,N}
    new{T,N}(x,q,ode_cache)
  end
end

function allocate_algebraic_cache(
  ::Val{1},
  odeop::PODEOperator,
  x::NonaffinePTArray,
  μ::Table,
  t::Real,
  ode_cache::PTODECacheType)

  b = allocate_residual(odeop,μ,t,x,ode_cache)
  RBIntegrationCache(x,b,ode_cache)
end

function allocate_algebraic_cache(
  ::Val{2},
  odeop::PODEOperator,
  x::NonaffinePTArray,
  μ::Table,
  t::Real,
  ode_cache::PTODECacheType)

  A = allocate_jacobian(odeop,μ,t,x,ode_cache)
  RBIntegrationCache(x,A,ode_cache)
end

function update_cache!(cache::RBIntegrationCache,x::NonaffinePTArray)
  xold = cache.x
  @. xold = x
  cache
end

struct RBIntegrationDomain{T,N}
  feop::PTFEOperator
  meas::Measure
  idx_space::Vector{Int}
  times::Vector{<:Real}
  idx_time::Vector{Int}
  cache::RBIntegrationCache{T,N}
  function RBIntegrationDomain(
    feop::PTFEOperator,
    meas::Measure,
    idx_space::Vector{Int},
    times::Vector{<:Real},
    idx_time::Vector{Int},
    cache::RBIntegrationCache{T,N}) where {T,N}
    new{T,N}(feop,meas,idx_space,times,idx_time,cache)
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

  x = NonaffinePTArray([zeros(T,red_nfree)])
  μ = testitem(op.μ)
  t = testitem(times)
  red_ode_cache = allocate_cache(red_odeop,μ,t)
  red_cache = allocate_algebraic_cache(Val(N),red_odeop,μ,t,x,red_ode_cache)

  RBIntegrationDomain(red_feop,red_meas,recast_idx_space,red_times,time_to_parent_time,red_cache)
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

function update_cache!(rbintd::RBIntegrationDomain,x::NonaffinePTArray,μ::Table)
  xidx = selectidx(x,rbintd;nparams=length(μ))
  cache = rbintd.cache
  update_cache!(cache,xidx)
  rbintd
end
