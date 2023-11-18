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
  op::PTOperator
  meas::Measure
  idx_space::Vector{Int}
  idx_time::Vector{Int}
  cache::PTArray{AbstractArray{T,N}}
  function RBIntegrationDomain(
    op::PTOperator,
    meas::Measure,
    idx_space::Vector{Int},
    idx_time::Vector{Int},
    cache::PTArray{<:AbstractArray{T,N}}) where {T,N}
    new{T,N}(op,meas,idx_space,idx_time,cache)
  end
end

function RBIntegrationDomain(
  op::PTOperator,
  nzm::NnzMatrix,
  trian::TriangulationWithTags,
  idx_space::Vector{Int},
  idx_time::Vector{Int},
  ::NTuple{N,RBSpace{T}};
  nparams::Int=10) where {T,N}

  feop = op.odeop.feop
  model = get_background_model(trian)
  μ = realization(feop,nparams)
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

  x = NonaffinePTArray([zeros(T,red_nfree) for _ = 1:length(μ)*length(red_times)])
  red_ode_cache = allocate_cache(red_odeop,μ,red_times)
  red_cache = allocate_algebraic_cache(Val(N),red_odeop,μ,red_times,x,red_ode_cache,red_meas)

  red_op = get_ptoperator(red_odeop,μ,red_times,op.dtθ,x,ode_cache,x)

  RBIntegrationDomain(red_op,red_meas,recast_idx_space,time_to_parent_time,red_cache)
end

function get_space_row_idx(op::PTOperator,dom::RBIntegrationDomain)
  feop = op.odeop.feop
  test = get_test(feop)
  nrows = num_free_dofs(test)
  idx_space_row, = vec_to_mat_idx(dom.idx_space,nrows)
  return idx_space_row
end

function selectidx(a::PTArray,dom::RBIntegrationDomain;kwargs...)
  idx_space = dom.idx_space
  idx_time = dom.idx_time
  selectidx(a,idx_space,idx_time;kwargs...)
end

# selectidx(x,dom;nparams=length(μ))
function jacobian!(cache,dom::Vector{<:RBIntegrationDomain},x::PTArray,μ::Table,i=1)
  meas = map(get_measure,dom)
  idx_space = map(get_idx_space,dom)
  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobian!(A,op.odeop,op.μ,op.tθ,(vθ,vθ),i,(1.0,1/op.dtθ)[i],op.ode_cache,args...)
end

function jacobian!(
  cache,
  feop,
  ::PTArray,
  i::Int,
  args...)

  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobian!(A,op.odeop,op.μ,op.tθ,(vθ,vθ),i,(1.0,1/op.dtθ)[i],op.ode_cache,args...)
end

function residual!(
  cache,
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  uh::S,
  fecache,
  meas::Vector{Measure},
  idx::Vector{Vector{Int}}) where {T,S}

  b,Mcache = cache
  V = get_test(op)
  v = get_fe_basis(V)
  res = get_residual(op)
  dc = res(μ,t,uh,v,meas...)
  trian = get_domains(dc)
  ntrian = num_domains(dc)
  setsize!(Mcache,(length(idx),ntrian))
  M = Mcache.array
  Mvec = Vector{typeof(M)}(undef,ntrian)
  for (i,itrian) in enumerate(trian)
    vecdata = collect_cell_vector(V,dc,itrian)
    assemble_vector_add!(b,op.assem,vecdata)
    @inbounds for n = eachindex(b)
      M[:,n] = b[n][idx]
    end
    Mvec[i] = copy(M)
  end
  Mvec
end

function jacobian!(
  cache,
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  uh::S,
  i::Integer,
  γᵢ::Real,
  fecache,
  meas::Vector{Measure},
  idx::Vector{Vector{Int}}) where {T,S}

  A,Mcache = cache
  Uh = get_trial(op)(μ,t)
  V = get_test(op)
  u = get_trial_fe_basis(Uh)
  v = get_fe_basis(V)
  jac = get_jacobian(op)
  dc = γᵢ*jac[i](μ,t,uh,u,v,meas...)
  trian = get_domains(dc)
  ntrian = num_domains(dc)
  setsize!(Mcache,(length(idx),ntrian))
  M = Mcache.array
  Mvec = Vector{typeof(M)}(undef,ntrian)
  for (i,itrian) in enumerate(trian)
    matdata = collect_cell_matrix(Uh,V,dc,itrian)
    assemble_matrix_add!(A,op.assem,matdata)
    @inbounds for n = eachindex(A)
      M[:,n] = A[n][idx].nzval
    end
    Mvec[i] = copy(M)
  end
  Mvec
end
