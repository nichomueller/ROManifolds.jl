function RBSteady.reduced_operator(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  args...;
  dir=datadir(),
  kwargs...)

  fesnaps,festats = solution_snapshots(solver,feop,args...;kwargs...)
  save(fesnaps,dir)
  save(festats,dir;label="fe")
  reduced_operator(solver,feop,fesnaps)
end

function RBSteady.reduced_operator(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  s::AbstractArray)

  red_test,red_trial = reduced_fe_space(solver,feop,s)
  odeop = get_algebraic_operator(feop)
  reduced_operator(solver,odeop,red_test,red_trial,s)
end

function RBSteady.reduced_operator(
  solver::RBSolver,
  odeop::ODEParamOperator,
  red_test::FESubspace,
  red_trial::FESubspace,
  s::AbstractArray)

  red_lhs,red_rhs = reduced_weak_form(solver,odeop,red_test,red_trial,s)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = map(get_domains,red_lhs)
  new_odeop = change_triangulation(odeop,trians_rhs,trians_lhs)
  GenericTransientRBOperator(new_odeop,red_test,red_trial,red_lhs,red_rhs)
end

function RBSteady.reduced_operator(
  solver::RBSolver,
  odeop::ODEParamOperator{LinearNonlinearParamODE},
  red_test::FESubspace,
  red_trial::FESubspace,
  s::AbstractArray)

  red_op_lin = reduced_operator(solver,get_linear_operator(odeop),red_test,red_trial,s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(odeop),red_test,red_trial,s)
  LinearNonlinearTransientRBOperator(red_op_lin,red_op_nlin)
end

abstract type TransientRBOperator{T} <: ODEParamOperatorWithTrian{T} end

RBSteady.allocate_rbcache(op::TransientRBOperator,r::TransientRealization) = @abstractmethod

function ODEs.update_odeopcache!(
  odeopcache,
  op::TransientRBOperator,
  r::TransientRealization)

  msg = "A space-time ROM is time-independent, thus the cache can be correctly
    initialized before the call to solve!"
  @notimplemented msg
end

struct GenericTransientRBOperator{T} <: TransientRBOperator{T}
  op::ODEParamOperatorWithTrian{T}
  trial::FESubspace
  test::FESubspace
  lhs::TupOfAffineContribution
  rhs::AffineContribution
end

FESpaces.get_trial(op::GenericTransientRBOperator) = op.trial
FESpaces.get_test(op::GenericTransientRBOperator) = op.test
ParamDataStructures.realization(op::GenericTransientRBOperator;kwargs...) = realization(op.op;kwargs...)
ParamSteady.get_fe_operator(op::GenericTransientRBOperator) = ParamSteady.get_fe_operator(op.op)
ParamSteady.get_vector_index_map(op::GenericTransientRBOperator) = get_vector_index_map(op.op)
ParamSteady.get_matrix_index_map(op::GenericTransientRBOperator) = get_matrix_index_map(op.op)
RBSteady.get_fe_trial(op::GenericTransientRBOperator) = get_trial(op.op)
RBSteady.get_fe_test(op::GenericTransientRBOperator) = get_test(op.op)

function ODEs.allocate_odecache(
  fesolver,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  @abstractmethod
end

function ODEs.allocate_odeopcache(
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_odeopcache(op.op,r,us)
end

function RBSteady.allocate_rbcache(
  op::GenericTransientRBOperator,
  r::TransientRealization)

  rb_lhs_cache = allocate_jacobian(op,r)
  rb_rhs_cache = allocate_residual(op,r)
  syscache = (rb_lhs_cache,rb_rhs_cache)
  rbopcache = get_trial(op)(r)
  return syscache,rbopcache
end

function Algebra.allocate_residual(
  op::GenericTransientRBOperator,
  r::TransientRealization,
  args...)

  rhs_rb = RBSteady.allocate_hypred_cache(op.rhs,r)
  return rhs_rb
end

function Algebra.allocate_jacobian(
  op::GenericTransientRBOperator,
  r::TransientRealization,
  args...)

  lhs_coeff = map(lhs -> RBSteady.allocate_coefficient(lhs,r),op.lhs)
  lhs_hypred = RBSteady.allocate_hyper_reduction(first(op.lhs),r)
  lhs_rb = (lhs_coeff,lhs_hypred)
  return lhs_rb
end

function Algebra.residual!(
  cache,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache;
  kwargs...)

  b,b̂ = cache
  feb = fe_residual!(b,op,r,us,odeopcache)
  inv_project!(b̂,op.rhs,feb)
end

function Algebra.jacobian!(
  cache,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  A,Â = cache
  feA = fe_jacobian!(A,op,r,us,ws,odeopcache)
  inv_project!(Â,op.lhs,feA)
end

function RBSteady.fe_residual!(
  b,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  red_params = 1:num_params(r)
  red_times = union_indices_time(op.rhs)
  red_pt_indices = range_2d(red_params,red_times,num_params(r))
  red_r = r[red_params,red_times]

  red_b,red_us,red_odeopcache = select_fe_quantities_at_indices(b,us,odeopcache,vec(red_pt_indices))
  residual!(red_b,op.op,red_r,red_us,red_odeopcache)
  RBSteady.select_at_indices(red_b,op.rhs,red_pt_indices)
end

function RBSteady.fe_jacobian!(
  A,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  red_params = 1:num_params(r)
  red_times = union_indices_time(op.lhs)
  red_pt_indices = range_2d(red_params,red_times,num_params(r))
  red_r = r[red_params,red_times]

  red_A,red_us,red_odeopcache = select_fe_quantities_at_indices(A,us,odeopcache,vec(red_pt_indices))
  jacobian!(red_A,op.op,red_r,red_us,ws,red_odeopcache)
  map(red_A,op.lhs) do red_A,lhs
    RBSteady.select_at_indices(red_A,lhs,red_pt_indices)
  end
end

struct LinearNonlinearTransientRBOperator <: TransientRBOperator{LinearNonlinearParamODE}
  op_linear::GenericTransientRBOperator{<:AbstractLinearParamODE}
  op_nonlinear::GenericTransientRBOperator{NonlinearParamODE}
end

ParamSteady.get_linear_operator(op::LinearNonlinearTransientRBOperator) = op.op_linear
ParamSteady.get_nonlinear_operator(op::LinearNonlinearTransientRBOperator) = op.op_nonlinear

function FESpaces.get_test(op::LinearNonlinearTransientRBOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_nonlinear)
end

function FESpaces.get_trial(op::LinearNonlinearTransientRBOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_nonlinear)
end

function ParamDataStructures.realization(op::LinearNonlinearTransientRBOperator;kwargs...)
  realization(op.op_nonlinear;kwargs...)
end

function ParamSteady.get_fe_operator(op::LinearNonlinearTransientRBOperator)
  join_operators(ParamSteady.get_fe_operator(op.op_linear),ParamSteady.get_fe_operator(op.op_nonlinear))
end

function ParamSteady.get_vector_index_map(op::LinearNonlinearTransientRBOperator)
  @check all(get_vector_index_map(op.op_linear) .== get_vector_index_map(op.op_nonlinear))
  get_vector_index_map(op.op_linear)
end

function ParamSteady.get_matrix_index_map(op::LinearNonlinearTransientRBOperator)
  @check all(get_matrix_index_map(op.op_linear) .== get_matrix_index_map(op.op_nonlinear))
  get_matrix_index_map(op.op_linear)
end

function RBSteady.get_fe_trial(op::LinearNonlinearTransientRBOperator)
  @check RBSteady.get_fe_trial(op.op_linear) === get_fe_trial(op.op_nonlinear)
  RBSteady.get_fe_trial(op.op_nonlinear)
end

function RBSteady.get_fe_test(op::LinearNonlinearTransientRBOperator)
  @check RBSteady.get_fe_test(op.op_linear) === RBSteady.get_fe_test(op.op_nonlinear)
  RBSteady.get_fe_test(op.op_nonlinear)
end

function ODEs.allocate_odecache(
  fesolver,
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  @abstractmethod
end

function ODEs.allocate_odeopcache(
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  @notimplemented
end

function RBSteady.allocate_rbcache(
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization)

  cache_lin = RBSteady.allocate_rbcache(get_linear_operator(op),r)
  cache_nlin = RBSteady.allocate_rbcache(get_nonlinear_operator(op),r)
  return (cache_lin,cache_nlin)
end

function Algebra.allocate_residual(
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  args...)

  @notimplemented
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  args...)

  @notimplemented
end

function Algebra.residual!(
  cache,
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache;
  kwargs...)

  @notimplemented
end

function Algebra.jacobian!(
  cache,
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  @notimplemented
end

# Solve a POD-MDEIM problem

function RBSteady.init_online_cache!(
  solver::RBSolver,
  op::TransientRBOperator,
  r::TransientRealization)

  fe_trial = get_fe_trial(op)(r)
  trial = get_trial(op)(r)
  y = zero_free_values(fe_trial)
  x̂ = zero_free_values(trial)

  fesolver = get_fe_solver(solver)
  odecache = allocate_odecache(fesolver,op,r,(y,))
  rbcache = RBSteady.allocate_rbcache(op,r)

  cache = solver.cache
  cache.fecache = (y,odecache)
  cache.rbcache = (x̂,rbcache)
  return
end

function RBSteady.online_cache!(
  solver::RBSolver,
  op::TransientRBOperator,
  r::TransientRealization)

  cache = solver.cache
  y, = cache.fecache
  x̂, = cache.rbcache
  if param_length(r) != param_length(y)
    RBSteady.init_online_cache!(solver,op,r)
  end
  return
end

function Algebra.solve(
  solver::RBSolver,
  op::TransientRBOperator{NonlinearParamODE},
  r::AbstractRealization)

  @notimplemented "Split affine from nonlinear operator when running the RB solve"
end

function Algebra.solve(
  solver::RBSolver,
  op::TransientRBOperator,
  r::TransientRealization;
  kwargs...)

  cache = solver.cache
  if isnothing(cache.fecache) || isnothing(cache.rbcache)
    RBSteady.init_online_cache!(solver,op,r)
  else
    RBSteady.online_cache!(solver,op,r)
  end
  solve!(cache,solver,op,r;kwargs...)
end

function Algebra.solve!(
  cache,
  solver::RBSolver,
  op::TransientRBOperator,
  r::TransientRealization;
  kwargs...)

  y,odecache = cache.fecache
  x̂,rbcache = cache.rbcache

  t = @timed solve!(x̂,solver,op,r,(y,),odecache,rbcache)
  stats = CostTracker(t,nruns=num_params(r))

  return x̂,stats
end

function Algebra.solve!(
  x̂::AbstractVector,
  solver::RBSolver,
  op::TransientRBOperator,
  r::TransientRealization,
  statefe::Tuple{Vararg{AbstractVector}},
  odecache,
  rbcache;
  kwargs...)

  fesolver = get_fe_solver(solver)

  sysslvr = fesolver.sysslvr
  odeslvrcache,odeopcache = odecache
  _...,sysslvrcache = odeslvrcache

  stageop = get_stage_operator(fesolver,op,r,statefe,odecache,rbcache)
  solve!(x̂,sysslvr,stageop,sysslvrcache)
  return x̂
end

function Algebra.solve!(
  x̂::AbstractVector,
  solver::RBSolver,
  op::TransientRBOperator{LinearNonlinearParamODE},
  r::TransientRealization,
  statefe::Tuple{Vararg{AbstractVector}},
  odecache,
  rbcache;
  kwargs...)

  fesolver = get_fe_solver(solver)
  nls = fesolver.sysslvr
  x = statefe[1]

  stageop = get_stage_operator(fesolver,op,r,statefe,odecache,rbcache)
  solve!(x̂,nls,stageop,r,x;kwargs...)
  return x̂
end

# cache utils

function RBSteady.select_evalcache_at_indices(us::Tuple{Vararg{ConsecutiveArrayOfArrays}},odeopcache,indices)
  @unpack Us,Uts,tfeopcache,const_forms = odeopcache
  new_xhF = ()
  new_Us = ()
  for i = eachindex(Us)
    new_Us = (new_Us...,RBSteady.select_fe_space_at_indices(Us[i],indices))
    new_XhF_i = ConsecutiveArrayOfArrays(us[i].data[:,indices])
    new_xhF = (new_xhF...,new_XhF_i)
  end
  new_odeopcache = ODEOpFromTFEOpCache(new_Us,Uts,tfeopcache,const_forms)
  return new_xhF,new_odeopcache
end

function RBSteady.select_evalcache_at_indices(us::Tuple{Vararg{BlockVectorOfVectors}},odeopcache,indices)
  @unpack Us,Uts,tfeopcache,const_forms = odeopcache
  new_xhF = ()
  new_Us = ()
  for i = eachindex(Us)
    spacei = Us[i]
    VT = spacei.vector_type
    style = spacei.multi_field_style
    spacesi = [RBSteady.select_fe_space_at_indices(spaceij,indices) for spaceij in spacei]
    new_Us = (new_Us...,MultiFieldFESpace(VT,spacesi,style))
    new_XhF_i = mortar([ConsecutiveArrayOfArrays(us_i.data[:,indices]) for us_i in blocks(us[i])])
    new_xhF = (new_xhF...,new_XhF_i)
  end
  new_odeopcache = ODEOpFromTFEOpCache(new_Us,Uts,tfeopcache,const_forms)
  return new_xhF,new_odeopcache
end

function RBSteady.select_slvrcache_at_indices(cache::TupOfArrayContribution,indices)
  red_cache = ()
  for c in cache
    red_cache = (red_cache...,RBSteady.select_slvrcache_at_indices(c,indices))
  end
  return red_cache
end

function select_fe_quantities_at_indices(cache,us,odeopcache,indices)
  # returns the cache in the appropriate time-parameter locations
  red_cache = RBSteady.select_slvrcache_at_indices(cache,indices)
  # does the same with the stage variable `us` and the ode cache `odeopcache`
  red_us,red_odeopcache = RBSteady.select_evalcache_at_indices(us,odeopcache,indices)

  return red_cache,red_us,red_odeopcache
end

get_entry(s::AbstractParamArray,is,ipt) = consecutive_getindex(s,is,ipt)
get_entry(s::ParamSparseMatrix,is,ipt) = param_getindex(s,ipt)[is]

function RBSteady.select_at_indices(
  ::TransientHyperReduction,
  s::AbstractParamArray,
  ids_space,ids_time,ids_param)

  @check length(ids_space) == length(ids_time)
  entry = zeros(eltype2(a),length(ids_space))
  entries = array_of_consecutive_arrays(entry,length(ids_param))
  @inbounds for ip = param_eachindex(entries)
    for (i,(is,it)) in enumerate(zip(ids_space,ids_time))
      ipt = ip+(it-1)*length(ids_param)
      v = get_entry(a,is,ipt)
      consecutive_setindex!(entries,v,i,ip)
    end
  end
  return entries
end

function RBSteady.select_at_indices(
  ::TransientHyperReduction{<:TransientReduction},
  a::AbstractParamArray,
  ids_space,ids_time,ids_param)

  entry = zeros(eltype2(a),length(ids_space),length(ids_time))
  entries = array_of_consecutive_arrays(entry,length(ids_param))
  @inbounds for ip = param_eachindex(entries)
    for (i,it) in enumerate(ids_time)
      ipt = ip+(it-1)*length(ids_param)
      v = get_entry(a,ids_space,ipt)
      consecutive_setindex!(entries,v,:,i,ip)
    end
  end
  return entries
end

function RBSteady.select_at_indices(s::AbstractArray,a::TransientHyperReduction,indices::Range2D)
  ids_space = get_indices_space(a)
  ids_param = indices.axis1
  common_ids_time = indices.axis2
  domain_time = get_integration_domain_time(a)
  ids_time = RBSteady.ordered_common_locations(domain_time,common_ids_time)
  RBSteady.select_at_indices(a,s,ids_space,ids_time,ids_param)
end

function RBSteady.select_at_indices(
  s::ArrayContribution,a::AffineContribution,indices)
  contribution(s.trians) do trian
    RBSteady.select_at_indices(s[trian],a[trian],indices)
  end
end
