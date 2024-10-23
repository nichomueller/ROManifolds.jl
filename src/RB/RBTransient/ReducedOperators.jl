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

  red_trial,red_test = reduced_fe_space(solver,feop,s)
  odeop = get_algebraic_operator(feop)
  reduced_operator(solver,odeop,red_trial,red_test,s)
end

function RBSteady.reduced_operator(
  solver::RBSolver,
  odeop::ODEParamOperator,
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractArray)

  red_lhs,red_rhs = reduced_weak_form(solver,odeop,red_trial,red_test,s)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = map(get_domains,red_lhs)
  new_odeop = change_triangulation(odeop,trians_rhs,trians_lhs)
  GenericTransientRBOperator(new_odeop,red_trial,red_test,red_lhs,red_rhs)
end

function RBSteady.reduced_operator(
  solver::RBSolver,
  odeop::ODEParamOperator{LinearNonlinearParamODE},
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractArray)

  red_op_lin = reduced_operator(solver,get_linear_operator(odeop),red_trial,red_test,s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(odeop),red_trial,red_test,s)
  LinearNonlinearTransientRBOperator(red_op_lin,red_op_nlin)
end

abstract type TransientRBOperator{T} <: RBOperator{T} end

struct GenericTransientRBOperator{T} <: TransientRBOperator{T}
  op::ODEParamOperator{T}
  trial::RBSpace
  test::RBSpace
  lhs::TupOfAffineContribution
  rhs::AffineContribution
end

FESpaces.get_trial(op::GenericTransientRBOperator) = op.trial
FESpaces.get_test(op::GenericTransientRBOperator) = op.test
RBSteady.get_fe_trial(op::GenericTransientRBOperator) = get_trial(op.op)
RBSteady.get_fe_test(op::GenericTransientRBOperator) = get_test(op.op)

function allocate_odeparamcache(
  fesolver,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  @abstractmethod
end

function ParamSteady.allocate_paramcache(
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_paramcache(op.op,r,us)
end

function RBSteady.allocate_rbcache(
  fesolver,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  @abstractmethod
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

  lhs_rb = RBSteady.allocate_hypred_cache(op.lhs,r)
  return lhs_rb
end

function Algebra.residual!(
  cache,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  paramcache;
  kwargs...)

  b,b̂ = cache
  feb = fe_residual!(b,op,r,us,paramcache)
  inv_project!(b̂,op.rhs,feb)
end

function Algebra.jacobian!(
  cache,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  A,Â = cache
  feA = fe_jacobian!(A,op,r,us,ws,paramcache)
  inv_project!(Â,op.lhs,feA)
end

function RBSteady.fe_residual!(
  b,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  paramcache)

  red_params = 1:num_params(r)
  red_times = union_indices_time(op.rhs)
  red_pt_indices = range_2d(red_params,red_times,num_params(r))
  red_r = r[red_params,red_times]

  red_b,red_us,red_odeopcache = select_fe_quantities_at_indices(b,us,paramcache,vec(red_pt_indices))
  residual!(red_b,op.op,red_r,red_us,red_odeopcache)
  RBSteady.select_at_indices(red_b,op.rhs,red_pt_indices)
end

function RBSteady.fe_jacobian!(
  A,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  red_params = 1:num_params(r)
  red_times = union_indices_time(op.lhs)
  red_pt_indices = range_2d(red_params,red_times,num_params(r))
  red_r = r[red_params,red_times]

  red_A,red_us,red_odeopcache = select_fe_quantities_at_indices(A,us,paramcache,vec(red_pt_indices))
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

function RBSteady.get_fe_trial(op::LinearNonlinearTransientRBOperator)
  @check RBSteady.get_fe_trial(op.op_linear) === get_fe_trial(op.op_nonlinear)
  RBSteady.get_fe_trial(op.op_nonlinear)
end

function RBSteady.get_fe_test(op::LinearNonlinearTransientRBOperator)
  @check RBSteady.get_fe_test(op.op_linear) === RBSteady.get_fe_test(op.op_nonlinear)
  RBSteady.get_fe_test(op.op_nonlinear)
end

function allocate_odeparamcache(
  fesolver,
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  @abstractmethod
end

function ParamSteady.allocate_paramcache(
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  @notimplemented
end

function RBSteady.allocate_rbcache(
  fesolver,
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  cache_lin = RBSteady.allocate_rbcache(fesolver,get_linear_operator(op),r,us)
  cache_nlin = RBSteady.allocate_rbcache(fesolver,get_nonlinear_operator(op),r,us)
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
  paramcache;
  kwargs...)

  @notimplemented
end

function Algebra.jacobian!(
  cache,
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  @notimplemented
end

# Solve a POD-MDEIM problem

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

  y,odeparamcache = cache.fecache
  x̂,rbcache = cache.rbcache

  t = @timed solve!(x̂,solver,op,r,(y,),odeparamcache,rbcache)
  stats = CostTracker(t,nruns=num_params(r))

  return x̂,stats
end

function Algebra.solve!(
  x̂::AbstractVector,
  solver::RBSolver,
  op::TransientRBOperator,
  r::TransientRealization,
  statefe::Tuple{Vararg{AbstractVector}},
  odeparamcache,
  rbcache;
  kwargs...)

  fesolver = get_fe_solver(solver)

  sysslvr = fesolver.sysslvr
  odeslvrcache,paramcache = odeparamcache
  _...,sysslvrcache = odeslvrcache

  stageop = get_stage_operator(fesolver,op,r,statefe,odeparamcache,rbcache)
  solve!(x̂,sysslvr,stageop,sysslvrcache)
  return x̂
end

function Algebra.solve!(
  x̂::AbstractVector,
  solver::RBSolver,
  op::TransientRBOperator{LinearNonlinearParamODE},
  r::TransientRealization,
  statefe::Tuple{Vararg{AbstractVector}},
  odeparamcache,
  rbcache;
  kwargs...)

  fesolver = get_fe_solver(solver)
  nls = fesolver.sysslvr
  x = statefe[1]

  stageop = get_stage_operator(fesolver,op,r,statefe,odeparamcache,rbcache)
  solve!(x̂,nls,stageop,r,x;kwargs...)
  return x̂
end

# cache utils

function select_fe_space_at_indices(fs::FESpace,indices)
  @notimplemented
end

function select_fe_space_at_indices(fs::TrivialParamFESpace,indices)
  TrivialParamFESpace(fs.space,Val(length(indices)))
end

function select_fe_space_at_indices(fs::SingleFieldParamFESpace,indices)
  dvi = ConsecutiveParamArray(fs.dirichlet_values.data[:,indices])
  TrialParamFESpace(dvi,fs.space)
end

function select_slvrcache_at_indices(b::ConsecutiveParamArray,indices)
  ConsecutiveParamArray(b.data[:,indices])
end

function select_slvrcache_at_indices(A::ConsecutiveParamSparseMatrixCSC,indices)
  ConsecutiveParamSparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,A.data[:,indices])
end

function select_slvrcache_at_indices(A::BlockParamArray,indices)
  map(a -> select_slvrcache_at_indices(a,indices),blocks(A)) |> mortar
end

function select_slvrcache_at_indices(cache::ArrayContribution,indices)
  contribution(cache.trians) do trian
    select_slvrcache_at_indices(cache[trian],indices)
  end
end

function select_slvrcache_at_indices(cache::TupOfArrayContribution,indices)
  red_cache = ()
  for c in cache
    red_cache = (red_cache...,select_slvrcache_at_indices(c,indices))
  end
  return red_cache
end

function select_evalcache_at_indices(us::Tuple{Vararg{ConsecutiveParamVector}},paramcache,indices)
  @unpack trial,ptrial,feop_cache,const_forms = paramcache
  new_xhF = ()
  new_trial = ()
  for i = eachindex(trial)
    new_trial = (new_trial...,select_fe_space_at_indices(trial[i],indices))
    new_XhF_i = ConsecutiveParamArray(us[i].data[:,indices])
    new_xhF = (new_xhF...,new_XhF_i)
  end
  new_odeopcache = ParamCache(new_trial,ptrial,feop_cache,const_forms)
  return new_xhF,new_odeopcache
end

function select_evalcache_at_indices(us::Tuple{Vararg{BlockConsecutiveParamVector}},paramcache,indices)
  @unpack trial,ptrial,feop_cache,const_forms = paramcache
  new_xhF = ()
  new_trial = ()
  for i = eachindex(trial)
    spacei = trial[i]
    VT = spacei.vector_type
    style = spacei.multi_field_style
    spacesi = [select_fe_space_at_indices(spaceij,indices) for spaceij in spacei]
    new_trial = (new_trial...,MultiFieldFESpace(VT,spacesi,style))
    new_XhF_i = mortar([ConsecutiveParamArray(us_i.data[:,indices]) for us_i in blocks(us[i])])
    new_xhF = (new_xhF...,new_XhF_i)
  end
  new_odeopcache = ParamCache(new_trial,ptrial,feop_cache,const_forms)
  return new_xhF,new_odeopcache
end

function select_fe_quantities_at_indices(cache,us,paramcache,indices)
  # returns the cache in the appropriate time-parameter locations
  red_cache = select_slvrcache_at_indices(cache,indices)
  # does the same with the stage variable `us` and the ode cache `paramcache`
  red_us,red_odeopcache = select_evalcache_at_indices(us,paramcache,indices)

  return red_cache,red_us,red_odeopcache
end

get_entry(s::ConsecutiveParamVector,is,ipt) = get_all_data(s)[is,ipt]
get_entry(s::ParamSparseMatrix,is,ipt) = param_getindex(s,ipt)[is]

function RBSteady.select_at_indices(
  ::TransientHyperReduction,
  a::AbstractParamArray,
  ids_space,ids_time,ids_param)

  @check length(ids_space) == length(ids_time)
  entries = zeros(eltype2(a),length(ids_space),length(ids_param))
  @inbounds for ip = 1:length(ids_param)
    for (i,(is,it)) in enumerate(zip(ids_space,ids_time))
      ipt = ip+(it-1)*length(ids_param)
      v = get_entry(a,is,ipt)
      entries[i,ip] = v
    end
  end
  return ConsecutiveParamArray(entries)
end

function RBSteady.select_at_indices(
  ::TransientHyperReduction{<:TransientReduction},
  a::AbstractParamArray,
  ids_space,ids_time,ids_param)

  entries = zeros(eltype2(a),length(ids_space),length(ids_time),length(ids_param))
  @inbounds for ip = 1:length(ids_param)
    for (i,it) in enumerate(ids_time)
      ipt = ip+(it-1)*length(ids_param)
      v = get_entry(a,ids_space,ipt)
      entries[:,i,ip] = v
    end
  end
  return ConsecutiveParamArray(entries)
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
