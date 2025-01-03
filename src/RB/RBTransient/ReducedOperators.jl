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
  new_odeop = change_domains(odeop,trians_rhs,trians_lhs)
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

"""
    abstract type TransientRBOperator{O} <: ODEParamOperator{O,SplitDomains} end

Type representing reduced algebraic operators used within a reduced order modelling
framework in transient applications. A TransientRBOperator should contain the
following information:

- a reduced test and trial space, computed according to [`reduced_fe_space`](@ref)
- a hyper-reduced residual and jacobian, computed according to [`reduced_weak_form`](@ref)

Subtypes:

- [`GenericTransientRBOperator`](@ref)
- [`LinearNonlinearTransientRBOperator`](@ref)
"""
abstract type TransientRBOperator{O} <: ODEParamOperator{O,SplitDomains} end

function RBSteady.allocate_rbcache(fesolver::ODESolver,op::RBOperator,args...)
  @abstractmethod
end

"""
    struct GenericTransientRBOperator{O} <: TransientRBOperator{O}
      op::ODEParamOperator{O}
      trial::RBSpace
      test::RBSpace
      lhs::TupOfAffineContribution
      rhs::AffineContribution
    end

Transient counterpart of a [`GenericRBOperator`] used in steady problems. Fields:

- `op`: underlying high dimensional FE operator
- `trial`: reduced trial space
- `test`: reduced trial space
- `lhs`: hyper-reduced left hand side
- `rhs`: hyper-reduced right hand side

The major difference with respect to the steady setting is that the `lhs` is a
n-tuple of [`AffineContribution`](@ref), where n is the maximum order of the
time derivatives
"""
struct GenericTransientRBOperator{O} <: TransientRBOperator{O}
  op::ODEParamOperator{O}
  trial::RBSpace
  test::RBSpace
  lhs::TupOfAffineContribution
  rhs::AffineContribution
end

FESpaces.get_trial(op::GenericTransientRBOperator) = op.trial
FESpaces.get_test(op::GenericTransientRBOperator) = op.test
RBSteady.get_fe_trial(op::GenericTransientRBOperator) = get_trial(op.op)
RBSteady.get_fe_test(op::GenericTransientRBOperator) = get_test(op.op)

function Algebra.allocate_residual(
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  rbcache::RBCache)

  rbcache.b
end

function Algebra.allocate_jacobian(
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  rbcache::RBCache)

  rbcache.A
end

function Algebra.residual!(
  cache::HRParamArray,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  rbcache::RBCache)

  b = cache.fe_quantity
  paramcache = rbcache.paramcache

  feb = fe_residual!(b,op,r,us,paramcache)
  inv_project!(cache,op.rhs,feb)
end

function Algebra.residual!(
  cache::HRParamArray,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{RBParamVector}},
  rbcache::RBCache)

  ufe = ()
  for u in us
    # inv_project!(u.fe_data,rbcache.trial,u.data) this should already be executed
    ufe = (ufe...,u.fe_data)
  end
  residual!(cache,op,r,ufe,rbcache)
end

function Algebra.jacobian!(
  cache::HRParamArray,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  rbcache::RBCache)

  A = cache.fe_quantity
  paramcache = rbcache.paramcache

  feA = fe_jacobian!(A,op,r,us,ws,paramcache)
  inv_project!(cache,op.lhs,feA)
end

function Algebra.jacobian!(
  cache::HRParamArray,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{RBParamVector}},
  ws::Tuple{Vararg{Real}},
  rbcache::RBCache)

  ufe = ()
  for u in us
    # inv_project!(u.fe_data,rbcache.trial,u.data) this should already be executed
    ufe = (ufe...,u.fe_data)
  end
  jacobian!(cache,op,r,ufe,ws,rbcache)
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

"""
    struct LinearNonlinearTransientRBOperator <: TransientRBOperator{LinearNonlinearParamODE}
      op_linear::GenericTransientRBOperator{LinearParamODE}
      op_nonlinear::GenericTransientRBOperator{NonlinearParamODE}
    end

Extends the concept of [`GenericTransientRBOperator`](@ref) to accommodate the linear/nonlinear
splitting of terms in nonlinear applications
"""
struct LinearNonlinearTransientRBOperator <: TransientRBOperator{LinearNonlinearParamODE}
  op_linear::GenericTransientRBOperator{LinearParamODE}
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

function Algebra.allocate_residual(
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  rbcache::LinearNonlinearRBCache)

  rbcache.rbcache.b
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  rbcache::LinearNonlinearRBCache)

  rbcache.rbcache.A
end

function Algebra.residual!(
  cache,
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  rbcache::LinearNonlinearRBCache)

  nlop = get_nonlinear_operator(op)
  A_lin = rbcache.A
  b_lin = rbcache.b
  rbcache_nlin = rbcache.rbcache

  b_nlin = residual!(cache,nlop,r,us,rbcache_nlin)
  axpy!(1.0,b_lin,b_nlin)
  mul!(b_nlin,A_lin,us[end],true,true)

  return b_nlin
end

function Algebra.jacobian!(
  cache,
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  rbcache::LinearNonlinearRBCache)

  nlop = get_nonlinear_operator(op)
  A_lin = rbcache.A
  rbcache_nlin = rbcache.rbcache

  A_nlin = jacobian!(cache,nlop,r,us,ws,rbcache_nlin)
  axpy!(1.0,A_lin,A_nlin)

  return A_nlin
end

function Algebra.solve(solver::RBSolver,op::TransientRBOperator,r::TransientRealization)
  fe_trial = get_fe_trial(op)(r)
  x = zero_free_values(fe_trial)
  solve(solver,op,r,x)
end

function Algebra.solve(
  solver::RBSolver,
  op::TransientRBOperator{NonlinearParamODE},
  r::TransientRealization,
  x::AbstractParamVector)

  @notimplemented "Split affine from nonlinear operator when running the RB solve"
end

function Algebra.solve(
  solver::RBSolver,
  op::TransientRBOperator,
  r::TransientRealization,
  x::AbstractParamVector)

  fesolver = get_fe_solver(solver)
  trial = get_trial(op)(r)
  x̂ = zero_free_values(trial)

  rbcache = allocate_rbcache(fesolver,op,r,x)

  t = @timed solve!(x̂,fesolver,op,r,x,rbcache)
  stats = CostTracker(t,nruns=num_params(r),name="RB solver")

  return x̂,stats
end

# cache utils

function select_fe_space_at_indices(fs::FESpace,indices)
  @notimplemented
end

function select_fe_space_at_indices(fs::TrivialParamFESpace,indices)
  TrivialParamFESpace(fs.space,length(indices))
end

function select_fe_space_at_indices(fs::TrialParamFESpace,indices)
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
  @unpack trial,ptrial,feop_cache = paramcache
  new_xhF = ()
  new_trial = ()
  for i = eachindex(trial)
    new_trial = (new_trial...,select_fe_space_at_indices(trial[i],indices))
    new_XhF_i = ConsecutiveParamArray(us[i].data[:,indices])
    new_xhF = (new_xhF...,new_XhF_i)
  end
  new_odeopcache = ParamOpCache(new_trial,ptrial,feop_cache)
  return new_xhF,new_odeopcache
end

function select_evalcache_at_indices(us::Tuple{Vararg{BlockConsecutiveParamVector}},paramcache,indices)
  @unpack trial,ptrial,feop_cache = paramcache
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
  new_odeopcache = ParamOpCache(new_trial,ptrial,feop_cache)
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
