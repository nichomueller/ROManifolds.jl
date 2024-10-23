function reduced_operator(
  solver::RBSolver,
  feop::ParamFEOperator,
  args...;
  dir=datadir(),
  kwargs...)

  fesnaps,festats = solution_snapshots(solver,feop,args...;kwargs...)
  save(fesnaps,dir)
  save(festats,dir;label="fe")
  reduced_operator(solver,feop,fesnaps)
end

function reduced_operator(
  solver::RBSolver,
  feop::ParamFEOperator,
  s)

  red_trial,red_test = reduced_fe_space(solver,feop,s)
  op = get_algebraic_operator(feop)
  reduced_operator(solver,op,red_trial,red_test,s)
end

function reduced_operator(
  solver::RBSolver,
  op::ParamOperator,
  red_trial::RBSpace,
  red_test::RBSpace,
  s)

  red_lhs,red_rhs = reduced_weak_form(solver,op,red_trial,red_test,s)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = get_domains(red_lhs)
  new_op = change_triangulation(op,trians_rhs,trians_lhs)
  GenericRBOperator(new_op,red_trial,red_test,red_lhs,red_rhs)
end

function reduced_operator(
  solver::RBSolver,
  op::ParamOperator{LinearNonlinearParamEq},
  red_trial::RBSpace,
  red_test::RBSpace,
  s)

  red_op_lin = reduced_operator(solver,get_linear_operator(op),red_trial,red_test,s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(op),red_trial,red_test,s)
  LinearNonlinearRBOperator(red_op_lin,red_op_nlin)
end

struct RBCache
  A::HypRedCache
  b::HypRedCache
  trial::RBSpace
  paramcache::ParamCache
end

struct LinearNonlinearRBCache
  rbcache::RBCache
  A_lin::AbstractMatrix
  b_lin::AbstractVector
end

abstract type RBOperator{T} <: ParamOperator{T} end

function allocate_rbcache(op::RBOperator,args...)
  @abstractmethod
end

struct GenericRBOperator{T} <: RBOperator{T}
  op::ParamOperator{T}
  trial::RBSpace
  test::RBSpace
  lhs::AffineContribution
  rhs::AffineContribution
end

FESpaces.get_trial(op::GenericRBOperator) = op.trial
FESpaces.get_test(op::GenericRBOperator) = op.test
get_fe_trial(op::GenericRBOperator) = get_trial(op.op)
get_fe_test(op::GenericRBOperator) = get_test(op.op)

function ParamSteady.allocate_paramcache(
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector)

  allocate_paramcache(op.op,r,u)
end

function ParamSteady.update_paramcache!(
  paramcache,
  op::GenericRBOperator,
  r::Realization)

  update_paramcache!(paramcache,op.op,r)
end

function allocate_rbcache(
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector)

  paramcache = allocate_paramcache(op,r,x)

  b = allocate_residual(op.op,r,u,paramcache)
  coeffb,b̂ = allocate_hypred_cache(op.rhs,r)
  bcache = HypRedCache(b,coeffb,b̂)

  A = allocate_jacobian(op.op,r,u,paramcache)
  coeffA,Â = allocate_hypred_cache(op.lhs,r)
  Acache = HypRedCache(A,coeffA,Â)

  trial = evaluate(get_trial(op),r)

  return RBCache(Â,b̂,trial,paramcache)
end

function Algebra.allocate_residual(
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector,
  rbcache::RBCache)

  rbcache.b
end

function Algebra.allocate_jacobian(
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector,
  rbcache::RBCache)

  rbcache.A
end

function Algebra.residual!(
  cache::HypRedCache,
  op::GenericRBOperator,
  r::AbstractRealization,
  u::AbstractParamVector,
  rbcache::RBCache)

  b = cache.fe_quantity
  paramcache = rbcache.paramcache

  feb = fe_residual!(b,op,r,u,paramcache)
  inv_project!(cache,op.rhs,feb)
end

function Algebra.jacobian!(
  cache::HypRedCache,
  op::GenericRBOperator,
  r::AbstractRealization,
  u::AbstractParamVector,
  rbcache::RBCache)

  A = cache.fe_quantity
  paramcache = rbcache.paramcache

  feA = fe_jacobian!(A,op,r,u,paramcache)
  inv_project!(cache,op.lhs,feA)
end

function fe_jacobian!(
  A,
  op::GenericRBOperator,
  r::AbstractRealization,
  u::AbstractParamVector,
  paramcache)

  jacobian!(A,op.op,r,u,paramcache)
  Ai = select_at_indices(A,op.lhs)
  return Ai
end

function fe_residual!(
  b,
  op::GenericRBOperator,
  r::AbstractRealization,
  u::AbstractParamVector,
  paramcache)

  residual!(b,op.op,r,u,paramcache)
  bi = select_at_indices(b,op.rhs)
  return bi
end

"""
    struct LinearNonlinearRBOperator <: RBOperator{LinearNonlinearParamEq} end

Extends the concept of [`GenericRBOperator`](@ref) to accommodate the linear/nonlinear
splitting of terms in nonlinear applications

"""
struct LinearNonlinearRBOperator <: RBOperator{LinearNonlinearParamEq}
  op_linear::GenericRBOperator{<:LinearParamEq}
  op_nonlinear::GenericRBOperator{NonlinearParamEq}
end

ParamSteady.get_linear_operator(op::LinearNonlinearRBOperator) = op.op_linear
ParamSteady.get_nonlinear_operator(op::LinearNonlinearRBOperator) = op.op_nonlinear

function FESpaces.get_test(op::LinearNonlinearRBOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_nonlinear)
end

function FESpaces.get_trial(op::LinearNonlinearRBOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_nonlinear)
end

function get_fe_trial(op::LinearNonlinearRBOperator)
  @check get_fe_trial(op.op_linear) === get_fe_trial(op.op_nonlinear)
  get_fe_trial(op.op_nonlinear)
end

function get_fe_test(op::LinearNonlinearRBOperator)
  @check get_fe_test(op.op_linear) === get_fe_test(op.op_nonlinear)
  get_fe_test(op.op_nonlinear)
end

function ParamSteady.allocate_paramcache(
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector)

  allocate_paramcache(op.op,r,u)
end

function ParamSteady.update_paramcache!(
  paramcache,
  op::LinearNonlinearRBOperator,
  r::Realization)

  update_paramcache!(paramcache,op.op,r)
end

function allocate_rbcache(
  op::LinearNonlinearRBOperator,
  r::AbstractRealization,
  u::AbstractParamVector)

  lop = get_linear_operator(op)
  nlop = get_nonlinear_operator(op)

  rbcache = allocate_rbcache(nlop,r,u)
  A_lin = assemble_jacobian(op_lin,μ,u,paramcache)
  b_lin = assemble_residual(op_lin,μ,u,paramcache)

  return LinearNonlinearRBCache(rbcache,A_lin,b_lin)
end

function Algebra.allocate_residual(
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector,
  rbcache::LinearNonlinearRBCache)

  rbcache.rbcache.b
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector,
  rbcache::LinearNonlinearRBCache)

  rbcache.rbcache.A
end

function Algebra.residual!(
  cache::HypRedCache,
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector,
  rbcache::LinearNonlinearRBCache)

  nlop = get_nonlinear_operator(op)
  A_lin = rbcache.A_lin
  b_lin = rbcache.b_lin
  paramcache = rbcache.rbcache.paramcache

  b_nlin = residual!(cache,nlop,r,u,paramcache)
  axpy!(1.0,b_nlin,b_lin)
  mul!(b_nlin,A_lin,u,true,false)

  return b_nlin
end

function Algebra.jacobian!(
  cache::HypRedCache,
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector,
  rbcache::LinearNonlinearRBCache)

  nlop = get_nonlinear_operator(op)
  A_lin = rbcache.A_lin
  paramcache = rbcache.rbcache.paramcache

  A_nlin = jacobian!(cache,nlop,r,u,paramcache)
  axpy!(1.0,A_nlin,A_lin)

  return A_nlin
end

# Solve a POD-MDEIM problem

function online_cache(op::RBOperator,r::Realization)
  fe_trial = get_fe_trial(op)(r)
  trial = get_trial(op)(r)
  y = zero_free_values(fe_trial)
  x̂ = zero_free_values(trial)

  paramcache = allocate_paramcache(op,r,y)
  rbcache = allocate_rbcache(op,r)

  fecache = (y,paramcache)
  rbcache = (x̂,rbcache)
  return RBOnlineCache(fecache,rbcache)
end

function online_cache(op::LinearNonlinearRBOperator,r::Realization)
  fe_trial = get_fe_trial(op)(r)
  trial = get_trial(op)(r)
  y = zero_free_values(fe_trial)
  x̂ = zero_free_values(trial)

  paramcache = allocate_paramcache(op,r,y)
  rbcache = allocate_rbcache(op,r)

  fecache = (y,paramcache)
  rbcache = (x̂,rbcache)
  return RBOnlineCache(fecache,rbcache)
end

function online_cache!(solver::RBSolver,op::RBOperator,r::Realization)
  cache = solver.cache
  y,paramcache = cache.fecache
  if param_length(r) != param_length(y)
    init_online_cache!(solver,op,r)
  else
    paramcache = update_paramcache!(paramcache,op,r)
    cache.fecache = y,paramcache
  end
  return
end

function Algebra.solve(
  solver::RBSolver,
  op::RBOperator{NonlinearParamEq},
  r::AbstractRealization)

  @notimplemented "Split affine from nonlinear operator when running the RB solve"
end

function Algebra.solve(
  solver::RBSolver,
  op::RBOperator,
  r::AbstractRealization)

  fesolver = get_fe_solver(solver)
  fe_trial = get_fe_trial(op)(r)
  trial = get_trial(op)(r)
  x = zero_free_values(fe_trial)
  x̂ = zero_free_values(trial)

  rbcache = allocate_rbcache(op,r,x)

  t = @timed solve!(x̂,fesolver,op,r,y,rbcache)
  stats = CostTracker(t,nruns=num_params(r))

  return x̂,stats
end

function Algebra.solve!(
  x̂::AbstractVector,
  fesolver::LinearFESolver,
  op::RBOperator,
  r::Realization,
  x::AbstractVector,
  rbcache::RBCache)

  Â = jacobian(op,r,x,rbcache)
  b̂ = residual(op,r,x,rbcache)
  solve!(x̂,fesolver.ls,Â,b̂)
  return x̂
end

function Algebra.solve!(
  x̂::AbstractVector,
  fesolver::NonlinearFESolver,
  op::LinearNonlinearRBOperator,
  r::Realization,
  x::AbstractVector,
  rbcache::LinearNonlinearRBCache)

  Â = jacobian(op,r,x,rbcache)
  b̂ = residual(op,r,x,rbcache)
  dx̂ = similar(x̂)
  fill!(dx̂,zero(eltype(dx̂)))
  Â_item = testitem(Â)
  x̂_item = testitem(x̂)
  ss = symbolic_setup(nls.ls,Â_item)
  ns = numerical_setup(ss,Â_item,x̂_item)
  Algebra._solve_nr!(x̂,Â,b̂,dx̂,ns,fesolver.nls,op)
  return x̂
end

# cache utils

# selects the entries of the snapshots relevant to the reduced integration domain
# in `a`
function select_at_indices(s::AbstractArray,a::HyperReduction)
  s[get_integration_domain(a)]
end

function Arrays.return_cache(::typeof(select_at_indices),s::AbstractArray,a::HyperReduction,args...)
  select_at_indices(s,a,args...)
end

function Arrays.return_cache(
  ::typeof(select_at_indices),
  s::Union{BlockArray,BlockParamArray},
  a::BlockHyperReduction,
  args...)

  @check size(blocks(s)) == size(a)
  @notimplementedif isempty(findall(a.touched))
  i = findfirst(a.touched)
  cache = return_cache(select_at_indices,blocks(s)[i],a[i],args...)
  block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
  return block_cache
end

function select_at_indices(s::Union{BlockArray,BlockParamArray},a::BlockHyperReduction,args...)
  s′ = return_cache(select_at_indices,s,a,args...)
  for i = eachindex(a)
    if a.touched[i]
      s′[i] = select_at_indices(blocks(s)[i],a[i],args...)
    end
  end
  return ArrayBlock(s′,a.touched)
end

function select_at_indices(s::ArrayContribution,a::AffineContribution)
  contribution(s.trians) do trian
    select_at_indices(s[trian],a[trian])
  end
end
