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
  op::ParamOperator{<:LinearNonlinearParamEq},
  red_trial::RBSpace,
  red_test::RBSpace,
  s)

  red_op_lin = reduced_operator(solver,get_linear_operator(op),red_trial,red_test,s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(op),red_trial,red_test,s)
  LinearNonlinearRBOperator(red_op_lin,red_op_nlin)
end

struct RBCache{Ta,Tb} <: AbstractParamCache
  A::Ta
  b::Tb
  trial::RBSpace
  paramcache::ParamOpCache
end

struct LinearNonlinearRBCache <: AbstractParamCache
  rbcache::RBCache
  A::AbstractMatrix
  b::AbstractVector
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

function allocate_rbcache(
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector)

  paramcache = allocate_paramcache(op.op,r,u)

  A = allocate_jacobian(op.op,r,u,paramcache)
  coeffA,Â = allocate_hypred_cache(op.lhs,r)
  Acache = HRParamArray(A,coeffA,Â)

  b = allocate_residual(op.op,r,u,paramcache)
  coeffb,b̂ = allocate_hypred_cache(op.rhs,r)
  bcache = HRParamArray(b,coeffb,b̂)

  trial = evaluate(get_trial(op),r)

  return RBCache(Acache,bcache,trial,paramcache)
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
  cache::HRParamArray,
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector,
  rbcache::RBCache)

  b = cache.fe_quantity
  paramcache = rbcache.paramcache

  feb = fe_residual!(b,op,r,u,paramcache)
  inv_project!(cache,op.rhs,feb)
end

function Algebra.residual!(
  cache::HRParamArray,
  op::GenericRBOperator,
  r::Realization,
  u::RBParamVector,
  rbcache::RBCache)

  inv_project!(u.fe_data,rbcache.trial,u.data)
  residual!(cache,op,r,u.fe_data,rbcache)
end

function Algebra.jacobian!(
  cache::HRParamArray,
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector,
  rbcache::RBCache)

  A = cache.fe_quantity
  paramcache = rbcache.paramcache

  feA = fe_jacobian!(A,op,r,u,paramcache)
  inv_project!(cache,op.lhs,feA)
end

function Algebra.jacobian!(
  cache::HRParamArray,
  op::GenericRBOperator,
  r::Realization,
  u::RBParamVector,
  rbcache::RBCache)

  inv_project!(u.fe_data,rbcache.trial,u.data)
  jacobian!(cache,op,r,u.fe_data,rbcache)
end

function fe_jacobian!(
  A,
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache)

  jacobian!(A,op.op,r,u,paramcache)
  Ai = select_at_indices(A,op.lhs)
  return Ai
end

function fe_residual!(
  b,
  op::GenericRBOperator,
  r::Realization,
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
  op_linear::GenericRBOperator{LinearParamEq}
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

function allocate_rbcache(
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector)

  lop = get_linear_operator(op)
  nlop = get_nonlinear_operator(op)

  rbcache_lin = allocate_rbcache(lop,r,u)
  rbcache_nlin = allocate_rbcache(nlop,r,u)
  A_lin = jacobian(lop,r,u,rbcache_lin)
  b_lin = residual(lop,r,u,rbcache_lin)

  return LinearNonlinearRBCache(rbcache_nlin,A_lin,b_lin)
end

function allocate_rbcache(
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::RBParamVector)

  allocate_rbcache(op,r,u.fe_data)
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
  cache::HRParamArray,
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector,
  rbcache::LinearNonlinearRBCache)

  nlop = get_nonlinear_operator(op)
  A_lin = rbcache.A
  b_lin = rbcache.b
  rbcache_nlin = rbcache.rbcache

  b_nlin = residual!(cache,nlop,r,u,rbcache_nlin)
  axpy!(1.0,b_lin,b_nlin)
  mul!(b_nlin,A_lin,u,true,true)

  return b_nlin
end

function Algebra.jacobian!(
  cache::HRParamArray,
  op::LinearNonlinearRBOperator,
  r::Realization,
  u::AbstractParamVector,
  rbcache::LinearNonlinearRBCache)

  nlop = get_nonlinear_operator(op)
  A_lin = rbcache.A
  rbcache_nlin = rbcache.rbcache

  A_nlin = jacobian!(cache,nlop,r,u,rbcache_nlin)
  axpy!(1.0,A_lin,A_nlin)

  return A_nlin
end

# Solve a POD-MDEIM problem

function Algebra.solve(
  solver::RBSolver,
  op::RBOperator{NonlinearParamEq},
  r::Realization)

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

  t = @timed solve!(x̂,fesolver,op,r,x,rbcache)
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
  rmul!(b̂,-1)
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

  nls = fesolver.nls
  ŷ = RBParamVector(x̂,x)

  Âcache = allocate_jacobian(op,r,ŷ,rbcache)
  b̂cache = allocate_residual(op,r,ŷ,rbcache)

  Â_item = testitem(Âcache)
  x̂_item = testitem(x̂)
  dx̂ = allocate_in_domain(Â_item)
  fill!(dx̂,zero(eltype(dx̂)))
  ss = symbolic_setup(BackslashSolver(),Â_item)
  ns = numerical_setup(ss,Â_item,x̂_item)

  nlop = ParamNonlinearOperator(op,r,rbcache)
  Algebra._solve_nr!(ŷ,Âcache,b̂cache,dx̂,ns,nls,nlop)

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
