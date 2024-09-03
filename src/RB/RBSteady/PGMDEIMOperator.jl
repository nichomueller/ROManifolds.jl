"""
    reduced_operator(solver::RBSolver,op::PGOperator,
      s::Union{AbstractSteadySnapshots,BlockSnapshots}) -> PGMDEIMOperator

    reduced_operator(solver::RBSolver,op::TransientPGOperator,
      s::Union{AbstractTransientSnapshots,BlockSnapshots}) -> TransientPGMDEIMOperator

In steady settings, computes the composition operator
  [`PGOperator`][@ref] ∘ [`ReducedAlgebraicOperator`][@ref]

In transient settings, computes the composition operator
  [`TransientPGOperator`][@ref] ∘ [`ReducedAlgebraicOperator`][@ref]

This allows the projection of MDEIM-approximated residuals/jacobians onto the
FESubspace encoded in `op`. A change of triangulation occurs for residuals/jacobians
so that the numerical integration can be efficiently ran to assemble them

"""
function reduced_operator(
  solver::RBSolver,
  op::PGOperator,
  s)

  red_lhs,red_rhs = reduced_jacobian_residual(solver,op,s)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = map(get_domains,red_lhs)
  new_op = change_triangulation(op,trians_rhs,trians_lhs)
  PGMDEIMOperator(new_op,red_lhs,red_rhs)
end

function reduced_operator(
  solver::RBSolver,
  op::PGOperator{LinearNonlinearParamEq},
  s)

  red_op_lin = reduced_operator(solver,get_linear_operator(op),s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(op),s)
  LinearNonlinearPGMDEIMOperator(red_op_lin,red_op_nlin)
end

"""
    struct PGMDEIMOperator{T} <: RBOperator{T} end

Represents the composition operator

[`PGOperator`][@ref] ∘ [`ReducedAlgebraicOperator`][@ref]

This allows the projection of MDEIM-approximated residuals/jacobians `rhs` and
`lhs` onto the FESubspace encoded in `op`. In particular, the residual of a
PGMDEIMOperator is computed as follows:

1) numerical integration is performed to compute the residual on its reduced
  integration domain
2) the MDEIM online phase takes place for the assembly of the projected, MDEIM-
  approximated residual

The same reasoning holds for the jacobian

"""
struct PGMDEIMOperator{T} <: RBOperator{T}
  op::PGOperator{T}
  lhs
  rhs
end

FESpaces.get_trial(op::PGMDEIMOperator) = get_trial(op.op)
FESpaces.get_test(op::PGMDEIMOperator) = get_test(op.op)
ParamDataStructures.realization(op::PGMDEIMOperator;kwargs...) = realization(op.op;kwargs...)
ParamSteady.get_fe_operator(op::PGMDEIMOperator) = ParamSteady.get_fe_operator(op.op)
ParamSteady.get_vector_index_map(op::PGMDEIMOperator) = get_vector_index_map(op.op)
ParamSteady.get_matrix_index_map(op::PGMDEIMOperator) = get_matrix_index_map(op.op)
get_fe_trial(op::PGMDEIMOperator) = get_fe_trial(op.op)
get_fe_test(op::PGMDEIMOperator) = get_fe_test(op.op)

function Algebra.allocate_residual(op::PGMDEIMOperator,r::AbstractParamRealization,u::AbstractParamVector)
  allocate_residual(op.op,r,u)
end

function Algebra.allocate_jacobian(op::PGMDEIMOperator,r::AbstractParamRealization,u::AbstractParamVector)
  allocate_jacobian(op.op,r,u)
end

function ParamSteady.allocate_paramcache(
  op::PGMDEIMOperator,
  r::ParamRealization,
  u::AbstractParamVector)

  allocate_paramcache(op.op,r,u)
end

function ParamSteady.update_paramcache!(
  paramcache,
  op::PGMDEIMOperator,
  r::ParamRealization)

  @warn "For performance reasons, it would be best to update the cache at the very
    start of the online phase"
  update_paramcache!(paramcache,op.op,r)
end

function Algebra.residual!(
  b::Contribution,
  op::PGMDEIMOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector,
  paramcache)

  fe_sb = fe_residual!(b,op,r,u,paramcache)
  b̂ = mdeim_result(op.rhs,fe_sb)
  return b̂
end

function Algebra.jacobian!(
  A::Contribution,
  op::PGMDEIMOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector,
  paramcache)

  fe_sA = fe_jacobian!(A,op,r,u,paramcache)
  Â = mdeim_result(op.lhs,fe_sA)
  return Â
end

function jacobian_and_residual(solver::RBSolver,op::PGMDEIMOperator,s)
  x = get_values(s)
  r = get_realization(s)
  fesolver = get_fe_solver(solver)
  jacobian_and_residual(fesolver,op,r,x)
end

function select_fe_space_at_indices(fs::FESpace,indices)
  @notimplemented
end

function select_fe_space_at_indices(fs::TrivialParamFESpace,indices)
  TrivialParamFESpace(fs.space,Val(length(indices)))
end

function select_fe_space_at_indices(fs::SingleFieldParamFESpace,indices)
  dvi = ConsecutiveArrayOfArrays(fs.dirichlet_values.data[:,indices])
  TrialParamFESpace(dvi,fs.space)
end

function select_cache_at_indices(u::ConsecutiveArrayOfArrays,paramcache,indices)
  @unpack Us,Ups,pfeopcache,form = paramcache
  new_Us = select_fe_space_at_indices(Us,indices)
  new_XhF = ConsecutiveArrayOfArrays(u.data[:,indices])
  new_paramcache = ParamOpFromFEOpCache(new_Us,Uts,tfeopcache,const_forms)
  return new_xhF,new_paramcache
end

function select_cache_at_indices(us::BlockVectorOfVectors,odeopcache,indices)
  @unpack Us,Ups,pfeopcache,form = paramcache
  VT = Us.vector_type
  style = Us.multi_field_style
  spaces = select_fe_space_at_indices(Us,indices)
  new_Us = MultiFieldFESpace(VT,spaces,style)
  new_XhF = mortar([ConsecutiveArrayOfArrays(b.data[:,indices]) for b in blocks(u)])
  new_paramcache = ParamOpFromFEOpCache(new_Us,Uts,tfeopcache,const_forms)
  return new_xhF,new_paramcache
end

function select_slvrcache_at_indices(b::ConsecutiveArrayOfArrays,indices)
  ConsecutiveArrayOfArrays(b.data[:,indices])
end

function select_slvrcache_at_indices(A::MatrixOfSparseMatricesCSC,indices)
  MatrixOfSparseMatricesCSC(A.m,A.n,A.colptr,A.rowval,A.data[:,indices])
end

function select_slvrcache_at_indices(A::BlockArrayOfArrays,indices)
  map(a -> select_slvrcache_at_indices(a,indices),blocks(A)) |> mortar
end

function select_slvrcache_at_indices(cache::ArrayContribution,indices)
  contribution(cache.trians) do trian
    select_slvrcache_at_indices(cache[trian],indices)
  end
end

function select_fe_quantities_at_indices(cache,r,u,param_indices)
  slvrcache,paramcache = cache
  red_r = r[param_indices]
  red_cache = select_slvrcache_at_indices(cache,indices)
  red_u,red_paramcache = select_cache_at_indices(u,paramcache,indices)
  return red_cache,red_r,red_u,red_paramcache
end

# selects the entries of the snapshots relevant to the reduced integration domain
# in `a`
function _select_snapshots_at_indices(s,a)
  ids_space = get_indices_space(a)
  select_snapshots_entries(s,ids_space)
end

function _select_snapshots_at_indices(
  s::ArrayContribution,a::AffineContribution)
  contribution(s.trians) do trian
    _select_snapshots_at_indices(s[trian],a[trian])
  end
end

function fe_jacobian!(
  cache,
  op::PGMDEIMOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector,
  paramcache)

  A = jacobian!(cache,op.op,r,u,paramcache)
  Ai = _select_snapshots_at_indices(A,op.lhs)
  return Ai
end

function fe_residual!(
  cache,
  op::PGMDEIMOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector,
  paramcache)

  b = residual!(cache,op.op,r,u,paramcache)
  bi = _select_snapshots_at_indices(b,op.rhs)
  return bi
end

"""
    struct LinearNonlinearPGMDEIMOperator <: RBOperator{LinearNonlinearParamEq} end

Extends the concept of [`PGMDEIMOperator`](@ref) to accommodate the linear/nonlinear
splitting of terms in nonlinear applications

"""
struct LinearNonlinearPGMDEIMOperator <: RBOperator{LinearNonlinearParamEq}
  op_linear::PGMDEIMOperator{LinearParamEq}
  op_nonlinear::PGMDEIMOperator{NonlinearParamEq}
end

ParamSteady.get_linear_operator(op::LinearNonlinearPGMDEIMOperator) = op.op_linear
ParamSteady.get_nonlinear_operator(op::LinearNonlinearPGMDEIMOperator) = op.op_nonlinear

function FESpaces.get_test(op::LinearNonlinearPGMDEIMOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_nonlinear)
end

function FESpaces.get_trial(op::LinearNonlinearPGMDEIMOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_nonlinear)
end

function ParamDataStructures.realization(op::LinearNonlinearPGMDEIMOperator;kwargs...)
  realization(op.op_nonlinear;kwargs...)
end

function ParamSteady.get_fe_operator(op::LinearNonlinearPGMDEIMOperator)
  join_operators(get_fe_operator(op.op_linear),get_fe_operator(op.op_nonlinear))
end

function get_fe_trial(op::LinearNonlinearPGMDEIMOperator)
  @check get_fe_trial(op.op_linear) === get_fe_trial(op.op_nonlinear)
  get_fe_trial(op.op_nonlinear)
end

function get_fe_test(op::LinearNonlinearPGMDEIMOperator)
  @check get_fe_test(op.op_linear) === get_fe_test(op.op_nonlinear)
  get_fe_test(op.op_nonlinear)
end

function Algebra.allocate_residual(
  op::LinearNonlinearPGMDEIMOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector)

  b_lin = allocate_residual(op.op_linear,r,u)
  b_nlin = copy(b_lin)
  return b_lin,b_nlin
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearPGMDEIMOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector)

  A_lin = allocate_jacobian(op.op_linear,r,u)
  A_nlin = copy(A_lin)
  return A_lin,A_nlin
end

function Algebra.residual!(
  b::Tuple,
  op::LinearNonlinearPGMDEIMOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector)

  b̂_lin,b_nlin = b
  fe_sb_nlin = fe_residual!(b_nlin,op.op_nonlinear,r,u)
  b̂_nlin = mdeim_result(op.op_nonlinear.rhs,fe_sb_nlin)
  @. b̂_nlin = b̂_nlin + b̂_lin
  return b̂_nlin
end

function Algebra.jacobian!(
  A::Tuple,
  op::LinearNonlinearPGMDEIMOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector)

  Â_lin,A_nlin = A
  fe_sA_nlin = fe_jacobian!(A_nlin,op.op_nonlinear,r,u)
  Â_nlin = mdeim_result(op.op_nonlinear.lhs,fe_sA_nlin)
  @. Â_nlin = Â_nlin + Â_lin
  return Â_nlin
end

# Solve a POD-MDEIM problem

function Algebra.solve(solver::RBSolver,op::RBOperator,s)
  son = select_snapshots(s,online_params(solver))
  ron = get_realization(son)
  solve(solver,op,ron)
end

function Algebra.solve!(cache,solver::RBSolver,op::RBOperator,s)
  son = select_snapshots(s,online_params(solver))
  ron = get_realization(son)
  solve!(cache,solver,op,ron)
end

function Algebra.solve(
  solver::RBSolver,
  op::RBOperator{NonlinearParamEq},
  r::AbstractParamRealization)

  @notimplemented "Split affine from nonlinear operator when running the RB solve"
end

function Algebra.solve!(
  cache,
  solver::RBSolver,
  op::RBOperator{NonlinearParamEq},
  r::AbstractParamRealization)

  @notimplemented "Split affine from nonlinear operator when running the RB solve"
end

function Algebra.solve(
  solver::RBSolver,
  op::RBOperator,
  r::AbstractParamRealization)

  trial = get_trial(op)(r)
  fe_trial = get_fe_trial(op)(r)
  x̂ = zero_free_values(trial)
  y = zero_free_values(fe_trial)
  cache = x̂,y
  solve!(cache,solver,op,r)
end

function Algebra.solve!(
  cache,
  solver::RBSolver,
  op::RBOperator,
  r::AbstractParamRealization)

  x̂,y = cache
  fesolver = get_fe_solver(solver)

  t = @timed solve!((x̂,),fesolver,op,r,(y,))
  stats = CostTracker(t,num_params(r))

  trial = get_trial(op)(r)
  x = recast(x̂,trial)
  i = get_vector_index_map(op)
  s = Snapshots(x,i,r)

  return s,stats,cache
end

# for testing/visualization purposes

function pod_mdeim_error(solver,feop,op,s)
  state_red = get_state_reduction(solver)
  X = assemble_matrix_from_form(feop,get_norm(state_red))
  s′ = flatten_partition(s)
  pod_err = pod_error(get_trial(op),s′,X)
  mdeim_err = mdeim_error(solver,feop,op,s′)
  return pod_err,mdeim_err
end
