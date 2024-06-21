"""
    reduced_operator(solver::RBSolver,feop::ParamFEOperator,
      s::Union{AbstractSteadySnapshots,BlockSnapshots}) -> PGOperator

    reduced_operator(solver::RBSolver,feop::TransientParamFEOperator,
      s::Union{AbstractTransientSnapshots,BlockSnapshots}) -> TransientPGOperator

(Petrov-)Galerkin projection operator of an (algebraic) FE operator onto a
reduced vector subspace

"""
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
  trial::FESubspace,
  test::FESubspace,
  s)

  pop = PGOperator(op,trial,test)
  reduced_operator(solver,pop,s)
end

"""
    abstract type RBOperator{T<:ParamOperatorType} <: ParamOperatorWithTrian{T} end

Subtypes:
- [`PGOperator`](@ref)
- [`PGMDEIMOperator`](@ref)
- [`LinearNonlinearPGMDEIMOperator`](@ref)

"""
abstract type RBOperator{T<:ParamOperatorType} <: ParamOperatorWithTrian{T} end

"""
    struct PGOperator{T} <: RBOperator{T} end

Represents a projection operator of a [`ParamOperatorWithTrian`](@ref) object
onto the trial/test FESubspaces `trial` and `test`

"""
struct PGOperator{T} <: RBOperator{T}
  op::ParamOperatorWithTrian{T}
  trial::FESubspace
  test::FESubspace
end

FESpaces.get_trial(op::PGOperator) = op.trial
FESpaces.get_test(op::PGOperator) = op.test
ParamDataStructures.realization(op::PGOperator;kwargs...) = realization(op.op;kwargs...)
ParamSteady.get_fe_operator(op::PGOperator) = ParamSteady.get_fe_operator(op.op)
ParamSteady.get_vector_index_map(op::PGOperator) = get_vector_index_map(op.op)
ParamSteady.get_matrix_index_map(op::PGOperator) = get_matrix_index_map(op.op)
get_fe_trial(op::PGOperator) = get_trial(op.op)
get_fe_test(op::PGOperator) = get_test(op.op)

function ParamSteady.get_linear_operator(op::PGOperator)
  PGOperator(get_linear_operator(op.op),op.trial,op.test)
end

function ParamSteady.get_nonlinear_operator(op::PGOperator)
  PGOperator(get_nonlinear_operator(op.op),op.trial,op.test)
end

function ParamSteady.set_triangulation(
  op::PGOperator,
  trians_rhs,
  trians_lhs)

  PGOperator(set_triangulation(op.op,trians_rhs,trians_lhs),op.trial,op.test)
end

function ParamSteady.change_triangulation(
  op::PGOperator,
  trians_rhs,
  trians_lhs;
  kwargs...)

  PGOperator(change_triangulation(op.op,trians_rhs,trians_lhs;kwargs...),op.trial,op.test)
end

function Algebra.allocate_residual(op::PGOperator,r::AbstractParamRealization,u::AbstractParamVector)
  allocate_residual(op.op,r,u)
end

function Algebra.allocate_jacobian(op::PGOperator,r::AbstractParamRealization,u::AbstractParamVector)
  allocate_jacobian(op.op,r,u)
end

function ParamSteady.allocate_paramcache(
  op::PGOperator,
  r::ParamRealization,
  u::AbstractParamVector)

  allocate_paramcache(op.op,r,u)
end

function ParamSteady.update_paramcache!(
  paramcache,
  op::PGOperator,
  r::ParamRealization)

  update_odeopcache!(paramcache,op.op,r)
end

function Algebra.residual!(
  b::Contribution,
  op::PGOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector,
  paramcache)

  residual!(b,op.op,r,u,paramcache)
  i = get_vector_index_map(op)
  return Snapshots(b,i,r)
end

function Algebra.jacobian!(
  A::Contribution,
  op::PGOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector,
  paramcache)

  jacobian!(A,op.op,r,u,paramcache)
  i = get_matrix_index_map(op)
  return Snapshots(A,i,r)
end

"""
    jacobian_and_residual(solver::RBSolver, op::RBOperator, s::AbstractSteadySnapshots
      ) -> (AbstractSteadySnapshots, AbstractSteadySnapshots)
    jacobian_and_residual(solver::RBSolver, op::TransientRBOperator, s::AbstractTransientSnapshots
      ) -> (NTuple{N,AbstractTransientSnapshots}, AbstractTransientSnapshots)
    jacobian_and_residual(solver::RBSolver, op::RBOperator, s::BlockSnapshots
      ) -> (BlockSnapshots, BlockSnapshots)
    jacobian_and_residual(solver::RBSolver, op::TransientRBOperator, s::BlockSnapshots
      ) -> (NTuple{N,BlockSnapshots}, BlockSnapshots)

Returns the jacobians/residuals evaluated in the input snapshots `s`. In transient
settings, the jacobians are ntuples of order `N`, with `N` equal to the order of
the time derivative

"""
function jacobian_and_residual(solver::RBSolver,op::RBOperator,s)
  jacobian_and_residual(get_fe_solver(solver),op.op,s)
end

function jacobian_and_residual(fesolver::FESolver,op::ParamOperator,s)
  u = get_values(s)
  r = get_realization(s)
  A,b = jacobian_and_residual(fesolver,op,r,u)
  iA = get_matrix_index_map(op)
  ib = get_matrix_index_map(op)
  return Snapshots(A,iA,r),Snapshots(b,ib,r)
end

function jacobian_and_residual(::FESolver,op::ParamOperator,r::ParamRealization,u::AbstractVector)
  pcache = allocate_paramcache(fesolver,op,r,u)
  A = jacobian(op,r,u,pcache)
  b = residual(op,r,u,pcache)
  return A,b
end
