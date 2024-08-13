function RBSteady.reduced_operator(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  s)

  red_trial,red_test = reduced_fe_space(solver,feop,s)
  op = get_algebraic_operator(feop)
  reduced_operator(solver,op,red_trial,red_test,s)
end

function RBSteady.reduced_operator(
  solver::RBSolver,
  op::ODEParamOperator,
  trial::FESubspace,
  test::FESubspace,
  s)

  pop = TransientPGOperator(op,trial,test)
  reduced_operator(solver,pop,s)
end

"""
    abstract type TransientRBOperator{T<:ODEParamOperatorType} <: ODEParamOperatorWithTrian{T} end

Subtypes:
- [`TransientPGOperator`](@ref)
- [`TransientPGMDEIMOperator`](@ref)
- [`LinearNonlinearTransientPGMDEIMOperator`](@ref)

"""
abstract type TransientRBOperator{T<:ODEParamOperatorType} <: ODEParamOperatorWithTrian{T} end

"""
    struct TransientPGOperator{T} <: TransientRBOperator{T} end

Represents a projection operator of a [`ODEParamOperatorWithTrian`](@ref) object
onto the trial/test FESubspaces `trial` and `test`

"""
struct TransientPGOperator{T} <: TransientRBOperator{T}
  op::ODEParamOperatorWithTrian{T}
  trial::FESubspace
  test::FESubspace
end

FESpaces.get_trial(op::TransientPGOperator) = op.trial
FESpaces.get_test(op::TransientPGOperator) = op.test
ParamDataStructures.realization(op::TransientPGOperator;kwargs...) = realization(op.op;kwargs...)
ParamSteady.get_fe_operator(op::TransientPGOperator) = ParamSteady.get_fe_operator(op.op)
ParamSteady.get_vector_index_map(op::TransientPGOperator) = get_vector_index_map(op.op)
ParamSteady.get_matrix_index_map(op::TransientPGOperator) = get_matrix_index_map(op.op)
RBSteady.get_fe_trial(op::TransientPGOperator) = get_trial(op.op)
RBSteady.get_fe_test(op::TransientPGOperator) = get_test(op.op)

function ParamSteady.get_linear_operator(op::TransientPGOperator)
  TransientPGOperator(get_linear_operator(op.op),op.trial,op.test)
end

function ParamSteady.get_nonlinear_operator(op::TransientPGOperator)
  TransientPGOperator(get_nonlinear_operator(op.op),op.trial,op.test)
end

function ParamSteady.set_triangulation(
  op::TransientPGOperator,
  trians_rhs,
  trians_lhs)

  TransientPGOperator(set_triangulation(op.op,trians_rhs,trians_lhs),op.trial,op.test)
end

function ParamSteady.change_triangulation(op::TransientPGOperator,trians_rhs,trians_lhs)
  TransientPGOperator(change_triangulation(op.op,trians_rhs,trians_lhs),op.trial,op.test)
end

function ODEs.allocate_odecache(
  fesolver::ThetaMethod,
  op::TransientPGOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  dt,θ = fesolver.dt,fesolver.θ
  dtθ = θ*dt
  shift!(r,dt*(θ-1))

  (odeslvrcache,odeopcache) = allocate_odecache(fesolver,op.op,r,us)
  update_odeopcache!(odeopcache,op.op,r)
  shift!(r,dt*(1-θ))

  return (odeslvrcache,odeopcache)
end

function ODEs.allocate_odeopcache(
  op::TransientPGOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_odeopcache(op.op,r,us)
end

function ODEs.update_odeopcache!(
  odeopcache,
  op::TransientPGOperator,
  r::TransientParamRealization)

  @warn "For performance reasons, it would be best to update the cache at the very
    start, given that the online phase of a space-time ROM is time-independent"
  update_odeopcache!(odeopcache,op.op,r)
end

function Algebra.allocate_residual(
  op::TransientPGOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_residual(op.op,r,us,odeopcache)
end

function Algebra.allocate_jacobian(
  op::TransientPGOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_jacobian(op.op,r,us,odeopcache)
end

function Algebra.residual!(
  b::Contribution,
  op::TransientPGOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache;
  kwargs...)

  residual!(b,op.op,r,us,odeopcache;kwargs...)
  i = get_vector_index_map(op)
  return Snapshots(b,i,r)
end

function Algebra.jacobian!(
  A::TupOfArrayContribution,
  op::TransientPGOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  jacobian!(A,op.op,r,us,ws,odeopcache)
  i = get_matrix_index_map(op)
  return Snapshots(A,i,r)
end

function RBSteady.jacobian_and_residual(solver::RBSolver,op::TransientRBOperator,s)
  fesolver = get_fe_solver(solver)
  sjac = select_snapshots(s,RBSteady.jac_params(solver))
  sres = select_snapshots(s,RBSteady.res_params(solver))
  us_jac,us_res = (get_values(sjac),),(get_values(sres),)
  r_jac,r_res = get_realization(sjac),get_realization(sres)
  println("before jac")
  A = jacobian(fesolver,op.op,r_jac,us_jac)
  println("after jac, before res")
  b = residual(fesolver,op.op,r_res,us_res)
  println("after res")
  iA = get_matrix_index_map(op.op)
  ib = get_vector_index_map(op.op)
  return Snapshots(A,iA,r_jac),Snapshots(b,ib,r_res)
end
