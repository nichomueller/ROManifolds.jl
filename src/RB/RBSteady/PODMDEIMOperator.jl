function reduced_operator(
  solver::RBSolver,
  op::PODOperator,
  s::AbstractSnapshots)

  red_lhs,red_rhs = reduced_jacobian_residual(solver,op,s)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = map(get_domains,red_lhs)
  new_op = change_triangulation(op,trians_rhs,trians_lhs)
  PODMDEIMOperator(new_op,red_lhs,red_rhs)
end

function reduced_operator(
  solver::RBSolver,
  op::PODOperator{LinearNonlinearParamODE},
  s::AbstractSnapshots)

  red_op_lin = reduced_operator(solver,get_linear_operator(op),s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(op),s)
  LinearNonlinearPODMDEIMOperator(red_op_lin,red_op_nlin)
end

struct PODMDEIMOperator{T} <: RBOperator{T}
  op::PODOperator{T}
  lhs
  rhs
end

FESpaces.get_trial(op::PODMDEIMOperator) = get_trial(op.op)
FESpaces.get_test(op::PODMDEIMOperator) = get_test(op.op)
ParamDataStructures.realization(op::PODMDEIMOperator;kwargs...) = realization(op.op;kwargs...)
ParamFESpaces.get_fe_operator(op::PODMDEIMOperator) = ParamODEs.get_fe_operator(op.op)
get_fe_trial(op::PODMDEIMOperator) = get_fe_trial(op.op)
get_fe_test(op::PODMDEIMOperator) = get_fe_test(op.op)

function Algebra.allocate_residual(op::PODMDEIMOperator,r::AbstractParamRealization,u::AbstractParamVector)
  allocate_residual(op.op,r,u)
end

function Algebra.allocate_jacobian(op::PODMDEIMOperator,r::AbstractParamRealization,u::AbstractParamVector)
  allocate_jacobian(op.op,r,u)
end

function Algebra.residual!(
  b::Contribution,
  op::PODMDEIMOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector)

  fe_sb = fe_residual!(b,op,r,u,odeopcache)
  b̂ = mdeim_result(op.rhs,fe_sb)
  return b̂
end

function Algebra.jacobian!(
  A::Contribution,
  op::PODMDEIMOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector)

  fe_sA = fe_jacobian!(A,op,r,u,ws,odeopcache)
  Â = mdeim_result(op.lhs,fe_sA)
  return Â
end

function jacobian_and_residual(solver::RBSolver,op::PODMDEIMOperator,s::AbstractSnapshots)
  x = get_values(s)
  r = get_realization(s)
  fesolver = get_fe_solver(solver)
  jacobian_and_residual(fesolver,op,r,x)
end

function _select_snapshots_at_space_locations(s,a)
  ids_space = get_indices_space(a)
  select_snapshots_entries(s,ids_space)
end

function _select_snapshots_at_space_locations(
  s::ArrayContribution,a::AffineContribution)
  contribution(s.trians) do trian
    _select_snapshots_at_space_time_locations(s[trian],a[trian])
  end
end

function fe_jacobian!(
  cache,
  op::PODMDEIMOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector)

  A = jacobian!(cache,op.op,r,u)
  Ai = _select_snapshots_at_space_locations(A,op.lhs)
  return Ai
end

function fe_residual!(
  cache,
  op::PODMDEIMOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector)

  b = residual!(cache,op.op,r,u)
  bi = _select_snapshots_at_space_locations(b,op.rhs)
  return bi
end

struct LinearNonlinearPODMDEIMOperator <: RBOperator{LinearNonlinearParamODE}
  op_linear::PODMDEIMOperator
  op_nonlinear::PODMDEIMOperator
  function LinearNonlinearPODMDEIMOperator(op_linear,op_nonlinear)
    @check isa(op_linear,PODMDEIMOperator{LinearParamODE})
    new(op_linear,op_nonlinear)
  end
end

ParamFESpaces.get_linear_operator(op::LinearNonlinearPODMDEIMOperator) = op.op_linear
ParamFESpaces.get_nonlinear_operator(op::LinearNonlinearPODMDEIMOperator) = op.op_nonlinear

function FESpaces.get_test(op::LinearNonlinearPODMDEIMOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_nonlinear)
end

function FESpaces.get_trial(op::LinearNonlinearPODMDEIMOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_nonlinear)
end

function ParamDataStructures.realization(op::LinearNonlinearPODMDEIMOperator;kwargs...)
  realization(op.op_nonlinear;kwargs...)
end

function ParamFESpaces.get_fe_operator(op::LinearNonlinearPODMDEIMOperator)
  join_operators(get_fe_operator(op.op_linear),get_fe_operator(op.op_nonlinear))
end

function get_fe_trial(op::LinearNonlinearPODMDEIMOperator)
  @check get_fe_trial(op.op_linear) === get_fe_trial(op.op_nonlinear)
  get_fe_trial(op.op_nonlinear)
end

function get_fe_test(op::LinearNonlinearPODMDEIMOperator)
  @check get_fe_test(op.op_linear) === get_fe_test(op.op_nonlinear)
  get_fe_test(op.op_nonlinear)
end

function Algebra.allocate_residual(
  op::LinearNonlinearPODMDEIMOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector)

  b_lin = allocate_residual(op.op_linear,r,u)
  b_nlin = copy(b_lin)
  return b_lin,b_nlin
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearPODMDEIMOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector)

  A_lin = allocate_jacobian(op.op_linear,r,u)
  A_nlin = copy(A_lin)
  return A_lin,A_nlin
end

function Algebra.residual!(
  b::Tuple,
  op::LinearNonlinearPODMDEIMOperator,
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
  op::LinearNonlinearPODMDEIMOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector)

  Â_lin,A_nlin = A
  fe_sA_nlin = fe_jacobian!(A_nlin,op.op_nonlinear,r,u)
  Â_nlin = mdeim_result(op.op_nonlinear.lhs,fe_sA_nlin)
  @. Â_nlin = Â_nlin + Â_lin
  return Â_nlin
end

# Solve a POD-MDEIM problem

function Algebra.solve(solver::RBSolver,op::RBOperator,s::AbstractSnapshots)
  son = select_snapshots(s,online_params(solver))
  ron = get_realization(son)
  solve(solver,op,ron)
end

function Algebra.solve(
  solver::RBSolver,
  op::RBOperator{NonlinearParamODE},
  r::AbstractParamRealization)

  @notimplemented "Split affine from nonlinear operator when running the RB solve"
end

function Algebra.solve(
  solver::RBSolver,
  op::RBOperator,
  r::AbstractParamRealization)

  stats = @timed begin
    fesolver = get_fe_solver(solver)
    trial = get_trial(op)(r)
    fe_trial = get_fe_trial(op)(r)
    x̂ = zero_free_values(trial)
    y = zero_free_values(fe_trial)
    solve!((x̂,),fesolver,op,r,(y,))
  end

  x = recast(x̂,trial)
  s = Snapshots(x,r)
  cs = ComputationalStats(stats,num_params(r))
  return s,cs
end

# for testing/visualization purposes

function pod_mdeim_error(solver,feop,op::RBOperator,s::AbstractSnapshots)
  pod_err = pod_error(get_trial(op),s,assemble_norm_matrix(feop))
  mdeim_err = mdeim_error(solver,feop,op,s)
  return pod_err,mdeim_err
end
