mutable struct RBOnlineCache
  fecache
  rbcache
end

"""
    struct RBSolver{A,B,C,D} end

Wrapper around a FE solver (e.g. [`FESolver`](@ref) or [`ODESolver`](@ref)) with
additional information on the reduced basis (RB) method employed to solve a given
problem dependent on a set of parameters. A RB method is a projection-based
reduced order model where

1) a suitable subspace of a FESpace is sought, of dimension n ≪ Nₕ
2) a matrix-based discrete empirical interpolation method (MDEEIM) is performed
  to approximate the manifold of the parametric residuals and jacobians
3) the EIM approximations are compressed with (Petrov-)Galerkin projections
  onto the subspace
4) for every desired choice of parameters, numerical integration is performed, and
  the resulting n × n system of equations is cheaply solved

In particular:

- ϵ: tolerance used in the projection-based truncated proper orthogonal
  decomposition (TPOD) or in the tensor train singular value decomposition (TT-SVD),
  where a basis spanning the reduced subspace is computed
- nparams_state: number of snapshots considered when running TPOD or TT-SVD
- nparams_res: number of snapshots considered when running MDEIM for the residual
- nparams_jac: number of snapshots considered when running MDEIM for the jacobian
- nparams_test:  number of snapshots considered when computing the error the RB
  method commits with respect to the FE procedure

"""
struct RBSolver{A<:GridapType,B}
  fesolver::A
  state_reduction::Reduction
  residual_reduction::Reduction
  jacobian_reduction::B
  cache::RBOnlineCache
end

function RBSolver(
  fesolver::GridapType,
  state_reduction::Reduction,
  residual_reduction::Reduction,
  jacobian_reduction)

  cache = RBOnlineCache(nothing,nothing)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction,cache)
end

function RBSolver(
  fesolver::FESolver,
  state_reduction::Reduction;
  nparams_res=20,
  nparams_jac=20)

  red_style = ReductionStyle(state_reduction)
  residual_reduction = MDEIMReduction(red_style;nparams=nparams_res)
  jacobian_reduction = MDEIMReduction(red_style;nparams=nparams_jac)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function RBSolver(fesolver::FESolver,args...;nparams_state=50,kwargs...)
  state_reduction = PODReduction(args...;nparams=nparams_state)
  RBSolver(fesolver,state_reduction,kwargs...)
end

get_fe_solver(s::RBSolver) = s.fesolver
get_state_reduction(s::RBSolver) = s.state_reduction
get_residual_reduction(s::RBSolver) = s.residual_reduction
get_jacobian_reduction(s::RBSolver) = s.jacobian_reduction

num_state_params(s::RBSolver) = num_params(s.state_reduction)
num_res_params(s::RBSolver) = num_params(s.residual_reduction)
num_jac_params(s::RBSolver) = num_params(s.jacobian_reduction)

num_offline_params(s::RBSolver) = max(num_state_params(s),num_res_params(s),num_jac_params(s))
offline_params(s::RBSolver) = 1:num_offline_params(s)
res_params(s::RBSolver) = 1:num_res_params(s)
jac_params(s::RBSolver) = 1:num_jac_params(s)

"""
    solution_snapshots(solver::RBSolver,op::ParamFEOperator;kwargs...) -> AbstractSteadySnapshots
    solution_snapshots(solver::RBSolver,op::TransientParamFEOperator;kwargs...) -> AbstractTransientSnapshots

The problem is solved several times, and the solution snapshots are returned along
with the information related to the computational expense of the FE method

"""
function solution_snapshots(
  solver::RBSolver,
  op::Union{ParamFEOperator,ParamOperator};
  nparams=num_offline_params(solver),
  r=realization(op;nparams))

  fesolver = get_fe_solver(solver)
  index_map = get_vector_index_map(op)
  values,stats = solve(fesolver,op;r)
  snaps = Snapshots(values,index_map,r)
  return snaps,stats
end

function residual_snapshots(solver::RBSolver,op::ParamOperator,s)
  fesolver = get_fe_solver(solver)
  sres = select_snapshots(s,res_params(solver))
  us_res = get_values(sres)
  r_res = get_realization(sres)
  b = residual(fesolver,op,r_res,us_res)
  ib = get_vector_index_map(op)
  return Snapshots(b,ib,r_res)
end

function jacobian_snapshots(solver::RBSolver,op::ParamOperator,s)
  fesolver = get_fe_solver(solver)
  sjac = select_snapshots(s,jac_params(solver))
  us_jac = get_values(sjac)
  r_jac = get_realization(sjac)
  A = jacobian(fesolver,op,r_jac,us_jac)
  iA = get_matrix_index_map(op)
  return Snapshots(A,iA,r_jac)
end
