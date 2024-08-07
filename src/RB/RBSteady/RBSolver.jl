"""
    create_dir(dir::String) -> Nothing

Recursive creation of a directory `dir`

"""
function create_dir(dir::String)
  if !isdir(dir)
    parent_dir, = splitdir(dir)
    create_dir(parent_dir)
    mkdir(dir)
  end
  return
end

abstract type MDEIMStyle end
struct SpaceMDEIM <: MDEIMStyle end

get_mdeim_style_filename(::SpaceMDEIM) = "space_mdeim"

"""
    struct RBSolver{S,M} end

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
  where a basis spanning the reduced subspace is computed.
- mdeim_style: in transient applications, the user can choose to perform MDEIM in
  space only or in space and time (default)
- nsnaps_state: number of snapshots considered when running TPOD or TT-SVD
- nsnaps_res: number of snapshots considered when running MDEIM for the residual
- nsnaps_jac: number of snapshots considered when running MDEIM for the jacobian
- nsnaps_test:  number of snapshots considered when computing the error the RB
  method commits with respect to the FE procedure

"""
struct RBSolver{S,M}
  fesolver::S
  ϵ::Float64
  mdeim_style::M
  nsnaps_state::Int
  nsnaps_res::Int
  nsnaps_jac::Int
  nsnaps_test::Int
end

function RBSolver(
  fesolver::FESolver,
  ϵ::Float64;
  mdeim_style=SpaceMDEIM(),
  nsnaps_state=50,
  nsnaps_res=20,
  nsnaps_jac=20,
  nsnaps_test=10)

  RBSolver(fesolver,ϵ,mdeim_style,nsnaps_state,nsnaps_res,nsnaps_jac,nsnaps_test)
end

get_fe_solver(s::RBSolver) = s.fesolver
num_offline_params(solver::RBSolver) = max(solver.nsnaps_state,num_mdeim_params(solver))
offline_params(solver::RBSolver) = 1:num_offline_params(solver)
num_online_params(solver::RBSolver) = solver.nsnaps_test
online_params(solver::RBSolver) = 1+num_offline_params(solver):num_online_params(solver)+num_offline_params(solver)
ParamDataStructures.num_params(solver::RBSolver) = num_offline_params(solver) + num_online_params(solver)
num_res_params(solver::RBSolver) = solver.nsnaps_res
res_params(solver::RBSolver) = 1:num_res_params(solver)
num_jac_params(solver::RBSolver) = solver.nsnaps_jac
jac_params(solver::RBSolver) = 1:num_jac_params(solver)
num_mdeim_params(solver::RBSolver) = max(num_res_params(solver),num_jac_params(solver))
mdeim_params(solver::RBSolver) = 1:num_mdeim_params(solver)
get_tol(solver::RBSolver) = solver.ϵ

function get_test_directory(solver::RBSolver;dir=datadir())
  keyword = get_mdeim_style_filename(solver.mdeim_style)
  test_dir = joinpath(dir,keyword * "_$(solver.ϵ)")
  create_dir(test_dir)
  test_dir
end

"""
    fe_solutions(solver::RBSolver,op::ParamFEOperator;kwargs...
      ) -> (AbstractSteadySnapshots, ComputationalStats)
    fe_solutions(solver::RBSolver,op::TransientParamFEOperator;kwargs...
      ) -> (AbstractTransientSnapshots, ComputationalStats)

The problem is solved several times, and the solution snapshots are returned along
with the information related to the computational expense of the FE method

"""
function fe_solutions(solver::RBSolver,op::ParamFEOperator;nparams=50,r=realization(op;nparams))
  fesolver = get_fe_solver(solver)
  index_map = get_vector_index_map(op)
  values,cost = solve(fesolver,op)
  snaps = Snapshots(values,index_map,r)
  return snaps,cost
end

function Algebra.solve(rbsolver::RBSolver,feop,args...;kwargs...)
  fesnaps,festats = fe_solutions(rbsolver,feop,args...)
  rbop = reduced_operator(rbsolver,feop,fesnaps)
  rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
  results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)
  return results
end

"""
    nonlinear_rb_solve!(x̂,x,A,b,A_cache,b_cache,dx̂,ns,nls,op,trial) -> x̂

Newton - Raphson for a RB problem

"""
function nonlinear_rb_solve!(x̂,x,A,b,A_cache,b_cache,dx̂,ns,nls,op,trial)
  A_lin, = A_cache
  max0 = maximum(abs,b)

  for k in 1:nls.max_nliters
    rmul!(b,-1)
    solve!(dx̂,ns,b)
    x̂ .+= dx̂
    x .= recast(x̂,trial)

    b = residual!(b_cache,op,x)

    A = jacobian!(A_cache,op,x)
    numerical_setup!(ns,A)

    b .+= A_lin*x̂
    maxk = maximum(abs,b)
    println(maxk)

    maxk < 1e-6*max0 && return

    if k == nls.max_nliters
      @unreachable
    end
  end
end
