"""Recursive creation of a directory"""
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

struct RBSolver{S,M}
  fesolver::S
  ϵ::Float64
  mdeim_style::M
  nsnaps_state::Int
  nsnaps_mdeim::Int
  nsnaps_test::Int
end

function RBSolver(
  fesolver::FESolver,
  ϵ::Float64;
  mdeim_style=SpaceMDEIM(),
  nsnaps_state=50,
  nsnaps_mdeim=20,
  nsnaps_test=10)

  RBSolver(fesolver,ϵ,mdeim_style,nsnaps_state,nsnaps_mdeim,nsnaps_test)
end

get_fe_solver(s::RBSolver) = s.fesolver
num_offline_params(solver::RBSolver) = solver.nsnaps_state
offline_params(solver::RBSolver) = 1:num_offline_params(solver)
num_online_params(solver::RBSolver) = solver.nsnaps_test
online_params(solver::RBSolver) = 1+num_offline_params(solver):num_online_params(solver)+num_offline_params(solver)
ParamDataStructures.num_params(solver::RBSolver) = num_offline_params(solver) + num_online_params(solver)
num_mdeim_params(solver::RBSolver) = solver.nsnaps_mdeim
mdeim_params(solver::RBSolver) = 1:num_mdeim_params(solver)
get_tol(solver::RBSolver) = solver.ϵ

function get_test_directory(solver::RBSolver;dir=datadir())
  keyword = get_mdeim_style_filename(solver.mdeim_style)
  test_dir = joinpath(dir,keyword * "_$(solver.ϵ)")
  create_dir(test_dir)
  test_dir
end

function fe_solutions(solver::RBSolver,op::ParamFEOperator;nparams=50,r=realization(op;nparams))
  fesolver = get_fe_solver(solver)
  index_map = get_vector_index_map(op)

  stats = @timed begin
    values = solve(fesolver,op)
  end

  snaps = Snapshots(values,index_map,r)
  cs = ComputationalStats(stats,nparams)
  return snaps,cs
end

function Algebra.solve(rbsolver::RBSolver,feop,args...;kwargs...)
  fesnaps,festats = fe_solutions(rbsolver,feop,args...)
  rbop = reduced_operator(rbsolver,feop,fesnaps)
  rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
  results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)
  return results
end

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

    maxk < 1e-5*max0 && return

    if k == nls.max_nliters
      @unreachable
    end
  end
end
