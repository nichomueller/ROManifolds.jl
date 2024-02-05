struct RBSolver{S}
  info::RBInfo
  fesolver::S
end

get_fe_solver(s::RBSolver) = s.fesolver
get_info(s::RBSolver) = s.info

function RBSolver(fesolver,dir;kwargs...)
  info = RBInfo(dir;kwargs...)
  RBSolver(info,fesolver)
end

function get_method_operator(
  solver::RBSolver{ThetaMethod},
  op::RBOperator,
  r::TransientParamRealization)

  fesolver = get_fe_solver(solver)
  dt = fesolver.dt
  θ = fesolver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ

  x,y = _init_free_values(op,r)

  ode_cache = allocate_cache(op,r)
  ode_cache = update_cache!(ode_cache,op,r)

  ThetaMethodParamOperator(op.feop,r,dtθ,x,ode_cache,y)
end

function collect_residuals_and_jacobians(solver::RBSolver,op::RBOperator)
  nparams = num_mdeim_params(solver.info)
  r = realization(op.feop;nparams)

  nlop = get_method_operator(solver,op,r)
  x = nlop.u0

  b = residual(nlop,x)
  A = jacobian(nlop,x)

  return b,A
end

function Algebra.solve(solver::RBSolver,op::RBOperator{<:LinearSolver},r::ParamRealization)
  cache = allocate_cache(op,r)
  x0 = zero_initial_guess(op,r)
  A = jacobian(op,r,x0,cache)
  b = residual(op,r,x0,cache)

  ss = symbolic_setup(solver.fesolver,A)
  ns = numerical_setup(ss,A)

  x = allocate_in_domain(A)
  fill!(x,0.0)
  solve!(x,ns,b)

  return x
end

function Algebra.solve(solver::RBSolver,op::RBOperator,r::ParamRealization)
  cache = allocate_cache(op,r)
  x0 = zero_initial_guess(op,r)
  A = jacobian(op,r,x0,cache)
  b = residual(op,r,x0,cache)
  dx = similar(b)

  ss = symbolic_setup(solver.fesolver,A)
  ns = numerical_setup(ss,A)

  x = allocate_in_domain(A)
  fill!(x,0.0)
  Algebra._solve_nr!(x,A,b,dx,ns,nls,op)

  return x
end
