struct RBSolver{S}
  info::RBInfo
  fesolver::S
end

const RBThetaMethod = RBSolver{ThetaMethod}

get_fe_solver(s::RBSolver) = s.fesolver
get_info(s::RBSolver) = s.info

function RBSolver(fesolver,dir;kwargs...)
  info = RBInfo(dir;kwargs...)
  RBSolver(info,fesolver)
end

function reduced_operator(solver::RBSolver,op::RBOperator)
  red_mat,red_vec = reduced_matrix_vector_form(solver,op)
  trians_mat = map(red_mat) do m
    collect(keys(m))
  end
  trians_vec = collect(keys(red_vec))
  red_op = reduced_operator(op,trians_mat,trians_vec)
  return red_op
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
