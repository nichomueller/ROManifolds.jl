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

function collect_matrices_vectors!(
  solver::ThetaRBSolver{Affine},
  op::RBOperator,
  s::AbstractTransientSnapshots,
  cache)

  fesolver = get_fe_solver(solver)
  dt = fesolver.dt
  θ = fesolver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ
  r = get_realization(s)

  if isnothing(cache)
    vθ = similar(s.values)
    vθ .= 0.0
    A,b = _allocate_matrix_and_vector(op,r)
    ode_cache = allocate_cache(op,r)
  else
    A,b,ode_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op.feop,r)

  sA,sb = ODETools._matrix_and_vector!(A,b,op,r,dtθ,vθ,ode_cache,vθ)
  cache = A,b,ode_cache,vθ

  return sA,sb,cache
end

function collect_matrices_vectors!(
  solver::ThetaRBSolver,
  op::RBOperator,
  s::AbstractTransientSnapshots,
  cache)

  fesolver = get_fe_solver(solver)
  dt = fesolver.dt
  θ = fesolver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ
  r = get_realization(s)

  if isnothing(cache)
    vθ = similar(s.values)
    vθ .= 0.0
    A,b = _allocate_matrix_and_vector(op,r)
    ode_cache = allocate_cache(op,r)
  else
    A,b,ode_cache,vθ = cache
  end

  ode_cache = update_cache!(ode_cache,op.feop,r)

  sθ = shift_time!(s,dt,θ)
  uθ = sθ.values
  sA,sb = ODETools._matrix_and_vector!(A,b,op,r,dtθ,uθ,ode_cache,vθ)
  cache = A,b,ode_cache,vθ

  return sA,sb,cache
end

function reduced_operator(
  solver::RBSolver,
  op::RBOperator,
  s::AbstractTransientSnapshots)

  nparams = num_mdeim_params(solver.info)
  smdeim = select_snapshots(s,Base.OneTo(nparams))
  (red_mat,red_mat_t),red_vec = reduced_matrix_vector_form(solver,op,smdeim)
  trians_mat = get_domains(red_mat)
  trians_mat_t = get_domains(red_mat_t)
  trians_vec = get_domains(red_vec)
  red_op = reduced_operator(op,trians_vec,trians_mat,trians_mat_t)
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
