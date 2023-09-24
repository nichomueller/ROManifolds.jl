struct PODESolution{T}
  solver::ODESolver
  op::PODEOperator
  μ::AbstractVector
  u0::PTArray
  t0::Real
  tf::Real

  function PODESolution(
    solver::ODESolver,
    op::PODEOperator,
    μ::AbstractVector,
    u0::PTArray,
    t0::Real,
    tf::Real)

    test = get_test(op.feop)
    T = typeof(test)
    new{T}(solver,op,μ,u0,t0,tf)
  end
end

function Base.iterate(sol::PODESolution)
  uf = copy(sol.u0)
  u0 = copy(sol.u0)
  t0 = sol.t0
  n = 0
  cache = nothing

  uf,tf,cache = solution_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  u0 .= uf
  n += 1
  state = (uf,u0,tf,n,cache)

  return (uf,tf,n),state
end

function Base.iterate(sol::PODESolution,state)
  uf,u0,t0,n,cache = state

  if t0 >= sol.tf - 100*eps()
    return nothing
  end

  uf,tf,cache = solution_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  u0 .= uf
  n += 1
  state = (uf,u0,tf,n,cache)

  return (uf,tf,n),state
end

function collect_solutions(
  solver::ODESolver,
  op::PTFEOperator,
  uh0::Function,
  μ::Table,
  t0::Real,
  tf::Real)

  ode_op = get_algebraic_operator(op)
  u0 = get_free_dof_values(uh0(μ))
  uμt = PODESolution(solver,ode_op,μ,u0,t0,tf)
  num_iter = Int(tf/solver.dt)
  solutions = allocate_solution(ode_op,num_iter)
  for (u,t,n) in uμt
    printstyled("Computing fe solution at time $t for every parameter\n";color=:blue)
    solutions[n] = copy(get_solution(ode_op,u))
  end
  return solutions
end

function collect_residuals(
  solver::ThetaMethod,
  op::PTFEOperator,
  uh0::Function,
  sols::Vector{<:PTArray},
  μ::Table,
  t0::Real,
  tf::Real)

  dt,θ = solver.dt,solver.θ
  dtθ = θ == 0.0 ? dt : dt*θ
  times = t0:dt:(tf-dt) .+ dtθ
  ode_op = get_algebraic_operator(op)
  u0 = get_free_dof_values(uh0(μ))
  solsθ = copy(sols)
  solsθ .= sols*θ + hcat(u0,sols[2:end]...)*(1-θ)
  ode_cache = allocate_cache(ode_op,μ,times)
  ode_cache = update_cache!(ode_cache,ode_op,μ,times)
  nlop = PThetaMethodNonlinearOperator(ode_op,μ,times,dtθ,solsθ,ode_cache,solsθ)

  printstyled("Computing fe residuals for every time and parameter\n";color=:blue)
  ress = residual(nlop,solsθ)
  return NnzMatrix(ress)
end

function collect_jacobians(
  solver::ThetaMethod,
  op::PTFEOperator,
  uh0::Function,
  sols::Vector{<:PTArray},
  μ::Table,
  t0::Real,
  tf::Real)

  dt,θ = solver.dt,solver.θ
  dtθ = θ == 0.0 ? dt : dt*θ
  times = t0:dt:(tf-dt) .+ dtθ
  ode_op = get_algebraic_operator(op)
  u0 = get_free_dof_values(uh0(μ))
  sols_prev = attach_initial_condition(sols,u0)
  solsθ = copy(sols)
  solsθ.nonzero_val .= sols.nonzero_val*θ + sols_prev*(1-θ)
  ode_cache = allocate_cache(ode_op,μ,times)
  ode_cache = update_cache!(ode_cache,ode_op,μ,times)
  nlop = PThetaMethodNonlinearOperator(ode_op,μ,times,dtθ,solsθ,ode_cache,solsθ)

  nnz_jacs = map(eachindex(op.jacs)) do i
    printstyled("Computing fe jacobian #$i for every time and parameter\n";color=:blue)
    jacs_i = jacobian(nlop,solsθ,i)
    nnz_jacs_i = map(NnzVector,jacs_i)
    NnzMatrix(nnz_jacs_i)
  end
  return nnz_jacs
end

for f in (:allocate_solution,:get_solution)
  @eval begin
    function $f(op::PODEOperator,args...)
      $f(get_test(op.feop),args...)
    end
  end
end

function allocate_solution(::FESpace,niter::Int)
  Vector{PTArray}(undef,niter)
end

function allocate_solution(::MultiFieldFESpace,niter::Int)
  Vector{Vector{PTArray}}(undef,niter)
end

function get_solution(::FESpace,free_values::PTArray)
  free_values
end

function get_solution(fe::MultiFieldFESpace,free_values::PTArray)
  blocks = map(1:length(fe.spaces)) do i
    restrict_to_field(fe,free_values,i)
  end
  PTArray.(blocks)
end
