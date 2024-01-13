function FESpaces.get_algebraic_operator(
  fesolver::ODESolver,
  feop::TransientPFEOperator,
  sols::PTArray,
  params::Table)

  dtθ = fesolver.θ == 0.0 ? fesolver.dt : fesolver.dt*fesolver.θ
  times = get_stencil_times(fesolver)
  ode_cache = allocate_cache(feop,params,times)
  ode_cache = update_cache!(ode_cache,feop,params,times)
  sols_cache = zero(sols)
  get_algebraic_operator(feop,params,times,dtθ,sols,ode_cache,sols_cache)
end

struct PODESolution
  solver::ODESolver
  op::TransientPFEOperator
  μ::AbstractVector
  u0::AbstractVector
  t0::Real
  tf::Real
end

Base.length(sol::PODESolution) = Int((sol.tf-sol.t0)/sol.solver.dt)

function Base.iterate(sol::PODESolution)
  uf = copy(sol.u0)
  u0 = copy(sol.u0)
  t0 = sol.t0
  n = 0
  cache = nothing

  uf,tf,cache = solve_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  u0 .= uf
  n += 1
  state = (uf,u0,tf,n,cache)

  return (uf,n),state
end

function Base.iterate(sol::PODESolution,state)
  uf,u0,t0,n,cache = state

  if t0 >= sol.tf - 100*eps()
    return nothing
  end

  uf,tf,cache = solve_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  u0 .= uf
  n += 1
  state = (uf,u0,tf,n,cache)

  return (uf,n),state
end

Algebra.symbolic_setup(s::BackslashSolver,mat::PTArray) = symbolic_setup(s,testitem(mat))

Algebra.symbolic_setup(s::BackslashSolver,mat::AbstractArray{<:PTArray}) = symbolic_setup(s,testitem(mat))

Algebra.symbolic_setup(s::LUSolver,mat::PTArray) = symbolic_setup(s,testitem(mat))

Algebra.symbolic_setup(s::LUSolver,mat::AbstractArray{<:PTArray}) = symbolic_setup(s,testitem(mat))

function Algebra._check_convergence(nls,b::PTArray,m0)
  m = maximum(abs,b)
  return all(m .< nls.tol * m0)
end
