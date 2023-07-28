abstract type ParamODESolution <: GridapType end

struct GenericParamODESolution{T} <: ParamODESolution
  solver::ODESolver
  op::ParamODEOperator
  μ::AbstractVector
  u0::T
  t0::Real
  tF::Real
end

function solve_step!(
  uF::Union{AbstractVector,Tuple{Vararg{AbstractVector}}},
  solver::ODESolver,
  op::ParamODEOperator,
  μ::AbstractVector,
  u0::Union{AbstractVector,Tuple{Vararg{AbstractVector}}},
  t0::Real) # -> (uF,tF,cache)

  solve_step!(uF,solver,op,μ,u0,t0,nothing)
end

function solve(
  solver::ODESolver,
  op::ParamODEOperator,
  μ::AbstractVector,
  u0::T,
  t0::Real,
  tF::Real) where {T}

  GenericParamODESolution{T}(solver,op,μ,u0,t0,tF)
end

function Base.iterate(
  sol::GenericParamODESolution{T}) where {T<:AbstractVector}

  uF = copy(sol.u0)
  u0 = copy(sol.u0)
  t0 = sol.t0

  # Solve step
  uF,tF,cache = solve_step!(uF,sol.solver,sol.op,sol.μ,u0,t0)

  # Update
  u0 .= uF
  state = (uF,u0,tF,cache)

  return (uF,tF),state
end

function Base.iterate(
  sol::GenericParamODESolution{T},
  state) where {T<:AbstractVector}

  uF,u0,t0,cache = state

  if t0 >= sol.tF - 100*eps()
    return nothing
  end

  # Solve step
  uF,tF,cache = solve_step!(uF,sol.solver,sol.op,sol.μ,u0,t0,cache)

  # Update
  u0 .= uF
  state = (uF,u0,tF,cache)

  return (uF,tF),state
end

function Base.iterate(
  sol::GenericParamODESolution{T}) where {T<:Tuple{Vararg{AbstractVector}}}

  uF = ()
  u0 = ()
  for i in eachindex(sol.u0)
    uF = (uF...,copy(sol.u0[i]))
    u0 = (u0...,copy(sol.u0[i]))
  end
  t0 = sol.t0

  # Solve step
  uF,tF,cache = solve_step!(uF,sol.solver,sol.op,sol.μ,u0,t0)

  # Update
  for i in eachindex(uF)
    u0[i] .= uF[i]
  end
  state = (uF,u0,tF,cache)

  return (uF[1],tF),state
end

function Base.iterate(
  sol::GenericParamODESolution{T},
  state) where {T<:Tuple{Vararg{AbstractVector}}}

  uF,u0,t0,cache = state

  if t0 >= sol.tF - 100*eps()
    return nothing
  end

  # Solve step
  uF,tF,cache = solve_step!(uF,sol.solver,sol.op,sol.μ,u0,t0,cache)

  # Update
  for i in eachindex(uF)
    u0[i] .= uF[i]
  end
  state = (uF,u0,tF,cache)

  return (uF[1],tF),state
end

"""
It represents a FE function at a set of time steps. It is a wrapper of a ODE
solution for free values combined with data for Dirichlet values. Thus, it is a
lazy iterator that computes the solution at each time step when accessing the
solution.
"""
struct ParamTransientFESolution
  psol::ParamODESolution
  trial
end

function ParamTransientFESolution(
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::AbstractVector,
  uh0,
  t0::Real,
  tF::Real)

  ode_op = get_algebraic_operator(op)
  u0 = get_free_dof_values(uh0)
  ode_sol = solve(solver,ode_op,μ,u0,t0,tF)
  trial = get_trial(op)

  ParamTransientFESolution(ode_sol,trial)
end

function ParamTransientFESolution(
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::AbstractVector,
  xh0::Tuple{Vararg{Any}},
  t0::Real,
  tF::Real)

  ode_op = get_algebraic_operator(op)
  x0 = ()
  for xhi in xh0
    x0 = (x0...,get_free_dof_values(xhi))
  end
  ode_sol = solve(solver,ode_op,μ,u0,t0,tF)
  trial = get_trial(op)

  ParamTransientFESolution(ode_sol,trial)
end

function solve(
  solver::ODESolver,
  op::ParamTransientFEOperator,
  params::Table)

  collect = CollectSolutionMap(solver,op)
  lazy_map(collect,params)
end

function Base.iterate(sol::ParamTransientFESolution)

  psolnext = iterate(sol.psol)

  if isnothing(psolnext)
    return nothing
  end

  (uF,tF),psolstate = psolnext

  Uh = allocate_trial_space(sol.trial)
  Uh = evaluate!(Uh,sol.trial,sol.psol.μ,tF)
  uh = FEFunction(Uh,uF)

  state = (Uh,psolstate)

  return (uh,tF),state
end

function Base.iterate(sol::ParamTransientFESolution,state)

  Uh,psolstate = state

  psolnext = iterate(sol.psol,psolstate)

  if isnothing(psolnext)
    return nothing
  end

  (uF,tF),psolstate = psolnext

  Uh = evaluate!(Uh,sol.trial,sol.psol.μ,tF)
  uh = FEFunction(Uh,uF)

  state = (Uh,psolstate)

  return (uh,tF),state
end

function Arrays.return_value(sol::ParamTransientFESolution)
  sol_μ1 = iterate(sol)
  (uh1,_),_ = sol_μ1
  uh1
end

function Base.length(sol::ParamTransientFESolution)
  get_time_ndofs(sol.psol.solver)
end
