abstract type ParamODESolution <: GridapType end

struct GenericParamODESolution{T} <: ParamODESolution
  solver::ODESolver
  op::ParamODEOperator
  μ::Vector{Float}
  u0::T
  t0::Real
  tF::Real
end

function Gridap.ODEs.ODETools.solve_step!(
  uF::Union{AbstractVector,Tuple{Vararg{AbstractVector}}},
  solver::ODESolver,
  op::ParamODEOperator,
  μ::Vector{Float},
  u0::Union{AbstractVector,Tuple{Vararg{AbstractVector}}},
  t0::Real) # -> (uF,tF,cache)

  Gridap.ODEs.ODETools.solve_step!(uF,solver,op,μ,u0,t0,nothing)
end

function Gridap.ODEs.ODETools.solve(
  solver::ODESolver,
  op::ParamODEOperator,
  μ::Vector{Float},
  u0::T,
  t0::Real,
  tf::Real) where {T}

  GenericParamODESolution{T}(solver,op,μ,u0,t0,tf)
end

function Base.iterate(
  sol::GenericParamODESolution{T}) where {T<:AbstractVector}

  uf = copy(sol.u0)
  u0 = copy(sol.u0)
  t0 = sol.t0

  # Solve step
  uf,tf,cache = Gridap.ODEs.ODETools.solve_step!(uf,sol.solver,sol.op,sol.μ,u0,t0)

  # Update
  u0 .= uf
  state = (uf,u0,tf,cache)

  return (uf, tf), state
end

function Base.iterate(
  sol::GenericParamODESolution{T},
  state) where {T<:AbstractVector}

  uf,u0,t0,cache = state

  if t0 >= sol.tF - 100*eps()
    return nothing
  end

  # Solve step
  uf,tf,cache = Gridap.ODEs.ODETools.solve_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  # Update
  u0 .= uf
  state = (uf,u0,tf,cache)

  return (uf,tf),state
end

function Base.iterate(
  sol::GenericParamODESolution{T}) where {T<:Tuple{Vararg{AbstractVector}}}

  uf = ()
  u0 = ()
  for i in 1:length(sol.u0)
    uf = (uf...,copy(sol.u0[i]))
    u0 = (u0...,copy(sol.u0[i]))
  end
  t0 = sol.t0

  # Solve step
  uf,tf,cache = Gridap.ODEs.ODETools.solve_step!(uf,sol.solver,sol.op,sol.μ,u0,t0)

  # Update
  for i in 1:length(uf)
    u0[i] .= uf[i]
  end
  state = (uf,u0,tf,cache)

  return (uf[1],tf),state
end

function Base.iterate(
  sol::GenericParamODESolution{T},
  state) where {T<:Tuple{Vararg{AbstractVector}}}

  uf,u0,t0,cache = state

  if t0 >= sol.tF - 100*eps()
    return nothing
  end

  # Solve step
  uf,tf,cache = Gridap.ODEs.ODETools.solve_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  # Update
  for i in 1:length(uf)
    u0[i] .= uf[i]
  end
  state = (uf,u0,tf,cache)

  return (uf[1],tf),state
end

"""
It represents a FE function at a set of time steps. It is a wrapper of a ODE
solution for free values combined with data for Dirichlet values. Thus, it is a
lazy iterator that computes the solution at each time step when accessing the
solution.
"""
struct ParamTransientFESolution
  odesol::ParamODESolution
  trial
end

function ParamTransientFESolution(
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::Vector{Float},
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
  μ::Vector{Float},
  uh0,
  t0::Real,
  tF::Real)

  ode_op = get_algebraic_operator(op)
  u0 = get_free_dof_values(uh0)
  ode_sol = solve(solver,ode_op,μ,u0,t0,tF)
  trial = get_trial(op)

  ParamTransientFESolution(ode_sol,trial)
end

function Gridap.ODEs.ODETools.solve(
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::Vector{Vector{Float}},
  u0,
  t0::Real,
  tF::Real)

  Broadcasting(p -> ParamTransientFESolution(solver,op,p,u0,t0,tF))(μ)
end

function Gridap.ODEs.ODETools.solve(
  solver::ODESolver,
  op::ParamTransientFEOperator,
  t0::Real,
  tF::Real,
  n=100)

  μ = realization(op,n)
  trial = get_trial(op)
  u0 = zero(trial(first(μ),t0))
  solve(solver,op,μ,u0,t0,tF)
end

function Base.iterate(sol::ParamTransientFESolution)

  odesolnext = iterate(sol.odesol)

  if isnothing(odesolnext)
    return nothing
  end

  (uf,tF),odesolstate = odesolnext

  Uh = allocate_trial_space(sol.trial)
  Uh = evaluate!(Uh,sol.trial,sol.odesol.μ,tF)
  uh = FEFunction(Uh,uf)

  state = (Uh,odesolstate)

  (uh,tF),state
end

function Base.iterate(sol::ParamTransientFESolution, state)

  Uh, odesolstate = state

  odesolnext = iterate(sol.odesol,odesolstate)

  if isnothing(odesolnext)
    return nothing
  end

  (uf,tF),odesolstate = odesolnext

  Uh = evaluate!(Uh,sol.trial,sol.odesol.μ,tF)
  uh = FEFunction(Uh,uf)

  state = (Uh,odesolstate)

  (uh,tF),state
end
