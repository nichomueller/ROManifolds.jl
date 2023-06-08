abstract type ParamODESolution <: GridapType end

struct GenericParamODESolution{T} <: ParamODESolution
  solver::ODESolver
  op::ParamODEOperator
  μ::AbstractVector
  u0::T
  t0::Real
  tF::Real
  k::Int
end

function Gridap.ODEs.TransientFETools.solve_step!(
  uF::Union{AbstractVector,Tuple{Vararg{AbstractVector}}},
  solver::ODESolver,
  op::ParamODEOperator,
  μ::AbstractVector,
  u0::Union{AbstractVector,Tuple{Vararg{AbstractVector}}},
  t0::Real) # -> (uF,tF,cache)

  solve_step!(uF,solver,op,μ,u0,t0,nothing)
end

function Gridap.FESpaces.solve(
  solver::ODESolver,
  op::ParamODEOperator,
  μ::AbstractVector,
  u0::T,
  k::Int) where T

  t0,tF = solver.t0,solver.tF
  GenericParamODESolution{T}(solver,op,μ,u0,t0,tF,k)
end

function Gridap.FESpaces.solve(
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::AbstractVector,
  uh0,
  k::Int)

  ode_op = get_algebraic_operator(op)
  u0 = get_free_dof_values(uh0)
  solve(solver,ode_op,μ,u0,k)
end

function Gridap.FESpaces.solve(
  solver::ODESolver,
  op::ParamTransientFEOperator,
  params::Table)

  uh0 = solver.uh0
  [solve(solver,op,μk,uh0(μk),k) for (k,μk) in enumerate(params)]
end

function Gridap.FESpaces.solve(
  solver::ODESolver,
  op::ParamTransientFEOperator,
  n_snap::Int)

  params = realization(op,n_snap)
  solve(solver,op,params)
end

function Base.iterate(
  sol::GenericParamODESolution{<:AbstractVector})

  uF = copy(sol.u0)
  u0 = copy(sol.u0)
  t0 = sol.t0

  # Solve step
  uF,tF,cache = solve_step!(uF,sol.solver,sol.op,sol.μ,u0,t0)

  # Update
  u0 .= uF
  state = (uF,u0,tF,cache)

  # Multi field
  Uh = _allocate_trial_space(sol)
  uh = _split_solutions(Uh,uF)

  return uh,state
end

function Base.iterate(
  sol::GenericParamODESolution{<:AbstractVector},
  state)

  uF,u0,t0,cache = state

  if t0 >= sol.tF - 100*eps()
    return nothing
  end

  # Solve step
  uF,tF,cache = solve_step!(uF,sol.solver,sol.op,sol.μ,u0,t0,cache)

  # Update
  u0 .= uF
  state = (uF,u0,tF,cache)

  # Multi field
  Uh = _allocate_trial_space(sol)
  uh = _split_solutions(Uh,uF)

  return uh,state
end

function _allocate_trial_space(sol::ParamODESolution)
  allocate_trial_space(first(sol.op.feop.trials))
end

function _split_solutions(::TrialFESpace,u::AbstractVector)
  u
end

function _split_solutions(Uh::MultiFieldFESpace,u::AbstractVector)
  map(1:length(Uh.spaces)) do i
    restrict_to_field(Uh,u,i)
  end
end
