struct ParamODESolution{T} <: GridapType
  solver::ODESolver
  op::ParamODEOperator
  μ::AbstractVector
  u0::T
  t0::Real
  tF::Real
end

function Base.length(sol::ParamODESolution)
  get_time_ndofs(sol.solver)
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

  ParamODESolution{T}(solver,op,μ,u0,t0,tF)
end

function Base.iterate(
  sol::ParamODESolution{T}) where {T<:AbstractVector}

  uF = copy(sol.u0)
  u0 = copy(sol.u0)
  t0 = sol.t0

  # Solve step
  uF,tF,cache = solve_step!(uF,sol.solver,sol.op,sol.μ,u0,t0)

  # Update
  u0 .= uF
  state = (uF,u0,tF,cache)
  uF = postprocess(sol,uF)

  return (uF,tF),state
end

function Base.iterate(
  sol::ParamODESolution{T},
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
  uF = postprocess(sol,uF)

  return (uF,tF),state
end

function Base.iterate(
  sol::ParamODESolution{T}) where {T<:Tuple{Vararg{AbstractVector}}}

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
  uF1 = postprocess(sol,uF[1])

  return (uF1,tF),state
end

function Base.iterate(
  sol::ParamODESolution{T},
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
  uF1 = postprocess(sol,uF[1])

  return (uF1,tF),state
end

function Algebra.solve(
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::AbstractVector,
  uh0,
  t0::Real,
  tF::Real)

  ode_op = get_algebraic_operator(op)
  u0 = get_free_dof_values(uh0)
  solve(solver,ode_op,μ,u0,t0,tF)
end

function Algebra.solve(
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
  solve(solver,ode_op,μ,x0,t0,tF)
end

function Arrays.return_value(sol::ParamODESolution)
  sol1 = iterate(sol)
  (uh1,_),_ = sol1
  uh1
end

function postprocess(sol::ParamODESolution,uF::AbstractArray)
  Uh = allocate_trial_space(get_trial(sol.op.feop))
  if isa(Uh,MultiFieldFESpace)
    blocks = map(1:length(Uh.spaces)) do i
      MultiField.restrict_to_field(Uh,uF,i)
    end
    return mortar(blocks)
  else
    return uF
  end
end
