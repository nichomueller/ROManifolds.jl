abstract type ParamODESolution <: GridapType end

struct GenericParamODESolution{T} <: ParamODESolution
  op::ParamODEOperator
  solver::ODESolver
  μ::AbstractVector
  u0::T
  t0::Real
  tF::Real
  k::Int
end

function Gridap.ODEs.TransientFETools.solve_step!(
  uF::Union{AbstractVector,Tuple{Vararg{AbstractVector}}},
  op::ParamODEOperator,
  solver::ODESolver,
  μ::AbstractVector,
  u0::Union{AbstractVector,Tuple{Vararg{AbstractVector}}},
  t0::Real) # -> (uF,tF,cache)

  solve_step!(uF,op,solver,μ,u0,t0,nothing)
end

function Gridap.FESpaces.solve(
  op::ParamODEOperator,
  solver::ODESolver,
  μ::AbstractVector,
  u0::T,
  k::Int) where T

  t0,tF = solver.t0,solver.tF
  GenericParamODESolution{T}(op,solver,μ,u0,t0,tF,k)
end

function Gridap.FESpaces.solve(
  op::ParamTransientFEOperator,
  solver::ODESolver,
  μ::AbstractVector,
  uh0,
  k::Int)

  ode_op = get_algebraic_operator(op)
  u0 = get_free_dof_values(uh0)
  solve(solver,ode_op,μ,u0,k)
end

function Gridap.FESpaces.solve(
  op::ParamTransientFEOperator,
  solver::ODESolver,
  params::Table)

  uh0 = solver.uh0
  [solve(op,solver,μk,uh0(μk),k) for (k,μk) in enumerate(params)]
end

function Base.iterate(
  sol::GenericParamODESolution{<:AbstractVector})

  uF = copy(sol.u0)
  u0 = copy(sol.u0)
  t0 = sol.t0

  # Solve step
  uF,tF,cache = solve_step!(uF,sol.op,sol.solver,sol.μ,u0,t0)

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
  uF,tF,cache = solve_step!(uF,sol.op,sol.solver,sol.μ,u0,t0,cache)

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

function _split_solutions(trial::MultiFieldFESpace,u::AbstractVector)
  map(1:length(trial.spaces)) do i
    restrict_to_field(trial,u,i)
  end
end

function solution_cache(test::FESpace,solver::ODESolver)
  space_ndofs = test.nfree
  time_ndofs = get_time_ndofs(solver)
  cache = fill(1.,space_ndofs,time_ndofs)
  NnzArray(cache)
end

function solution_cache(test::MultiFieldFESpace,solver::ODESolver)
  map(t->solution_cache(t,solver),test.spaces)
end

function collect_solution!(cache,sol::ParamODESolution)
  printstyled("Computing snapshot $(sol.k)\n";color=:blue)
  n = 1
  if isa(cache,NnzArray)
    for soln in sol
      setindex!(cache,soln,:,n)
      n += 1
    end
  else
    for soln in sol
      map((cache,sol) -> setindex!(cache,sol,:,n),cache,soln)
      n += 1
    end
  end
  printstyled("Successfully computed snapshot $(sol.k)\n";color=:blue)

  cache
end
