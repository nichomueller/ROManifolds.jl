function collect_solutions(op,solver,μ,u0)
  ode_op = get_algebraic_operator(op)
  uμt = PODESolution(solver,ode_op,μ,u0,t0,tF)
  solutions = PTArray[]
  for (u,t) in uμt
    printstyled("Computing fe solution at time $t for every parameter\n";color=:blue)
    push!(solutions,u)
  end
  return solutions
end

struct PODESolution
  solver::ODESolver
  op::PODEOperator
  μ::AbstractVector
  u0::PTArray
  t0::Real
  tF::Real
end

function solve_step!(
  uf::PTArray,
  solver::ThetaMethod,
  op::PODEOperator,
  μ::AbstractVector,
  u0::PTArray,
  t0::Real)

  solve_step!(uf,solver,op,μ,u0,t0,nothing)
end

function Base.iterate(sol::PODESolution)
  uf = copy(sol.u0)
  u0 = copy(sol.u0)
  t0 = sol.t0

  uf,tf,cache = solve_step!(uf,sol.solver,sol.op,sol.μ,u0,t0)

  u0.array .= uf.array
  state = (uf,u0,tf,cache)

  return (uf,tf),state
end

function Base.iterate(sol::PODESolution,state)
  uf,u0,t0,cache = state

  if t0 >= sol.tF - ϵ
    return nothing
  end

  uf,tf,cache = solve_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  u0.array .= uf.array
  state = (uf,u0,tf,cache)

  return (uf,tf),state
end

function postprocess(op::PODEOperator,uF::AbstractArray)
  Uh = get_trial(op.feop)
  Uh0 = allocate_trial_space(Uh)
  if isa(Uh0,MultiFieldFESpace)
    blocks = map(1:length(Uh0.spaces)) do i
      MultiField.restrict_to_field(Uh0,uF,i)
    end
    return mortar(blocks)
  else
    return uF
  end
end
