function solve(
  solver::ODESolver,
  op::ParamTransientFEOperator,
  params::Table)

  ode_op = get_algebraic_operator(op)
  cache = get_solution_cache(solver,op)
  T = op.test.vector_type
  sol = Vector{T}(undef,length(get_times(solver)))
  sols = Vector{typeof(sol)}(undef,length(params))
  for (k,μ) in enumerate(params)
    sols[k] = solve!(sol,solver,ode_op,μ,solver.uh0(μ),cache)
  end
end

function solve!(
  sol::AbstractArray,
  solver::ODESolver,
  ode_op::ParamODEOperator,
  μ::AbstractArray,
  u0,
  cache)

  ode_cache,vec_cache = cache
  times = get_times(solver)
  uF = similar(vec_cache)
  @inbounds for (n,tF) = enumerate(times)
    uF,ode_cache = solve_step!(uF,solver,ode_op,μ,u0,tF,ode_cache)
    u0 = sol_update(u0,uF)
    sol[n] = postprocess(ode_op,uF)
  end
end

function sol_update(u0,uF)
  @. u0 = uF
  u0
end

function sol_update(u0::Tuple,uF::Tuple)
  for i in eachindex(uF)
    @. u0[i] = uF[i]
  end
  u0
end

function get_solution_cache(solver::ODESolver,op::ParamTransientFEOperator)
  ode_cache = nothing
  μ = realization(op)
  xh0 = solver.uh0(μ)
  if isa(ic,Tuple)
    vec_cache = ()
    for xhi in xh0
      vec_i = get_free_dof_values(xhi)
      vec_cache = (vec_cache...,vec_i)
    end
  else
    vec_cache = get_free_dof_values(xh0)
  end
  return ode_cache,vec_cache
end

function postprocess(op::ParamODEOperator,uF::AbstractArray)
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
