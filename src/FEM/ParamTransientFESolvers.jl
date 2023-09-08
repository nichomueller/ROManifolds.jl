function Arrays.return_type(
  ::PTMap,
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::AbstractVector,
  uh0::Any)

  ode_op = get_algebraic_operator(op)
  u0 = get_free_dof_values(uh0)
  u0p = postprocess(ode_op,u0)
  return typeof(fill(u0p))
end

function Arrays.return_cache(
  ::PTMap,
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::AbstractVector,
  uh0)

  nt = get_time_ndofs(solver)
  ode_op = get_algebraic_operator(op)
  u0 = get_free_dof_values(uh0)
  u0p = postprocess(ode_op,u0)
  uFp = copy(u0p)
  solc = u0p,uFp,nothing
  sola = fill(u0p,nt)
  return solc,sola
end

function Arrays.return_cache(
  ::PTMap,
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::AbstractVector,
  uh0::Tuple{Vararg{Any}})

  nt = get_time_ndofs(solver)
  ode_op = get_algebraic_operator(op)
  x0 = ()
  for xhi in xh0
    x0 = (x0...,get_free_dof_values(xhi))
  end
  u0p = postprocess(ode_op,u0)
  uFp = copy(u0p)
  solc = u0p,uFp,nothing
  sola = fill(u0p,nt)
  return solc,sola
end

function Arrays.evaluate!(
  cache,
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::AbstractVector,
  ::Any)

  solc, = cache
  u0,uF,_cache = solc
  ode_op = get_algebraic_operator(op)
  times = get_times(solver)
  @inbounds for (n,tF) = enumerate(times)
    uF,_cache = solve_step!(uF,solver,ode_op,μ,u0,tF,_cache)
    @. u0 = uF
    sola[n] = postprocess(op,uF)
  end
  return sola
end

function Arrays.evaluate!(
  cache,
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::AbstractVector,
  ::Tuple{Vararg{Any}})

  solc, = cache
  u0,uF,_cache = solc
  ode_op = get_algebraic_operator(op)
  times = get_times(solver)
  @inbounds for (n,tF) = enumerate(times)
    uF,_cache = solve_step!(uF,solver,ode_op,μ,u0,tF,_cache)
    for i in eachindex(uF)
      @. u0[i] = uF[i]
    end
    sola[n] = postprocess(op,uF)
  end
  return sola
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
