function Arrays.return_type(
  ::typeof(solve),
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::AbstractVector,
  uh0::Any)

  u0 = get_free_dof_values(uh0)
  u0p = postprocess(op,u0)
  return typeof(fill(u0p))
end

function Arrays.return_cache(
  ::typeof(solve),
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::AbstractVector,
  uh0)

  _get_u(uh0) = uh0
  _get_u(uh0::Tuple) = first(uh0)

  nt = get_time_ndofs(solver)
  u0 = get_free_dof_values(uh0)
  u0p = postprocess(op,u0)
  uFp = copy(u0p)
  solc = u0p,uFp,nothing
  sola = fill(u0p,nt)
  return solc,sola
end

function Arrays.return_cache(
  ::typeof(solve),
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::AbstractVector,
  uh0::Tuple{Vararg{Any}})

  nt = get_time_ndofs(solver)
  x0 = ()
  for xhi in xh0
    x0 = (x0...,get_free_dof_values(xhi))
  end
  u0p = postprocess(op,u0)
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
  return sols,cache
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
  return sols,cache
end

function solve(
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::AbstractVector,
  uh0::Any,
  cache)

  ode_op = get_algebraic_operator(op)
  u0 = get_free_dof_values(uh0)
  solve(solver,ode_op,μ,u0,cache)
end

function solve(
  solver::ODESolver,
  op::ParamTransientFEOperator,
  μ::AbstractVector,
  xh0::Tuple{Vararg{Any}},
  cache)

  ode_op = get_algebraic_operator(op)
  x0 = ()
  for xhi in xh0
    x0 = (x0...,get_free_dof_values(xhi))
  end
  solve(solver,ode_op,μ,x0,cache)
end

function solve(
  solver::ODESolver,
  op::ParamODEOperator,
  μ::AbstractVector,
  u0::T,
  cache) where {T<:AbstractVector}

  times = get_times(solver)
  uF = copy(u0)
  sols = Vector{T}(undef,length(times))
  @inbounds for (n,tF) = enumerate(times)
    uF,cache = solve_step!(uF,solver,op,μ,u0,tF,cache)
    @. u0 = uF
    sols[n] = postprocess(op,uF)
  end
  return sols,cache
end

function solve(
  solver::ODESolver,
  op::ParamODEOperator,
  μ::AbstractVector,
  u0::T,
  cache) where {T<:Tuple{Vararg{AbstractVector}}}

  times = get_times(solver)
  uF = ()
  for i in eachindex(u0)
    uF = (uF...,copy(u0[i]))
  end
  sols = Vector{T}(undef,length(times))
  @inbounds for (n,tF) = enumerate(times)
    uF,cache = solve_step!(uF,solver,op,μ,u0,tF,cache)
    for i in eachindex(uF)
      @. u0[i] = uF[i]
    end
    sols[n] = postprocess(op,uF)
  end
  return sols,cache
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
