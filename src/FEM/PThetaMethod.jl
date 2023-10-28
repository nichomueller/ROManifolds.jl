function solve_step!(
  uf::PTArray,
  solver::PThetaMethod,
  op::PODEOperator,
  μ::AbstractVector,
  u0::PTArray,
  t0::Real,
  cache)

  dt = solver.dt
  solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
  tθ = t0+dtθ

  if isnothing(cache)
    ode_cache = allocate_cache(op,μ,tθ)
    vθ = similar(u0)
    vθ .= 0.0
    nl_cache = nothing
  else
    ode_cache,vθ,nl_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,μ,tθ)

  nlop = PThetaMethodNonlinearOperator(op,μ,tθ,dtθ,u0,ode_cache,vθ)

  nl_cache = solve!(uf,solver.nls,nlop,nl_cache)

  if 0.0 < solver.θ < 1.0
    @. uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,nl_cache)
  tf = t0+dt
  return (uf,tf,cache)
end

"""
Nonlinear operator that represents the θ-method nonlinear operator at a
given time step, i.e., A(t,u_n+θ,(u_n+θ-u_n)/dt)
"""
struct PThetaMethodNonlinearOperator <: PNonlinearOperator
  odeop::PODEOperator
  μ
  tθ
  dtθ::Float
  u0::PTArray
  ode_cache
  vθ::PTArray
end

function get_nonlinear_operator(
  odeop::PODEOperator,μ,tθ,dtθ::Float,u0::PTArray,ode_cache,vθ::PTArray)
  PThetaMethodNonlinearOperator(odeop,μ,tθ,dtθ,u0,ode_cache,vθ)
end

for f in (:linear_operator,:nonlinear_operator)
  @eval begin
    function $f(op::PThetaMethodNonlinearOperator)
      feop = $f(op.odeop.feop)
      odeop = get_algebraic_operator(feop)
      return PThetaMethodNonlinearOperator(odeop,op.μ,op.tθ,op.dtθ,op.u0,op.ode_cache,op.vθ)
    end
  end
end

function Algebra.allocate_residual(
  op::PNonlinearOperator,
  x::PTArray)

  allocate_residual(op.odeop,op.μ,op.tθ,x,op.ode_cache)
end

function Algebra.allocate_jacobian(
  op::PNonlinearOperator,
  x::PTArray)

  allocate_jacobian(op.odeop,op.μ,op.tθ,x,op.ode_cache)
end

function Algebra.residual(op::PNonlinearOperator,x::PTArray,args...)
  b = allocate_residual(op,x)
  residual!(b,op,x,args...)
end

function residual!(
  b::PTArray,
  op::PThetaMethodNonlinearOperator,
  x::PTArray,
  args...)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(b))
  fill!(b,z)
  residual!(b,op.odeop,op.μ,op.tθ,(uF,vθ),op.ode_cache,args...)
end

for (f,g) in zip((:residual_for_trian!,:residual_for_idx!),(:residual_for_trian!,:residual!))
  @eval begin
    function $f(
      b::PTArray,
      op::PThetaMethodNonlinearOperator,
      x::PTArray,
      args...)

      uF = zero(x)
      vθ = op.vθ
      @. vθ = (x-op.u0)/op.dtθ
      z = zero(eltype(b))
      fill!(b,z)
      $g(b,op.odeop,op.μ,op.tθ,(uF,vθ),op.ode_cache,args...)
    end
  end
end

function Algebra.jacobian(op::PNonlinearOperator,x::PTArray,args...)
  A = allocate_jacobian(op,x)
  jacobian!(A,op,x,args...)
end

for (f,g) in zip((:jacobian!,:jacobian_for_trian!,:jacobian_for_idx!),
                 (:jacobian!,:jacobian_for_trian!,:jacobian!))
  @eval begin
    function $f(
      A::PTArray,
      op::PThetaMethodNonlinearOperator,
      x::PTArray,
      i::Int,
      args...)

      uF = x
      vθ = op.vθ
      @. vθ = (x-op.u0)/op.dtθ
      z = zero(eltype(A))
      fillstored!(A,z)
      $g(A,op.odeop,op.μ,op.tθ,(uF,vθ),i,(1.0,1/op.dtθ)[i],op.ode_cache,args...)
    end
  end
end
