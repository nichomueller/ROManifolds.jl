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

  nlop = PTThetaMethodOperator(op,μ,tθ,dtθ,u0,ode_cache,vθ)

  nl_cache = solve!(uf,solver.nls,nlop,nl_cache)

  if 0.0 < solver.θ < 1.0
    @. uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,nl_cache)
  tf = t0+dt
  return (uf,tf,cache)
end

function residual!(
  b::PTArray,
  op::PTThetaMethodOperator,
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
      op::PTThetaMethodOperator,
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

function Algebra.jacobian(op::PTAlgebraicOperator,x::PTArray,args...)
  A = allocate_jacobian(op,x)
  jacobian!(A,op,x,args...)
end

for (f,g) in zip((:jacobian!,:jacobian_for_trian!,:jacobian_for_idx!),
                 (:jacobian!,:jacobian_for_trian!,:jacobian!))
  @eval begin
    function $f(
      A::PTArray,
      op::PTThetaMethodOperator,
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
