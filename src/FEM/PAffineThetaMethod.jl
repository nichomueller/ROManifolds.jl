function solve_step!(
  uf::PTArray,
  solver::PThetaMethod,
  op::AffinePODEOperator,
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
    l_cache = nothing
    A,b = _allocate_matrix_and_vector(op,μ,t0,u0,ode_cache)
  else
    ode_cache,vθ,A,b,l_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,μ,tθ)

  _matrix_and_vector!(A,b,op,μ,tθ,dtθ,u0,ode_cache,vθ)
  afop = PTAffineOperator(A,b)

  l_cache = solve!(uf,solver.nls,afop,l_cache)

  uf .+= u0
  if 0.0 < solver.θ < 1.0
    @. uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,A,b,l_cache)
  tf = t0+dt
  return (uf,tf,cache)
end

for (f,g) in zip((:residual!,:residual_for_trian!,:residual_for_idx!),
                 (:residual!,:residual_for_trian!,:residual!))
  @eval begin
    function $f(
      b::PTArray,
      op::PTThetaAffineMethodOperator,
      ::PTArray,
      args...)

      vθ = op.vθ
      z = zero(eltype(b))
      fill!(b,z)
      $g(b,op.odeop,op.μ,op.tθ,(vθ,vθ),op.ode_cache,args...)
    end
  end
end

function Algebra.jacobian!(
  A::PTArray,
  op::PTThetaAffineMethodOperator,
  ::PTArray)

  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobians!(A,op.odeop,op.μ,op.tθ,(vθ,vθ),(1.0,1/op.dtθ),op.ode_cache)
end

for (f,g) in zip((:jacobian!,:jacobian_for_trian!,:jacobian_for_idx!),
                 (:jacobian!,:jacobian_for_trian!,:jacobian!))
  @eval begin
    function $f(
      A::PTArray,
      op::PTThetaAffineMethodOperator,
      ::PTArray,
      i::Int,
      args...)

      vθ = op.vθ
      z = zero(eltype(A))
      fillstored!(A,z)
      $g(A,op.odeop,op.μ,op.tθ,(vθ,vθ),i,(1.0,1/op.dtθ)[i],op.ode_cache,args...)
    end
  end
end

# SHORTCUTS
function _allocate_matrix_and_vector(odeop,μ,t0,u0,ode_cache)
  b = allocate_residual(odeop,μ,t0,u0,ode_cache)
  A = allocate_jacobian(odeop,μ,t0,u0,ode_cache)
  return A,b
end

function _matrix_and_vector!(
  A::PTArray,
  b::PTArray,
  op::AffinePODEOperator,
  μ,
  tθ,
  dtθ,
  u0,
  ode_cache,
  vθ)

  _matrix!(A,op,μ,tθ,dtθ,u0,ode_cache,vθ)
  _vector!(b,op,μ,tθ,dtθ,u0,ode_cache,vθ)
end

function _matrix!(
  A::PTArray,
  op::AffinePODEOperator,
  μ,
  tθ,
  dtθ,
  u0,
  ode_cache,
  vθ)

  z = zero(eltype(A))
  fillstored!(A,z)
  jacobians!(A,op,μ,tθ,(vθ,vθ),(1.0,1/dtθ),ode_cache)
end

function _vector!(
  b::PTArray,
  op::AffinePODEOperator,
  μ,
  tθ,
  dtθ,
  u0,
  ode_cache,
  vθ)

  z = zero(eltype(b))
  fill!(b,z)
  residual!(b,op,μ,tθ,(u0,vθ),ode_cache)
  b.array .*= -1.0
  b
end
