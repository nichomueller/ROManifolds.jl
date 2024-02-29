abstract type RBNonlinearOperator{T} <: NonlinearOperator end

# θ-Method specialization

struct RBThetaMethodParamOperator{C} <: NonlinearOperator
  odeop::PODMDEIMOperator{C}
  r::TransientParamRealization
  dtθ::Float64
  u0::AbstractVector
  ode_cache
  vθ::AbstractVector
end

function Algebra.allocate_residual(op::RBThetaMethodParamOperator,x::AbstractVector)
  allocate_residual(op.odeop,op.r,x,op.ode_cache)
end

function Algebra.allocate_jacobian(op::RBThetaMethodParamOperator,x::AbstractVector)
  allocate_jacobian(op.odeop,op.r,x,op.ode_cache)
end

function Algebra.zero_initial_guess(op::RBThetaMethodParamOperator)
  x0 = similar(op.u0)
  fill!(x0,zero(eltype(x0)))
  x0
end

function Algebra.residual!(
  cache::Tuple,
  op::RBThetaMethodParamOperator,
  x::AbstractVector)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  residual!(cache,op.odeop,op.r,(uF,vθ),op.ode_cache)
end

function Algebra.jacobian!(
  cache::Tuple,
  op::RBThetaMethodParamOperator,
  x::AbstractVector)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobians!(cache,op.odeop,op.r,(uF,vθ),(1.0,1/op.dtθ),op.ode_cache)
end

function Algebra.jacobian!(
  cache::Tuple,
  op::RBThetaMethodParamOperator,
  x::AbstractVector,
  i::Int)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  fecache, = cache
  for i = eachindex(fecache)
    LinearAlgebra.fillstored!(fecache[i],zero(eltype(fecache[i])))
  end
  γ = (1.0,1/op.dtθ)
  jacobian!(cache,op.odeop,op.r,(uF,vθ),i,γ[i],op.ode_cache)
end

function Algebra.solve!(
  x::AbstractVector,
  nls::NewtonRaphsonSolver,
  op::RBNonlinearOperator,
  cache::Nothing)

  b = residual(op,x)
  A = jacobian(op,x)
  dx = similar(b)
  fex = similar(op.u0)
  ss = symbolic_setup(nls.ls,A)
  ns = numerical_setup(ss,A)
  _solve_rb_nr!(x,fex,A,b,dx,ns,nls,op)
  NewtonRaphsonCache(A,b,dx,ns)
end

function _solve_rb_nr!(x,fex,A,b,dx,ns,nls,op)
  trial = get_trial(op)
  isconv, conv0 = Algebra._check_convergence(nls,b)
  if isconv; return; end

  for nliter in 1:nls.max_nliters
    rmul!(b,-1)
    solve!(dx,ns,b)
    x .+= dx
    fex .= recast(x,trial)

    residual!(b,op,fex)
    isconv = Algebra._check_convergence(nls,b,conv0)
    if isconv; return; end

    if nliter == nls.max_nliters
      @unreachable
    end

    jacobian!(A,op,fex)
    numerical_setup!(ns,A)
  end
end
