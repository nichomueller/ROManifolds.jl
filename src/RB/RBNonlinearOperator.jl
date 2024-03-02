abstract type RBNonlinearOperator{T} <: NonlinearOperator end

# θ-Method specialization

struct RBThetaMethodParamOperator{T} <: RBNonlinearOperator{T}
  odeop::RBOperator{T}
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

for T in (:AbstractVector,:Contribution,:Tuple)
  @eval begin
    function Algebra.residual!(
      b::$T,
      op::RBThetaMethodParamOperator,
      x::AbstractVector)

      uF = x
      vθ = op.vθ
      residual!(b,op.odeop,op.r,(uF,vθ),op.ode_cache)
    end
  end
end
for T in (:AbstractMatrix,:(Tuple{Vararg{Contribution}}),:Tuple)
  @eval begin
    function Algebra.jacobian!(
      A::$T,
      op::RBThetaMethodParamOperator,
      x::AbstractVector)

      uF = x
      vθ = op.vθ
      jacobians!(A,op.odeop,op.r,(uF,vθ),(1.0,1/op.dtθ),op.ode_cache)
    end

    function Algebra.jacobian!(
      A::$T,
      op::RBThetaMethodParamOperator,
      x::AbstractVector,
      i::Int)

      uF = x
      vθ = op.vθ
      γ = (1.0,1/op.dtθ)
      jacobian!(A,op.odeop,op.r,(uF,vθ),i,γ[i],op.ode_cache)
    end
  end
end

function Algebra.solve!(
  x::AbstractVector,
  nls::NewtonRaphsonSolver,
  op::RBNonlinearOperator{LinearNonlinear},
  cache)

  fex = similar(op.u0)
  (cache_jac_lin,cache_res_lin),(cache_jac_nlin,cache_res_nlin) = cache

  # linear res/jac, now they are treated as cache
  lop = op.odeop.op_linear
  A_lin,b_lin = ODETools._matrix_and_vector!(
    cache_jac_lin,cache_res_lin,lop,op.r,op.dtθ,op.u0,op.ode_cache,op.vθ)
  cache_jac = A_lin,cache_jac_nlin
  cache_res = b_lin,cache_res_nlin
  cache = cache_jac,cache_res

  # initial nonlinear res/jac
  b = residual!(cache_res,op,fex)
  A = jacobian!(cache_jac,op,fex)
  dx = similar(b)
  ss = symbolic_setup(nls.ls,A)
  ns = numerical_setup(ss,A)
  _solve_rb_nr!(x,fex,A,b,dx,ns,nls,op,cache)
end

function _solve_rb_nr!(x,fex,A,b,dx,ns,nls,op,cache)
  jac_cache,res_cache = cache
  trial = get_trial(op.odeop)(op.r)
  isconv, conv0 = Algebra._check_convergence(nls,b)
  if isconv; return; end

  for nliter in 1:nls.max_nliters
    rmul!(b,-1)
    solve!(dx,ns,b)
    x .+= dx
    fex = recast(x,trial)

    b = residual!(res_cache,op,fex)
    isconv = Algebra._check_convergence(nls,b,conv0)
    if isconv; return; end
    println(maximum(abs,b))
    if nliter == nls.max_nliters
      @unreachable
    end

    A = jacobian!(jac_cache,op,fex)
    numerical_setup!(ns,A)
  end
end
