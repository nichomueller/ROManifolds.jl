function reduced_operator(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  s::AbstractTransientSnapshots)

  info = get_info(solver)
  red_trial,red_test = reduced_fe_space(info,feop,s)
  odeop = get_algebraic_operator(feop)
  reduced_operator(solver,odeop,red_trial,red_test,s)
end

function reduced_operator(
  solver::RBSolver,
  odeop::ODEParamOperator,
  trial::TrialRBSpace,
  test::TestRBSpace,
  s::AbstractTransientSnapshots)

  pop = GalerkinProjectionOperator(odeop,trial,test)
  reduced_operator(solver,pop,s)
end

abstract type RBOperator{T} <: ODEParamOperator{T} end
const AffineRBOperator = RBOperator{Affine}

function get_fe_trial(op::RBOperator)
  @abstractmethod
end

function get_fe_test(op::RBOperator)
  @abstractmethod
end

function allocate_fe_residual(op::RBOperator)
  @abstractmethod
end

function allocate_fe_jacobian(op::RBOperator)
  @abstractmethod
end

function fe_residual!(op::RBOperator)
  @abstractmethod
end

function fe_jacobian!(op::RBOperator)
  @abstractmethod
end

function fe_jacobians!(op::RBOperator)
  @abstractmethod
end

function Algebra.solve(
  x::AbstractVector,
  solver::RBThetaMethod,
  op::RBOperator,
  r::TransientParamRealization)

  fesolver = get_fe_solver(solver)
  dt = fesolver.dt
  θ = fesolver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ

  ode_cache = allocate_cache(op,r)
  y = similar(x)
  y .= 0.0
  nl_cache = nothing

  ode_cache = update_cache!(ode_cache,op,r)

  nlop = ThetaMethodParamOperator(op,r,dtθ,x,ode_cache,y)

  solve!(x,fesolver.nls,nlop,nl_cache)

  return x
end

function Algebra.solve(
  x::AbstractVector,
  solver::RBThetaMethod,
  op::AffineRBOperator,
  r::TransientParamRealization)

  fesolver = get_fe_solver(solver)
  dt = fesolver.dt
  θ = fesolver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ

  ode_cache = allocate_cache(op,r)
  y = similar(x)
  y .= 0.0
  mat_cache,vec_cache = ODETools._allocate_matrix_and_vector(op,r,y,ode_cache)

  ode_cache = update_cache!(ode_cache,op,r)

  A,b = ODETools._matrix_and_vector!(mat_cache,vec_cache,op,r,dtθ,y,ode_cache,y)
  afop = AffineOperator(A,b)
  solve!(x,fesolver.nls,afop)

  return x
end

struct GalerkinProjectionOperator{T} <: RBOperator{T}
  feop::ODEParamOperator{T}
  trial::TrialRBSpace
  test::TestRBSpace
end

ReferenceFEs.get_order(op::GalerkinProjectionOperator) = get_order(op.feop)
FESpaces.get_trial(op::GalerkinProjectionOperator) = op.trial
FESpaces.get_test(op::GalerkinProjectionOperator) = op.test
FEM.realization(op::GalerkinProjectionOperator;kwargs...) = realization(op.feop;kwargs...)
FEM.get_fe_operator(op::GalerkinProjectionOperator) = FEM.get_fe_operator(op.feop)
get_fe_trial(op::GalerkinProjectionOperator) = get_trial(op.feop)
get_fe_test(op::GalerkinProjectionOperator) = get_test(op.feop)

function FEM.change_triangulation(
  op::GalerkinProjectionOperator,trians_lhs,trians_rhs)

  feop = FEM.get_fe_operator(op)
  new_feop = FEM.change_triangulation(feop,trians_lhs,trians_rhs)
  new_odeop = get_algebraic_operator(new_feop)
  GalerkinProjectionOperator(new_odeop,op.trial,op.test)
end

function TransientFETools.allocate_cache(
  op::GalerkinProjectionOperator,
  r::TransientParamRealization)

  allocate_cache(op.feop,r)
end

function TransientFETools.update_cache!(
  ode_cache,
  op::GalerkinProjectionOperator,
  r::TransientParamRealization)

  update_cache!(ode_cache,op.feop,r)
end

function allocate_fe_residual(
  op::GalerkinProjectionOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  allocate_residual(op.feop,r,x,ode_cache)
end

function allocate_fe_jacobian(
  op::GalerkinProjectionOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  allocate_jacobian(op.feop,r,x,ode_cache)
end

function fe_residual!(
  b::AbstractVector,
  op::GalerkinProjectionOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  residual!(b,op.feop,r,xhF,ode_cache)
end

function fe_jacobian!(
  A::AbstractMatrix,
  op::GalerkinProjectionOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  i::Integer,
  γᵢ::Real,
  ode_cache)

  jacobian!(A,op.feop,r,xhF,i,γᵢ,ode_cache)
end

function fe_jacobians!(
  A::AbstractMatrix,
  op::GalerkinProjectionOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  γ::Tuple{Vararg{Real}},
  ode_cache)

  ODETools.jacobians!(A,op.feop,r,xhF,γ,ode_cache)
end

function allocate_fe_matrix_and_vector(op::GalerkinProjectionOperator,r,u0,ode_cache)
  A = allocate_jacobian(op,r,u0,ode_cache)
  b = allocate_residual(op,r,u0,ode_cache)
  return A,b
end

function fe_matrix_and_vector!(A,b,op::GalerkinProjectionOperator,r,dtθ,u0,ode_cache,vθ)
  fe_matrix!(A,op,r,dtθ,u0,ode_cache,vθ)
  fe_vector!(b,op,r,dtθ,u0,ode_cache,vθ)
  return A,b
end

function fe_matrix!(A,op::GalerkinProjectionOperator,r,dtθ,u0,ode_cache,vθ)
  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  ODETools.jacobians!(A,op.feop,r,(vθ,vθ),(1.0,1/dtθ),ode_cache)
  return A
end

function fe_vector!(b,op::GalerkinProjectionOperator,r,dtθ,u0,ode_cache,vθ)
  residual!(b,op.feop,r,(u0,vθ),ode_cache)
  b .*= -1.0
  return b
end
