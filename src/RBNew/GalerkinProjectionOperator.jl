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

function get_reduced_trial(op::RBOperator)
  @abstractmethod
end

function get_reduced_test(op::RBOperator)
  @abstractmethod
end

struct GalerkinProjectionOperator{T} <: RBOperator{T}
  feop::ODEParamOperator{T}
  trial::TrialRBSpace
  test::TestRBSpace
end

ReferenceFEs.get_order(op::GalerkinProjectionOperator) = get_order(op.feop)
FESpaces.get_test(op::GalerkinProjectionOperator) = get_test(op.feop)
FESpaces.get_trial(op::GalerkinProjectionOperator) = get_trial(op.feop)
FEM.realization(op::GalerkinProjectionOperator;kwargs...) = realization(op.feop;kwargs...)
FEM.get_fe_operator(op::GalerkinProjectionOperator) = FEM.get_fe_operator(op.feop)
get_reduced_trial(op::GalerkinProjectionOperator) = op.trial
get_reduced_test(op::GalerkinProjectionOperator) = op.test

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

function Algebra.allocate_residual(
  op::GalerkinProjectionOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  allocate_residual(op.feop,r,x,ode_cache)
end

function Algebra.allocate_jacobian(
  op::GalerkinProjectionOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  allocate_jacobian(op.feop,r,x,ode_cache)
end

function Algebra.residual!(
  b::AbstractVector,
  op::GalerkinProjectionOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  residual!(b,op.feop,r,xhF,ode_cache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::GalerkinProjectionOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  i::Integer,
  γᵢ::Real,
  ode_cache)

  jacobian!(A,op.feop,r,xhF,i,γᵢ,ode_cache)
end

function ODETools.jacobians!(
  A::AbstractMatrix,
  op::GalerkinProjectionOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  γ::Tuple{Vararg{Real}},
  ode_cache)

  jacobians!(A,op.feop,r,xhF,γ,ode_cache)
end

function Algebra.zero_initial_guess(op::GalerkinProjectionOperator,r::TransientParamRealization)
  x0 = zero_initial_guess(op.feop)
  allocate_param_array(x0,length(r))
end

function ODETools._allocate_matrix_and_vector(op::GalerkinProjectionOperator,r,u0,ode_cache)
  A = allocate_jacobian(op,r,u0,ode_cache)
  b = allocate_residual(op,r,u0,ode_cache)
  return A,b
end

function ODETools._matrix_and_vector!(A,b,op::GalerkinProjectionOperator,r,dtθ,u0,ode_cache,vθ)
  sA = ODETools._matrix!(A,op,r,dtθ,u0,ode_cache,vθ)
  sb = ODETools._vector!(b,op,r,dtθ,u0,ode_cache,vθ)
  return sA,sb
end

function ODETools._matrix!(A,op::GalerkinProjectionOperator,r,dtθ,u0,ode_cache,vθ)
  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  ODETools.jacobians!(A,op.feop,r,(vθ,vθ),(1.0,1/dtθ),ode_cache)
  sA = map(A) do A
    Snapshots(A,r)
  end
  return sA
end

function ODETools._vector!(b,op::GalerkinProjectionOperator,r,dtθ,u0,ode_cache,vθ)
  residual!(b,op.feop,r,(u0,vθ),ode_cache)
  b .*= -1.0
  sb = Snapshots(b,r)
  return sb
end

function collect_matrices_vectors!(
  solver::RBThetaMethod,
  op::GalerkinProjectionOperator{Affine},
  s::AbstractTransientSnapshots,
  cache)

  fesolver = get_fe_solver(solver)
  dt = fesolver.dt
  θ = fesolver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ

  sθ = FEM.shift_time!(s,dt,θ)
  uθ = sθ.values
  r = get_realization(sθ)

  if isnothing(cache)
    ode_cache = allocate_cache(op,r)
    vθ = similar(uθ)
    vθ .= 0.0
    A,b = ODETools._allocate_matrix_and_vector(op,r,vθ,ode_cache)
  else
    A,b,ode_cache,vθ = cache
  end

  ode_cache = update_cache!(ode_cache,op.feop,r)

  sA,sb = ODETools._matrix_and_vector!(A,b,op,r,dtθ,vθ,ode_cache,vθ)
  cache = A,b,ode_cache,vθ

  return sA,sb,cache
end

function collect_matrices_vectors!(
  solver::RBThetaMethod,
  op::GalerkinProjectionOperator,
  s::AbstractTransientSnapshots,
  cache)

  fesolver = get_fe_solver(solver)
  dt = fesolver.dt
  θ = fesolver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ

  sθ = FEM.shift_time!(s,dt,θ)
  uθ = sθ.values
  r = get_realization(sθ)

  if isnothing(cache)
    vθ = similar(uθ)
    vθ .= 0.0
    A,b = ODETools._allocate_matrix_and_vector(op,r,vθ,ode_cache)
    ode_cache = allocate_cache(op,r)
  else
    A,b,ode_cache,vθ = cache
  end

  ode_cache = update_cache!(ode_cache,op.feop,r)

  sA,sb = ODETools._matrix_and_vector!(A,b,op,r,dtθ,uθ,ode_cache,vθ)
  cache = A,b,ode_cache,vθ

  return sA,sb,cache
end

function FEM.change_triangulation(
  op::GalerkinProjectionOperator,trians_lhs,trians_rhs)

  feop = FEM.get_fe_operator(op)
  new_feop = FEM.change_triangulation(feop,trians_lhs,trians_rhs)
  new_odeop = get_algebraic_operator(new_feop)
  GalerkinProjectionOperator(new_odeop,op.trial,op.test)
end
