function reduced_operator(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  s::S) where S

  red_trial,red_test = reduced_fe_space(solver,feop,s)
  odeop = get_algebraic_operator(feop)
  reduced_operator(solver,odeop,red_trial,red_test,s)
end

function reduced_operator(
  solver::RBSolver,
  odeop::ODEParamOperator,
  trial::RBSpace,
  test::RBSpace,
  s::S) where S

  pop = PODOperator(odeop,trial,test)
  reduced_operator(solver,pop,s)
end

abstract type RBOperator{T<:OperatorType} <: ODEOperator{T} end
const AffineRBOperator = RBOperator{Affine}

struct PODOperator{T,A,B} <: RBOperator{T}
  feop::ODEParamOperator{T}
  trial::A
  test::B
end

ReferenceFEs.get_order(op::PODOperator) = get_order(op.feop)
FESpaces.get_trial(op::PODOperator) = op.trial
FESpaces.get_test(op::PODOperator) = op.test
FEM.realization(op::PODOperator;kwargs...) = realization(op.feop;kwargs...)
FEM.get_fe_operator(op::PODOperator) = FEM.get_fe_operator(op.feop)
FEM.get_linear_operator(op::PODOperator{LinearNonlinear}) = PODOperator(get_linear_operator(op.feop),op.trial,op.test)
FEM.get_nonlinear_operator(op::PODOperator{LinearNonlinear}) = PODOperator(get_nonlinear_operator(op.feop),op.trial,op.test)
get_fe_trial(op::PODOperator) = get_trial(op.feop)
get_fe_test(op::PODOperator) = get_test(op.feop)

function FEM.change_triangulation(
  op::PODOperator,trians_rhs,trians_lhs)

  feop = FEM.get_fe_operator(op)
  new_feop = FEM.change_triangulation(feop,trians_rhs,trians_lhs)
  new_odeop = get_algebraic_operator(new_feop)
  PODOperator(new_odeop,op.trial,op.test)
end

function TransientFETools.allocate_cache(
  op::PODOperator,
  r::TransientParamRealization)

  allocate_cache(op.feop,r)
end

function TransientFETools.update_cache!(
  ode_cache,
  op::PODOperator,
  r::TransientParamRealization)

  update_cache!(ode_cache,op.feop,r)
end

function Algebra.allocate_residual(
  op::PODOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  allocate_residual(op.feop,r,x,ode_cache)
end

function Algebra.allocate_jacobian(
  op::PODOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  allocate_jacobian(op.feop,r,x,ode_cache)
end

function allocate_jacobian_and_residual(op::PODOperator,r,u0,ode_cache)
  A = allocate_jacobian(op,r,u0,ode_cache)
  b = allocate_residual(op,r,u0,ode_cache)
  return A,b
end

function Algebra.residual!(
  b,
  op::PODOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  residual!(b,op.feop,r,xhF,ode_cache)
  Snapshots(b,r)
end

function Algebra.jacobian!(
  A,
  op::PODOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  i::Integer,
  γᵢ::Real,
  ode_cache)

  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  jacobian!(A,op.feop,r,xhF,i,γᵢ,ode_cache)
  Snapshots(A,r)
end

function ODETools.jacobians!(
  A,
  op::PODOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  γ::Tuple{Vararg{Real}},
  ode_cache)

  map(eachindex(A)) do i
    jacobian!(A[i],op,r,xhF,i,γ[i],ode_cache)
  end |> Tuple
end

# θ-Method specialization

# FE jacobians and residuals computed from a RBOperator

function jacobian_and_residual(
  solver::ThetaMethodRBSolver,
  op::RBOperator,
  s::S) where S

  fesolver = get_fe_solver(solver)
  dt = fesolver.dt
  θ = fesolver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ

  x = get_values(s)
  r = get_realization(s)
  FEM.shift_time!(r,dt*(θ-1))

  y = similar(x)
  y .= 0.0
  ode_cache = allocate_cache(op,r)
  A,b = allocate_jacobian_and_residual(op,r,x,ode_cache)

  ode_cache = update_cache!(ode_cache,op,r)
  jacobian_and_residual!(A,b,op,r,dtθ,x,ode_cache,y)
end

function ODETools.jacobians!(A,op::AffineRBOperator,r,dtθ,u0,ode_cache,vθ)
  jacobians!(A,op,r,(vθ,vθ),(1.0,1/dtθ),ode_cache)
end

function ODETools.jacobians!(A,op::RBOperator,r,dtθ,u0,ode_cache,vθ)
  jacobians!(A,op,r,(u0,vθ),(1.0,1/dtθ),ode_cache)
end

function Algebra.residual!(b,op::AffineRBOperator,r,dtθ,u0,ode_cache,vθ)
  residual!(b,op,r,(vθ,vθ),ode_cache)
end

function Algebra.residual!(b,op::RBOperator,r,dtθ,u0,ode_cache,vθ)
  residual!(b,op,r,(u0,vθ),ode_cache)
end

function jacobian_and_residual!(A,b,op::RBOperator,r,dtθ,u0,ode_cache,vθ)
  sA = jacobians!(A,op,r,dtθ,u0,ode_cache,vθ)
  sb = residual!(b,op,r,dtθ,u0,ode_cache,vθ)
  return sA,sb
end

# for testing/visualization purposes

function projection_error(op::RBOperator,s::AbstractArray)
  feop = FEM.get_fe_operator(op)
  trial = get_trial(op)
  norm_matrix = assemble_norm_matrix(feop)
  projection_error(trial,s,norm_matrix)
end
