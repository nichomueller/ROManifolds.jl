
struct RBOperator{T} <: ODEOperator{T}
  feop::ODEParamOperator{T}
  trial::TrialRBSpace
  test::TestRBSpace
end

ReferenceFEs.get_order(op::RBOperator) = get_order(op.feop)
FESpaces.get_test(op::RBOperator) = get_test(op.feop)
FESpaces.get_trial(op::RBOperator) = get_trial(op.feop)
FEM.realization(op::RBOperator;kwargs...) = realization(op.feop;kwargs...)
FEM.get_fe_operator(op::RBOperator) = FEM.get_fe_operator(op.feop)

function TransientFETools.allocate_cache(
  op::RBOperator,
  r::TransientParamRealization)

  allocate_cache(op.feop,r)
end

function TransientFETools.update_cache!(
  ode_cache,
  op::RBOperator,
  r::TransientParamRealization)

  update_cache!(ode_cache,op.feop,r)
end

function Algebra.allocate_residual(
  op::RBOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  allocate_residual(op.feop,r,x,ode_cache)
end

function Algebra.allocate_jacobian(
  op::RBOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  allocate_jacobian(op.feop,r,x,ode_cache)
end

function Algebra.residual!(
  b::AbstractVector,
  op::RBOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  residual!(b,op.feop,r,xhF,ode_cache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::RBOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  i::Integer,
  γᵢ::Real,
  ode_cache)

  jacobian!(A,op.feop,r,xhF,i,γᵢ,ode_cache)
end

function ODETools.jacobians!(
  A::AbstractMatrix,
  op::RBOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  γ::Tuple{Vararg{Real}},
  ode_cache)

  jacobians!(A,op.feop,r,xhF,γ,ode_cache)
end

function Algebra.zero_initial_guess(op::RBOperator,r::TransientParamRealization)
  x0 = zero_initial_guess(op.feop)
  allocate_param_array(x0,length(r))
end

function ODETools._allocate_matrix_and_vector(op::RBOperator,r,u0,ode_cache)
  A = allocate_jacobian(op,r,u0,ode_cache)
  b = allocate_residual(op,r,u0,ode_cache)
  return A,b
end

function ODETools._matrix_and_vector!(A,b,op::RBOperator,r,dtθ,u0,ode_cache,vθ)
  sA = ODETools._matrix!(A,odeop,r,dtθ,u0,ode_cache,vθ)
  sb = ODETools._vector!(b,odeop,r,dtθ,u0,ode_cache,vθ)
  return sA,sb
end

function ODETools._matrix!(A,op::RBOperator,r,dtθ,u0,ode_cache,vθ)
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobians!(A,op.feop,r,(vθ,vθ),(1.0,1/dtθ),ode_cache)
  sA = map(A) do A
    Snapshots(A,r)
  end
  return sA
end

function ODETools._vector!(b,op::RBOperator,r,dtθ,u0,ode_cache,vθ)
  residual!(b,op.feop,r,(u0,vθ),ode_cache)
  b .*= -1.0
  sb = Snapshots(b,r)
  return sb
end

function reduced_operator(op::RBOperator,trians_vec,trians_mat,trians_mat_t)
  full_feop = FEM.get_fe_operator(op)
  reduced_feop = change_triangulation(full_feop,trians_vec,trians_mat,trians_mat_t)
  reduced_odeop = get_algebraic_operator(reduced_feop)
  RBOperator(reduced_odeop,op.trial,op.test)
end
