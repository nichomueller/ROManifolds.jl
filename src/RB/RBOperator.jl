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

  pop = RBOperator(odeop,trial,test)
  reduced_operator(solver,pop,s)
end

struct RBOperator{T,A,B}
  feop::ODEParamOperator{T}
  trial::A
  test::B
end

const AffineRBOperator = RBOperator{Affine}

ReferenceFEs.get_order(op::RBOperator) = get_order(op.feop)
FESpaces.get_trial(op::RBOperator) = op.trial
FESpaces.get_test(op::RBOperator) = op.test
FEM.realization(op::RBOperator;kwargs...) = realization(op.feop;kwargs...)
FEM.get_fe_operator(op::RBOperator) = FEM.get_fe_operator(op.feop)
FEM.get_linear_operator(op::RBOperator) = RBOperator(get_linear_operator(op.feop),op.trial,op.test)
FEM.get_nonlinear_operator(op::RBOperator) = RBOperator(get_nonlinear_operator(op.feop),op.trial,op.test)
get_fe_trial(op::RBOperator) = get_trial(op.feop)
get_fe_test(op::RBOperator) = get_test(op.feop)

function FEM.change_triangulation(
  op::RBOperator,trians_lhs,trians_rhs)

  feop = FEM.get_fe_operator(op)
  new_feop = FEM.change_triangulation(feop,trians_lhs,trians_rhs)
  new_odeop = get_algebraic_operator(new_feop)
  RBOperator(new_odeop,op.trial,op.test)
end

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

# convention: when computing full order matrices / vectors, we view them as snapshots

function fe_matrix_and_vector(
  solver::ThetaMethodRBSolver,
  op::RBOperator,
  s::S) where S

  fesolver = get_fe_solver(solver)
  dt = fesolver.dt
  θ = fesolver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ

  x = get_values(s)
  r = copy(get_realization(s))
  FEM.shift_time!(r,dt*(θ-1))

  y = similar(x)
  y .= 0.0
  ode_cache = allocate_cache(op,r)
  A,b = allocate_fe_matrix_and_vector(op,r,x,ode_cache)

  ode_cache = update_cache!(ode_cache,op,r)
  fe_matrix_and_vector!(A,b,op,r,dtθ,x,ode_cache,y)
end

function allocate_fe_vector(
  op::RBOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  allocate_residual(op.feop,r,x,ode_cache)
end

function allocate_fe_matrix(
  op::RBOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  allocate_jacobian(op.feop,r,x,ode_cache)
end

function allocate_fe_matrix_and_vector(op::RBOperator,r,u0,ode_cache)
  A = allocate_fe_matrix(op,r,u0,ode_cache)
  b = allocate_fe_vector(op,r,u0,ode_cache)
  return A,b
end

function fe_vector!(
  b,
  op::RBOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  residual!(b,op.feop,r,xhF,ode_cache)
  Snapshots(b,r)
end

function fe_matrix!(
  A,
  op::RBOperator,
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

function fe_matrix!(
  A,
  op::RBOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  γ::Tuple{Vararg{Real}},
  ode_cache)

  map(eachindex(A)) do i
    fe_matrix!(A[i],op,r,xhF,i,γ[i],ode_cache)
  end |> Tuple
end

function fe_matrix!(A,op::RBOperator,r,dtθ,u0,ode_cache,vθ)
  fe_matrix!(A,op,r,(vθ,vθ),(1.0,1/dtθ),ode_cache)
end

function fe_vector!(b,op::RBOperator,r,dtθ,u0,ode_cache,vθ)
  fe_vector!(b,op,r,(u0,vθ),ode_cache)
end

function fe_matrix_and_vector!(A,b,op::AffineRBOperator,r,dtθ,u0,ode_cache,vθ)
  sA = fe_matrix!(A,op,r,dtθ,vθ,ode_cache,vθ)
  sb = fe_vector!(b,op,r,dtθ,vθ,ode_cache,vθ)
  return sA,sb
end

function fe_matrix_and_vector!(A,b,op::RBOperator,r,dtθ,u0,ode_cache,vθ)
  sA = fe_matrix!(A,op,r,dtθ,u0,ode_cache,vθ)
  sb = fe_vector!(b,op,r,dtθ,u0,ode_cache,vθ)
  return sA,sb
end
