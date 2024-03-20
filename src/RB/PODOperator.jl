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

abstract type RBOperator{T<:ODEParamOperatorType} <: ODEParamOperatorWithTrian{T} end
const LinearRBOperator = RBOperator{LinearODE}

struct PODOperator{T} <: RBOperator{T}
  odeop::ODEParamOperatorWithTrian{T}
  trial::RBSpace
  test::RBSpace
end

FESpaces.get_trial(op::PODOperator) = op.trial
FESpaces.get_test(op::PODOperator) = op.test
FEM.realization(op::PODOperator;kwargs...) = realization(op.odeop;kwargs...)
FEM.get_fe_operator(op::PODOperator) = FEM.get_fe_operator(op.odeop)
get_fe_trial(op::PODOperator) = get_trial(op.odeop)
get_fe_test(op::PODOperator) = get_test(op.odeop)
ODEs.get_num_forms(op::PODOperator) = get_num_forms(op.odeop)
ODEs.get_forms(op::PODOperator) = get_forms(op.odeop)
ODEs.is_form_constant(op::PODOperator,k::Integer) = is_form_constant(op.odeop,k)

function FEM.get_linear_operator(op::PODOperator{LinearNonlinearParamODE})
  PODOperator(get_linear_operator(op.odeop),op.trial,op.test)
end

function FEM.get_nonlinear_operator(op::PODOperator{LinearNonlinearParamODE})
  PODOperator(get_nonlinear_operator(op.odeop),op.trial,op.test)
end

function FEM.set_triangulation(
  op::PODOperator,
  trians_rhs,
  trians_lhs)

  PODOperator(set_triangulation(op.odeop,trians_rhs,trians_lhs),op.trial,op.test)
end

function FEM.change_triangulation(
  op::PODOperator,
  trians_rhs,
  trians_lhs)

  PODOperator(change_triangulation(op.odeop,trians_rhs,trians_lhs),op.trial,op.test)
end

function ODEs.allocate_odeopcache(
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}})

  allocate_odeopcache(op.odeop,r,us)
end

function ODEs.update_odeopcache!(
  ode_cache,
  op::PODOperator,
  r::TransientParamRealization)

  update_odeopcache!(ode_cache,op.odeop,r)
end

function Algebra.allocate_residual(
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  allocate_residual(op.odeop,r,us,odeopcache)
end

function Algebra.allocate_jacobian(
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  allocate_jacobian(op.odeop,r,us,odeopcache)
end

function allocate_jacobian_and_residual(op::PODOperator,r,us,odeopcache)
  A = allocate_jacobian(op,r,us,odeopcache)
  b = allocate_residual(op,r,us,odeopcache)
  return A,b
end

function Algebra.residual!(
  b::Contribution,
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  kwargs...)

  residual!(b,op.odeop,r,xhF,ode_cache;kwargs...)
  return Snapshots(b,r)
end

function Algebra.jacobian!(
  A::Tuple{Vararg{Contribution}},
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  jacobian!(A,op.odeop,r,us,ws,odeopcache)
  return Snapshots.(A,r)
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

# function jacobian_and_residual(
#   solver::ThetaMethodRBSolver,
#   op::RBOperator{Nonlinear},
#   s::S) where S

#   function insert_basis!(x,Φ,i)
#     x .= Φ[:,mod(i-1,size(Φ,2))+1]
#   end
#   function insert_basis!(x,Φ::ArrayBlock,i)
#     for n = eachindex(Φ)
#       insert_basis!(x[Block(n)],Φ[n],i)
#     end
#   end

#   fesolver = get_fe_solver(solver)
#   dt = fesolver.dt
#   θ = fesolver.θ
#   θ == 0.0 ? dtθ = dt : dtθ = dt*θ

#   x = get_values(s)
#   r = get_realization(s)
#   FEM.shift_time!(r,dt*(θ-1))

#   red_trial = get_trial(op)
#   basis_space = get_basis_space(red_trial)

#   @inbounds for i = eachindex(x)
#     insert_basis!(x,basis_space,i)
#   end

#   y = similar(x)
#   y .= 0.0
#   ode_cache = allocate_cache(op,r)
#   A,b = allocate_jacobian_and_residual(op,r,x,ode_cache)

#   ode_cache = update_cache!(ode_cache,op,r)
#   jacobian_and_residual!(A,b,op,r,dtθ,x,ode_cache,y)
# end

function ODEs.jacobians!(A,op::LinearRBOperator,r,dtθ,u0,ode_cache,vθ)
  jacobians!(A,op,r,(vθ,vθ),(1.0,1/dtθ),ode_cache)
end

function ODEs.jacobians!(A,op::RBOperator,r,dtθ,u0,ode_cache,vθ)
  jacobians!(A,op,r,(u0,vθ),(1.0,1/dtθ),ode_cache)
end

function Algebra.residual!(b,op::LinearRBOperator,r,dtθ,u0,ode_cache,vθ)
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
