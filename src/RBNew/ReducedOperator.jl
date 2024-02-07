function reduced_operator(
  pop::GalerkinProjectionOperator,
  lhs_contribs::Tuple{Vararg{AffineContribution}},
  rhs_contribs::AffineContribution)

  trians_lhs = map(get_domains,lhs_contribs)
  trians_rhs = get_domains(rhs_contribs)

  new_pop = FEM.change_triangulation(pop,trians_lhs,trians_rhs)

  lhs = map(FEM.get_values,lhs_contribs)
  rhs = FEM.get_values(rhs_contribs)

  ReducedOperator(new_pop,lhs,rhs)
end

function reduced_operator(
  solver::RBSolver,
  op::GalerkinProjectionOperator,
  s::AbstractTransientSnapshots)

  red_lhs,red_rhs = reduced_matrix_vector_form(solver,op,s)
  red_op = reduced_operator(op,red_lhs,red_rhs)
  return red_op
end

struct ReducedOperator{T,L,R} <: RBOperator{T}
  pop::GalerkinProjectionOperator{T}
  lhs::L
  rhs::R
end

ReferenceFEs.get_order(op::ReducedOperator) = get_order(op.pop)
FESpaces.get_test(op::ReducedOperator) = get_test(op.pop)
FESpaces.get_trial(op::ReducedOperator) = get_trial(op.pop)
FEM.realization(op::ReducedOperator;kwargs...) = realization(op.pop;kwargs...)
FEM.get_fe_operator(op::ReducedOperator) = FEM.get_fe_operator(op.pop)

function TransientFETools.allocate_cache(
  op::ReducedOperator,
  r::TransientParamRealization)

  allocate_cache(op.pop,r)
end

function TransientFETools.update_cache!(
  ode_cache,
  op::ReducedOperator,
  r::TransientParamRealization)

  update_cache!(ode_cache,op.pop,r)
end

function Algebra.allocate_residual(
  op::ReducedOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  allocate_residual(op.pop,r,x,ode_cache)
end

function Algebra.allocate_jacobian(
  op::ReducedOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  allocate_jacobian(op.pop,r,x,ode_cache)
end

function Algebra.residual!(
  b::AbstractVector,
  op::ReducedOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  residual!(b,op.pop,r,xhF,ode_cache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::ReducedOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  i::Integer,
  γᵢ::Real,
  ode_cache)

  jacobian!(A,op.pop,r,xhF,i,γᵢ,ode_cache)
end

function ODETools.jacobians!(
  A::AbstractMatrix,
  op::ReducedOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  γ::Tuple{Vararg{Real}},
  ode_cache)

  jacobians!(A,op.pop,r,xhF,γ,ode_cache)
end

function Algebra.zero_initial_guess(op::ReducedOperator,r::TransientParamRealization)
  zero_initial_guess(op.pop,r)
end

function ODETools._allocate_matrix_and_vector(op::ReducedOperator,r,u0,ode_cache)
  ODETools._allocate_matrix_and_vector(op.pop,r,u0,ode_cache)
end

function ODETools._matrix_and_vector!(A,b,op::ReducedOperator,r,dtθ,u0,ode_cache,vθ)
  ODETools._matrix_and_vector!(A,b,op.pop,r,dtθ,u0,ode_cache,vθ)
end

function ODETools._matrix!(A,op::ReducedOperator,r,dtθ,u0,ode_cache,vθ)
  ODETools._matrix!(A,op.pop,r,dtθ,u0,ode_cache,vθ)
end

function ODETools._vector!(b,op::ReducedOperator,r,dtθ,u0,ode_cache,vθ)
  ODETools._vector!(b,op.pop,r,dtθ,u0,ode_cache,vθ)
end

function reduced_zero_initial_guess(op::ReducedOperator,r::TransientParamRealization)
  @abstractmethod
end

function allocate_reduced_matrix_and_vector(
  solver::RBThetaMethod,
  op::ReducedOperator,
  s::AbstractTransientSnapshots)

  r = get_realization(s)

  vθ = zero_initial_guess(op,r)
  ode_cache = allocate_cache(op,r)
  A,b = ODETools._allocate_matrix_and_vector(op,r,vθ,ode_cache)
  matvec_cache = A,b,ode_cache,vθ

  coeff_cache = allocate_mdeim_coefficient(op.lhs,op.rhs,r)

  lincomb_cache = allocate_mdeim_lincomb(op.lhs,op.rhs,r)

  return matvec_cache,coeff_cache,lincomb_cache
end

function reduced_matrix_and_vector!(
  solver::RBThetaMethod,
  op::ReducedOperator,
  s::AbstractTransientSnapshots,
  cache)

  matvec_cache,coeff_cache,lincomb_cache = cache
  coeff_A_cache,coeff_b_cache = coeff_cache
  lincomb_A_cache,lincomb_b_cache = lincomb_cache

  A,b = collect_matrices_vectors!(solver,op,s,matvec_cache)

  A_red = map(op.lhs) do lhs
    A_coeff = mdeim_coefficient!(coeff_A_cache,lhs,A)
    reduced_matrix!(lincomb_A_cache,lhs,A_coeff)
  end

  b_coeff = mdeim_coefficient!(coeff_b_cache,op.rhs,b)
  b_red = reduced_vector!(lincomb_b_cache,rhs,b_coeff)

  return A_red,b_red
end

function _common_reduced_times(op::ReducedOperator)
  ilhs = map(get_integration_domain,op.lhs)
  irhs = map(get_integration_domain,op.rhs)
  union_indices_time(ilhs...,irhs...)
end

function _select_snapshots_at_space_time_locations(s,a,ids_all_time)
  ids_space = get_indices_space(a)
  ids_time = get_indices_space(a)
  corresponding_ids_time = indexin(ids_all_time,ids_time)
  tensor_getindex(s,ids_space,corresponding_ids_time,:)
end

function collect_matrices_vectors!(
  solver::RBThetaMethod,
  op::ReducedOperator,
  s::AbstractTransientSnapshots,
  cache)

  ids_all_time = _common_reduced_times(op)
  sids = select_snapshots(s,:,ids_all_time)
  Aids,bids = collect_matrices_vectors!(solver,op.pop,sids,cache)
  A,b = map(zip(Aids,bids),zip(op.lhs,op.rhs)) do s,a
    _select_snapshots_at_space_time_locations(s,a,ids_all_time)
  end |> tuple_of_arrays
  return A,b
end
