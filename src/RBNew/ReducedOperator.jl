function reduced_operator(
  pop::GalerkinProjectionOperator,
  lhs_contribs::Tuple{Vararg{AffineContribution}},
  rhs_contribs::AffineContribution)

  trians_lhs = map(get_domains,lhs_contribs)
  trians_rhs = get_domains(rhs_contribs)

  new_pop = change_triangulation(pop,trians_lhs,trians_rhs)

  lhs = map(get_values,lhs_contribs)
  rhs = get_values(rhs_contribs)

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
FESpaces.get_trial(op::ReducedOperator) = get_trial(op.pop)
FESpaces.get_test(op::ReducedOperator) = get_test(op.pop)
FEM.realization(op::ReducedOperator;kwargs...) = realization(op.pop;kwargs...)
FEM.get_fe_operator(op::ReducedOperator) = FEM.get_fe_operator(op.pop)
get_fe_trial(op::ReducedOperator) = get_fe_trial(op.pop)
get_fe_test(op::ReducedOperator) = get_fe_test(op.pop)

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

# cache for residual/jacobians includes:
# 1) cache to assemble residuals/jacobians on reduced integration domain
# 2) cache to compute the mdeim coefficient
# 3) cache to perform the kronecker product between basis and coefficient

function Algebra.allocate_residual(
  op::ReducedOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  test = get_test(op)
  fe_b = allocate_fe_vector(op.pop,r,x,ode_cache)
  coeff_cache = allocate_mdeim_coeff(op.rhs,r)
  lincomb_cache = allocate_mdeim_lincomb(test,op.rhs,r)
  return fe_b,coeff_cache,lincomb_cache
end

function Algebra.allocate_jacobian(
  op::ReducedOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  trial = get_trial(op)
  test = get_test(op)
  fe_A = allocate_fe_matrix(op.pop,r,x,ode_cache)
  coeff_cache = allocate_mdeim_coeff(op.lhs,r)
  lincomb_cache = allocate_mdeim_lincomb(trial,test,op.lhs,r)
  return fe_A,coeff_cache,lincomb_cache
end

function Algebra.residual!(
  cache,
  op::ReducedOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  fe_b,coeff_cache,lincomb_cache = cache
  fe_vectors!(fe_b,op,r,xhF,ode_cache)
  b_coeff = mdeim_coeff!(coeff_cache,op.rhs,fe_b)
  b = mdeim_lincomb!(lincomb_cache,op.rhs,b_coeff)
  return b
end

function Algebra.jacobian!(
  cache,
  op::ReducedOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  fe_A,coeff_cache,lincomb_cache = cache
  fe_matrices!(fe_A,op,r,xhF,ode_cache)
  A_coeff = mdeim_coeff!(coeff_cache,op.lhs,fe_A)
  A = mdeim_lincomb!(lincomb_cache,op.lhs,A_coeff)
  return A
end

function _union_reduced_times(op::ReducedOperator)
  ilhs = ()
  for lhs in op.lhs
    ilhs = (ilhs...,map(get_integration_domain,lhs)...)
  end
  irhs = map(get_integration_domain,op.rhs)
  union_indices_time(ilhs...,irhs...)
end

function _select_snapshots_at_space_time_locations(s,a,ids_all_time)
  ids_space = get_indices_space(a)
  ids_time = get_indices_time(a)
  corresponding_ids_time = filter(!isnothing,indexin(ids_all_time,ids_time))
  cols = col_index(s,corresponding_ids_time,1:num_params(s))
  view(s,ids_space,cols)
end

function _select_snapshots_at_space_time_locations(s::AbstractVector,a::AbstractVector,ids_all_time)
  map((s,a)->_select_snapshots_at_space_time_locations(s,a,ids_all_time),s,a)
end

function fe_matrices!(
  cache,
  op::ReducedOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  ids_all_time = _union_reduced_times(op)
  xhFi = ()
  for i = eachindex(xhF)
    si = Snapshots(xhF[i],r)
    xhFi = (xhFi...,select_snapshots(si,:,ids_all_time))
  end
  A = fe_matrices!(cache,op.pop,xhFi,r,ode_cache)
  Ai = _select_snapshots_at_space_time_locations(A,op.lhs,ids_all_time)
  return Ai
end

function fe_vectors!(
  cache::RBThetaMethod,
  op::ReducedOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  ids_all_time = _union_reduced_times(op)
  xhFi = ()
  for i = eachindex(xhF)
    si = Snapshots(xhF[i],r)
    xhFi = (xhFi...,select_snapshots(si,:,ids_all_time))
  end
  b = fe_vectors!(cache,op.pop,xhFi,r,ode_cache)
  bi = _select_snapshots_at_space_time_locations(b,op.lhs,ids_all_time)
  return bi
end
