function reduced_operator(
  pop::GalerkinProjectionOperator,
  lhs::Tuple{Vararg{AffineContribution}},
  rhs::AffineContribution)

  trians_lhs = map(get_domains,lhs)
  trians_rhs = get_domains(rhs)
  new_pop = change_triangulation(pop,trians_lhs,trians_rhs)
  ReducedOperator(new_pop,lhs,rhs)
end

function reduced_operator(
  solver::RBSolver,
  op::GalerkinProjectionOperator,
  s::AbstractTransientSnapshots)

  red_lhs,red_rhs = reduced_matrix_vector_form(solver,op,s)
  red_op = reduced_operator(op,red_lhs,red_rhs)
  save(solver,red_op)
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
  lincomb_cache = allocate_mdeim_lincomb(test,r)
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
  coeff_cache = ()
  for i = 1:get_order(op)+1
    coeff_cache = (coeff_cache...,allocate_mdeim_coeff(op.lhs[i],r))
  end
  lincomb_cache = allocate_mdeim_lincomb(trial,test,r)
  return fe_A,coeff_cache,lincomb_cache
end

function Algebra.residual!(
  cache,
  op::ReducedOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  fe_b,coeff_cache,lincomb_cache = cache
  fe_sb = fe_vector!(fe_b,op,r,xhF,ode_cache)
  b_coeff = mdeim_coeff!(coeff_cache,op.rhs,fe_sb)
  mdeim_lincomb!(lincomb_cache,op.rhs,b_coeff)
  b = last(lincomb_cache)
  return b
end

function ODETools.jacobians!(
  cache,
  op::ReducedOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  γ::Tuple{Vararg{Real}},
  ode_cache)

  fe_A,coeff_cache,lincomb_cache = cache
  fe_sA = fe_matrix!(fe_A,op,r,xhF,γ,ode_cache)
  for i = 1:get_order(op)+1
    A_coeff = mdeim_coeff!(coeff_cache[i],op.lhs[i],fe_sA[i])
    mdeim_lincomb!(lincomb_cache,op.lhs[i],A_coeff)
  end
  A = last(lincomb_cache)
  return A
end

function ODETools._matrix_and_vector!(cache_mat,cache_vec,op::ReducedOperator,r,dtθ,u0,ode_cache,vθ)
  A = ODETools._matrix!(cache_mat,op,r,dtθ,u0,ode_cache,vθ)
  b = ODETools._vector!(cache_vec,op,r,dtθ,u0,ode_cache,vθ)
  return A,b
end

function ODETools._matrix!(cache,op::ReducedOperator,r,dtθ,u0,ode_cache,vθ)
  fe_A,coeff_cache,lincomb_cache = cache
  LinearAlgebra.fillstored!(fe_A,zero(eltype(fe_A)))
  A = ODETools.jacobians!(cache,op,r,(u0,vθ),(1.0,1/dtθ),ode_cache)
  return A
end

function ODETools._vector!(cache,op::ReducedOperator,r,dtθ,u0,ode_cache,vθ)
  b = residual!(cache,op,r,(u0,vθ),ode_cache)
  b .*= -1.0
  return b
end

function _union_reduced_times_mat(op::ReducedOperator)
  ilhs = ()
  for lhs in op.lhs
    for (trian,values) in lhs.dict
      ilhs = (ilhs...,get_integration_domain(values))
    end
  end
  union_indices_time(ilhs...)
end

function _union_reduced_times_vec(op::ReducedOperator)
  irhs = ()
  for (trian,values) in op.rhs.dict
    irhs = (irhs...,get_integration_domain(values))
  end
  union_indices_time(irhs...)
end

function _select_fe_quantities_at_time_locations(xhF,ode_cache,r,red_times)
  nparams = num_params(r)
  pt_indices = vec(transpose((red_times.-1)*nparams .+ collect(1:nparams)'))
  Us,Uts,fecache = ode_cache
  new_xhF = ()
  new_Us = ()
  for i = eachindex(xhF)
    spacei = Us[i].space
    dvi = ParamArray(Us[i].dirichlet_values[pt_indices])
    new_Us = (new_Us...,TrialParamFESpace(dvi,spacei))
    new_xhF = (new_xhF...,ParamArray(xhF[i][pt_indices]))
  end
  new_ode_cache = new_Us,Uts,fecache
  return new_xhF,new_ode_cache
end

function _select_snapshots_at_space_time_locations(s,a,red_times)
  ids_space = get_indices_space(a)
  ids_time = filter(!isnothing,indexin(get_indices_time(a),red_times))
  ids_param = Base.OneTo(num_params(s))
  snew = reverse_snapshots_at_indices(s,ids_space)
  select_snapshots(snew,ids_time,ids_param)
end

function _select_snapshots_at_space_time_locations(
  s::ArrayContribution,a::AffineContribution,red_times)
  b = array_contribution()
  for (trian,values) in s.dict
    b[trian] = _select_snapshots_at_space_time_locations(values,a[trian],red_times)
  end
  b
end

function fe_matrix!(
  cache,
  op::ReducedOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  γ::Tuple{Vararg{Real}},
  ode_cache)

  red_times = _union_reduced_times_mat(op)
  red_r = r[:,red_times]
  red_xhF,red_ode_cache = _select_fe_quantities_at_time_locations(xhF,ode_cache,r,red_times)
  A = fe_matrix!(cache,op.pop,red_r,red_xhF,γ,red_ode_cache)
  map(A,op.lhs) do A,lhs
    _select_snapshots_at_space_time_locations(A,lhs,red_times)
  end
end

function fe_vector!(
  cache,
  op::ReducedOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  red_times = _union_reduced_times_vec(op)
  red_xhF,red_ode_cache = _select_fe_quantities_at_time_locations(xhF,ode_cache,r,red_times)
  red_r = r[:,red_times]
  b = fe_vector!(cache,op.pop,red_r,red_xhF,red_ode_cache)
  bi = _select_snapshots_at_space_time_locations(b,op.rhs,red_times)
  return bi
end
