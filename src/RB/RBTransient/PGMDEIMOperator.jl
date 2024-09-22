function RBSteady.reduced_operator(
  solver::RBSolver,
  op::TransientPGOperator,
  s)

  red_lhs,red_rhs = reduced_weak_form(solver,op,s)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = map(get_domains,red_lhs)
  new_op = change_triangulation(op,trians_rhs,trians_lhs)
  TransientPGMDEIMOperator(new_op,red_lhs,red_rhs)
end

function RBSteady.reduced_operator(
  solver::RBSolver,
  op::TransientPGOperator{LinearNonlinearParamODE},
  s)

  red_op_lin = reduced_operator(solver,get_linear_operator(op),s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(op),s)
  LinearNonlinearTransientPGMDEIMOperator(red_op_lin,red_op_nlin)
end

"""
    struct TransientPGMDEIMOperator{T} <: TransientRBOperator{T}

Represents the composition operator

[`TransientPGOperator`][@ref] ∘ [`ReducedAlgebraicOperator`][@ref]

This allows the projection of MDEIM-approximated residuals/jacobians `rhs` and
`lhs` onto the FESubspace encoded in `op`. In particular, the residual of a
TransientPGMDEIMOperator is computed as follows:

1) numerical integration is performed to compute the residual on its reduced
  integration domain
2) the MDEIM online phase takes place for the assembly of the projected, MDEIM-
  approximated residual

The same reasoning holds for all the jacobians, which are as many as the order
of the temporal derivative

"""
struct TransientPGMDEIMOperator{T} <: TransientRBOperator{T}
  op::TransientPGOperator{T}
  lhs::TupOfAffineContribution
  rhs::AffineContribution
end

FESpaces.get_trial(op::TransientPGMDEIMOperator) = get_trial(op.op)
FESpaces.get_test(op::TransientPGMDEIMOperator) = get_test(op.op)
ParamDataStructures.realization(op::TransientPGMDEIMOperator;kwargs...) = realization(op.op;kwargs...)
ParamSteady.get_fe_operator(op::TransientPGMDEIMOperator) = ParamSteady.get_fe_operator(op.op)
ParamSteady.get_vector_index_map(op::TransientPGMDEIMOperator) = get_vector_index_map(op.op)
ParamSteady.get_matrix_index_map(op::TransientPGMDEIMOperator) = get_matrix_index_map(op.op)
RBSteady.get_fe_trial(op::TransientPGMDEIMOperator) = get_fe_trial(op.op)
RBSteady.get_fe_test(op::TransientPGMDEIMOperator) = get_fe_test(op.op)

function ODEs.allocate_odecache(
  fesolver::ThetaMethod,
  op::TransientPGMDEIMOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_odecache(fesolver,op.op,r,us)
end

function ODEs.allocate_odeopcache(
  op::TransientPGMDEIMOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_odeopcache(op.op,r,us)
end

function ODEs.update_odeopcache!(
  odeopcache,
  op::TransientPGMDEIMOperator,
  r::TransientRealization)

  update_odeopcache!(odeopcache,op.op,r)
end

function Algebra.allocate_residual(
  op::TransientPGMDEIMOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_residual(op.op,r,us,odeopcache)
end

function Algebra.allocate_jacobian(
  op::TransientPGMDEIMOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_jacobian(op.op,r,us,odeopcache)
end

function Algebra.residual!(
  cache,
  op::TransientPGMDEIMOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache;
  kwargs...)

  b,b̂ = cache
  fe_sb = fe_residual!(b,op,r,us,odeopcache)
  inv_project!(b̂,op.rhs,fe_sb)
end

function Algebra.jacobian!(
  cache,
  op::TransientPGMDEIMOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  A,Â = cache
  fe_sA = fe_jacobian!(A,op,r,us,ws,odeopcache)
  inv_project!(Â,op.lhs,fe_sA)
end

for f in (:(RBSteady.residual_snapshots),:(RBSteady.jacobian_snapshots))
  @eval begin
    function $f(solver::RBSolver,op::TransientPGMDEIMOperator,s)
      x = get_values(s)
      r = get_realization(s)
      fesolver = get_fe_solver(solver)
      odecache = allocate_odecache(fesolver,op,r,(x,))
      $f(fesolver,op,r,(x,),odecache)
    end
  end
end

function RBSteady.select_evalcache_at_indices(us::Tuple{Vararg{ConsecutiveArrayOfArrays}},odeopcache,indices)
  @unpack Us,Uts,tfeopcache,const_forms = odeopcache
  new_xhF = ()
  new_Us = ()
  for i = eachindex(Us)
    new_Us = (new_Us...,RBSteady.select_fe_space_at_indices(Us[i],indices))
    new_XhF_i = ConsecutiveArrayOfArrays(us[i].data[:,indices])
    new_xhF = (new_xhF...,new_XhF_i)
  end
  new_odeopcache = ODEOpFromTFEOpCache(new_Us,Uts,tfeopcache,const_forms)
  return new_xhF,new_odeopcache
end

function RBSteady.select_evalcache_at_indices(us::Tuple{Vararg{BlockVectorOfVectors}},odeopcache,indices)
  @unpack Us,Uts,tfeopcache,const_forms = odeopcache
  new_xhF = ()
  new_Us = ()
  for i = eachindex(Us)
    spacei = Us[i]
    VT = spacei.vector_type
    style = spacei.multi_field_style
    spacesi = [RBSteady.select_fe_space_at_indices(spaceij,indices) for spaceij in spacei]
    new_Us = (new_Us...,MultiFieldFESpace(VT,spacesi,style))
    new_XhF_i = mortar([ConsecutiveArrayOfArrays(us_i.data[:,indices]) for us_i in blocks(us[i])])
    new_xhF = (new_xhF...,new_XhF_i)
  end
  new_odeopcache = ODEOpFromTFEOpCache(new_Us,Uts,tfeopcache,const_forms)
  return new_xhF,new_odeopcache
end

function RBSteady.select_slvrcache_at_indices(cache::TupOfArrayContribution,indices)
  red_cache = ()
  for c in cache
    red_cache = (red_cache...,RBSteady.select_slvrcache_at_indices(c,indices))
  end
  return red_cache
end

function select_fe_quantities_at_indices(cache,us,odeopcache,indices)
  # returns the cache in the appropriate time-parameter locations
  red_cache = RBSteady.select_slvrcache_at_indices(cache,indices)
  # does the same with the stage variable `us` and the ode cache `odeopcache`
  red_us,red_odeopcache = RBSteady.select_evalcache_at_indices(us,odeopcache,indices)

  return red_cache,red_us,red_odeopcache
end

get_entry(s::AbstractParamArray,is,ipt) = consecutive_getindex(s,is,ipt)
get_entry(s::ParamSparseMatrix,is,ipt) = param_getindex(s,ipt)[is]

function RBSteady.select_at_indices(
  ::TransientHyperReduction,
  s::AbstractParamArray,
  ids_space,ids_time,ids_param)

  @check length(ids_space) == length(ids_time)
  entry = zeros(eltype2(a),length(ids_space))
  entries = array_of_consecutive_arrays(entry,length(ids_param))
  @inbounds for ip = param_eachindex(entries)
    for (i,(is,it)) in enumerate(zip(ids_space,ids_time))
      ipt = ip+(it-1)*length(ids_param)
      v = get_entry(a,is,ipt)
      consecutive_setindex!(entries,v,i,ip)
    end
  end
  return entries
end

function RBSteady.select_at_indices(
  ::TransientHyperReduction{<:TransientReduction},
  a::AbstractParamArray,
  ids_space,ids_time,ids_param)

  entry = zeros(eltype2(a),length(ids_space),length(ids_time))
  entries = array_of_consecutive_arrays(entry,length(ids_param))
  @inbounds for ip = param_eachindex(entries)
    for (i,it) in enumerate(ids_time)
      ipt = ip+(it-1)*length(ids_param)
      v = get_entry(a,ids_space,ipt)
      consecutive_setindex!(entries,v,:,i,ip)
    end
  end
  return entries
end

function RBSteady.select_at_indices(s::AbstractArray,a::TransientHyperReduction,indices::Range2D)
  ids_space = get_indices_space(a)
  ids_param = indices.axis1
  common_ids_time = indices.axis2
  domain_time = get_integration_domain_time(a)
  ids_time = RBSteady.ordered_common_locations(domain_time,common_ids_time)
  RBSteady.select_at_indices(a,s,ids_space,ids_time,ids_param)
end

function RBSteady.select_at_indices(
  s::ArrayContribution,a::AffineContribution,indices)
  contribution(s.trians) do trian
    RBSteady.select_at_indices(s[trian],a[trian],indices)
  end
end

function RBSteady.fe_jacobian!(
  A,
  op::TransientPGMDEIMOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  red_params = 1:num_params(r)
  red_times = union_indices_time(op.lhs)
  red_pt_indices = range_2d(red_params,red_times,num_params(r))
  red_r = r[red_params,red_times]

  red_A,red_us,red_odeopcache = select_fe_quantities_at_indices(A,us,odeopcache,vec(red_pt_indices))
  jacobian!(red_A,op.op,red_r,red_us,ws,red_odeopcache)
  map(red_A,op.lhs) do red_A,lhs
    RBSteady.select_at_indices(red_A,lhs,red_pt_indices)
  end
end

function RBSteady.fe_residual!(
  b,
  op::TransientPGMDEIMOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  red_params = 1:num_params(r)
  red_times = union_indices_time(op.rhs)
  red_pt_indices = range_2d(red_params,red_times,num_params(r))
  red_r = r[red_params,red_times]

  red_b,red_us,red_odeopcache = select_fe_quantities_at_indices(b,us,odeopcache,vec(red_pt_indices))
  residual!(red_b,op.op,red_r,red_us,red_odeopcache)
  RBSteady.select_at_indices(red_b,op.rhs,red_pt_indices)
end

function RBSteady.allocate_rbcache(
  op::TransientPGMDEIMOperator,
  r::TransientRealization)

  rhs_cache = RBSteady.allocate_hypred_cache(op.rhs,r)
  lhs_coeff = map(lhs -> RBSteady.allocate_coefficient(lhs,r),op.lhs)
  lhs_hypred = RBSteady.allocate_hyper_reduction(first(op.lhs),r)
  lhs_cache = (lhs_coeff,lhs_hypred)
  return lhs_cache,rhs_cache
end

"""
    struct LinearNonlinearTransientPGMDEIMOperator <: TransientRBOperator{LinearNonlinearParamODE} end

Extends the concept of [`TransientPGMDEIMOperator`](@ref) to accommodate the linear/nonlinear
splitting of terms in nonlinear applications

"""
struct LinearNonlinearTransientPGMDEIMOperator <: TransientRBOperator{LinearNonlinearParamODE}
  op_linear::TransientPGMDEIMOperator{<:AbstractLinearParamODE}
  op_nonlinear::TransientPGMDEIMOperator{NonlinearParamODE}
end

ParamSteady.get_linear_operator(op::LinearNonlinearTransientPGMDEIMOperator) = op.op_linear
ParamSteady.get_nonlinear_operator(op::LinearNonlinearTransientPGMDEIMOperator) = op.op_nonlinear

function FESpaces.get_test(op::LinearNonlinearTransientPGMDEIMOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_nonlinear)
end

function FESpaces.get_trial(op::LinearNonlinearTransientPGMDEIMOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_nonlinear)
end

function ParamDataStructures.realization(op::LinearNonlinearTransientPGMDEIMOperator;kwargs...)
  realization(op.op_nonlinear;kwargs...)
end

function ParamSteady.get_fe_operator(op::LinearNonlinearTransientPGMDEIMOperator)
  join_operators(ParamSteady.get_fe_operator(op.op_linear),ParamSteady.get_fe_operator(op.op_nonlinear))
end

function RBSteady.get_fe_trial(op::LinearNonlinearTransientPGMDEIMOperator)
  @check RBSteady.get_fe_trial(op.op_linear) === get_fe_trial(op.op_nonlinear)
  RBSteady.get_fe_trial(op.op_nonlinear)
end

function RBSteady.get_fe_test(op::LinearNonlinearTransientPGMDEIMOperator)
  @check RBSteady.get_fe_test(op.op_linear) === RBSteady.get_fe_test(op.op_nonlinear)
  RBSteady.get_fe_test(op.op_nonlinear)
end

function ParamSteady.get_vector_index_map(op::LinearNonlinearTransientPGMDEIMOperator)
  @check all(get_vector_index_map(op.op_linear) .== get_vector_index_map(op.op_nonlinear))
  get_vector_index_map(op.op_linear)
end

function ParamSteady.get_matrix_index_map(op::LinearNonlinearTransientPGMDEIMOperator)
  @check all(get_matrix_index_map(op.op_linear) .== get_matrix_index_map(op.op_nonlinear))
  get_matrix_index_map(op.op_linear)
end

function ODEs.allocate_odeopcache(
  op::LinearNonlinearTransientPGMDEIMOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_odeopcache(op.op_nonlinear,r,us)
end

function ODEs.update_odeopcache!(
  ode_cache,
  op::LinearNonlinearTransientPGMDEIMOperator,
  r::TransientRealization)

  update_odeopcache!(ode_cache,op.op_nonlinear,r)
end

function Algebra.allocate_residual(
  op::LinearNonlinearTransientPGMDEIMOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  b_lin = allocate_residual(op.op_linear,r,us,odeopcache)
  b_nlin = copy(b_lin)
  return b_lin,b_nlin
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearTransientPGMDEIMOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  A_lin = allocate_jacobian(op.op_linear,r,us,odeopcache)
  A_nlin = copy(A_lin)
  return A_lin,A_nlin
end

function Algebra.residual!(
  b::Tuple,
  op::LinearNonlinearTransientPGMDEIMOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache;
  kwargs...)

  b̂_lin,b_nlin = b
  fe_sb_nlin = fe_residual!(b_nlin,op.op_nonlinear,r,us,odeopcache)
  b̂_nlin = mdeim_result(op.op_nonlinear.rhs,fe_sb_nlin)
  @. b̂_nlin = b̂_nlin + b̂_lin
  return b̂_nlin
end

function Algebra.jacobian!(
  A::Tuple,
  op::LinearNonlinearTransientPGMDEIMOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  Â_lin,A_nlin = A
  fe_sA_nlin = fe_jacobian!(A_nlin,op.op_nonlinear,r,us,ws,odeopcache)
  Â_nlin = mdeim_result(op.op_nonlinear.lhs,fe_sA_nlin)
  @. Â_nlin = Â_nlin + Â_lin
  return Â_nlin
end

function RBSteady.allocate_rbcache(
  op::LinearNonlinearTransientPGMDEIMOperator,
  r::TransientRealization)

  cache_lin = RBSteady.allocate_rbcache(get_linear_operator(op),r)
  cache_nlin = RBSteady.allocate_rbcache(get_nonlinear_operator(op),r)
  return (cache_lin,cache_nlin)
end

# Solve a POD-MDEIM problem

function RBSteady.init_online_cache!(
  solver::RBSolver,
  op::TransientRBOperator,
  r::TransientRealization,
  y::AbstractParamVector)

  fesolver = get_fe_solver(solver)
  odecache = allocate_odecache(fesolver,op,r,(y,))
  rbcache = RBSteady.allocate_rbcache(op,r)

  cache = solver.cache
  cache.fecache = (y,odecache)
  cache.rbcache = rbcache
  return
end

function RBSteady.online_cache!(
  solver::RBSolver,
  op::TransientRBOperator,
  r::TransientRealization)

  cache = solver.cache
  (y,odecache) = cache.fecache
  param_length(r) != param_length(y) && RBSteady.init_online_cache!(solver,op,r,y)
  return
end

function Algebra.solve(
  solver::RBSolver,
  op::TransientRBOperator{NonlinearParamODE},
  r::AbstractRealization)

  @notimplemented "Split affine from nonlinear operator when running the RB solve"
end

function Algebra.solve(
  solver::RBSolver,
  op::TransientRBOperator,
  r::TransientRealization)

  fesolver = get_fe_solver(solver)
  trial = get_trial(op)(r)
  fe_trial = get_fe_trial(op)(r)
  x̂ = zero_free_values(trial)
  y = zero_free_values(fe_trial)
  RBSteady.init_online_cache!(solver,op,r,y)
  solve!(x̂,solver,op,r)
end

function Algebra.solve!(
  x̂,
  solver::RBSolver,
  op::TransientRBOperator,
  r::TransientRealization)

  RBSteady.online_cache!(solver,op,r)
  cache = solver.cache
  y,odecache = cache.fecache
  rbcache = cache.rbcache

  fesolver = get_fe_solver(solver)

  t = @timed solve!((x̂,),fesolver,op,r,(y,),(odecache,rbcache))
  stats = CostTracker(t,num_params(r))

  trial = get_trial(op)(r)
  x = inv_project(trial,x̂)

  return x,stats
end
