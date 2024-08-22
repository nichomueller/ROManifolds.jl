function RBSteady.reduced_operator(
  solver::RBSolver,
  op::TransientPGOperator,
  s)

  red_lhs,red_rhs = reduced_jacobian_residual(solver,op,s)
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
  lhs
  rhs
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
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_odecache(fesolver,op.op,r,us)
end

function ODEs.allocate_odeopcache(
  op::TransientPGMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_odeopcache(op.op,r,us)
end

function ODEs.update_odeopcache!(
  ode_cache,
  op::TransientPGMDEIMOperator,
  r::TransientParamRealization)

  @warn "For performance reasons, it would be best to update the cache at the very
    start, given that the online phase of a space-time ROM is time-independent"
  update_odeopcache!(ode_cache,op.op,r)
end

function Algebra.allocate_residual(
  op::TransientPGMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_residual(op.op,r,us,odeopcache)
end

function Algebra.allocate_jacobian(
  op::TransientPGMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_jacobian(op.op,r,us,odeopcache)
end

function Algebra.residual!(
  b::Contribution,
  op::TransientPGMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache;
  kwargs...)

  fe_sb = fe_residual!(b,op,r,us,odeopcache)
  b̂ = mdeim_result(op.rhs,fe_sb)
  return b̂
end

function Algebra.jacobian!(
  A::TupOfArrayContribution,
  op::TransientPGMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  fe_sA = fe_jacobian!(A,op,r,us,ws,odeopcache)
  Â = mdeim_result(op.lhs,fe_sA)
  return Â
end

function RBSteady.jacobian_and_residual(solver::RBSolver,op::TransientPGMDEIMOperator,s)
  x = get_values(s)
  r = get_realization(s)
  fesolver = get_fe_solver(solver)
  odecache = allocate_odecache(fesolver,op,r,(x,))
  jacobian_and_residual(fesolver,op,r,(x,),odecache)
end

function RBSteady.select_cache_at_indices(us::Tuple{Vararg{ConsecutiveArrayOfArrays}},odeopcache,indices)
  @unpack Us,Uts,tfeopcache,const_forms = odeopcache
  new_xhF = ()
  new_Us = ()
  for i = eachindex(us)
    new_Us = (new_Us...,RBSteady.select_fe_space_at_indices(Us[i],indices))
    new_XhF_i = ConsecutiveArrayOfArrays(us[i].data[:,indices])
    new_xhF = (new_xhF...,new_XhF_i)
  end
  new_odeopcache = ODEOpFromTFEOpCache(new_Us,Uts,tfeopcache,const_forms)
  return new_xhF,new_odeopcache
end

function RBSteady.select_cache_at_indices(us::Tuple{Vararg{BlockVectorOfVectors}},odeopcache,indices)
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

function RBSteady.select_fe_quantities_at_indices(cache,r,us,odeopcache,param_ids,time_ids)
  # common temporal reduced integration domain
  red_r = r[:,time_ids]
  # indices corresponding to the newly found temporal reduced integration domain,
  # for every parameter
  indices = _param_time_range(param_ids,time_ids,num_params(r))
  # returns the cache in the appropriate time-parameter locations
  red_cache = RBSteady.select_slvrcache_at_indices(cache,indices)
  # does the same with the stage variable `us` and the ode cache `odeopcache`
  red_us,red_odeopcache = RBSteady.select_cache_at_indices(us,odeopcache,indices)
  return red_cache,red_r,red_us,red_odeopcache
end

# from `red_times`, i.e. the union of all the reduced time indices across every
# affine contribution, finds the entries of the reduced time indices specific to
# the affine decomposition `a`
_time_locations(a,red_times)::Vector{Int} = filter(!isnothing,indexin(get_indices_time(a),red_times))

# selects the entries of the snapshots relevant to the reduced integration domain
# in `a`
function RBSteady._select_snapshots_at_indices(s::AbstractSnapshots,a,red_times)
  ids_space = RBSteady.get_indices_space(a)
  ids_time::Vector{Int} = _time_locations(a,red_times)
  select_snapshots_entries(s,ids_space,ids_time)
end

function RBSteady._select_snapshots_at_indices(s::BlockSnapshots,a,red_times)
  ids_space = RBSteady.get_indices_space(a)
  active_block_ids = get_touched_blocks(a)
  block_map = BlockMap(size(a),active_block_ids)
  blocks = [_time_locations(a[i],red_times) for i in active_block_ids]
  ids_time = return_cache(block_map,blocks...)
  select_snapshots_entries(s,ids_space,ids_time)
end

function RBSteady._select_snapshots_at_indices(
  s::ArrayContribution,a::AffineContribution,red_times)
  contribution(s.trians) do trian
    RBSteady._select_snapshots_at_indices(s[trian],a[trian],red_times)
  end
end

function RBSteady.fe_jacobian!(
  cache,
  op::TransientPGMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  red_params = 1:num_params(r)
  red_times = union_reduced_times(op.lhs)
  red_cache,red_r,red_us,red_odeopcache = select_fe_quantities_at_indices(
    cache,r,us,odeopcache,red_params,red_times)
  A = jacobian!(red_cache,op.op,red_r,red_us,ws,red_odeopcache)
  Ai = map(A,op.lhs) do A,lhs
    RBSteady._select_snapshots_at_indices(A,lhs,red_times)
  end
  return Ai
end

function RBSteady.fe_residual!(
  cache,
  op::TransientPGMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  red_params = 1:num_params(r)
  red_times = union_reduced_times(op.rhs)
  red_cache,red_r,red_us,red_odeopcache = select_fe_quantities_at_indices(
    cache,r,us,odeopcache,red_params,red_times)
  b = residual!(red_cache,op.op,red_r,red_us,red_odeopcache)
  bi = RBSteady._select_snapshots_at_indices(b,op.rhs,red_times)
  return bi
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
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_odeopcache(op.op_nonlinear,r,us)
end

function ODEs.update_odeopcache!(
  ode_cache,
  op::LinearNonlinearTransientPGMDEIMOperator,
  r::TransientParamRealization)

  update_odeopcache!(ode_cache,op.op_nonlinear,r)
end

function Algebra.allocate_residual(
  op::LinearNonlinearTransientPGMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  b_lin = allocate_residual(op.op_linear,r,us,odeopcache)
  b_nlin = copy(b_lin)
  return b_lin,b_nlin
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearTransientPGMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  A_lin = allocate_jacobian(op.op_linear,r,us,odeopcache)
  A_nlin = copy(A_lin)
  return A_lin,A_nlin
end

function Algebra.residual!(
  b::Tuple,
  op::LinearNonlinearTransientPGMDEIMOperator,
  r::TransientParamRealization,
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
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  Â_lin,A_nlin = A
  fe_sA_nlin = fe_jacobian!(A_nlin,op.op_nonlinear,r,us,ws,odeopcache)
  Â_nlin = mdeim_result(op.op_nonlinear.lhs,fe_sA_nlin)
  @. Â_nlin = Â_nlin + Â_lin
  return Â_nlin
end

# Solve a POD-MDEIM problem

function Algebra.solve(solver::RBSolver,op::TransientRBOperator,s)
  son = select_snapshots(s,RBSteady.online_params(solver))
  ron = get_realization(son)
  solve(solver,op,ron)
end

function Algebra.solve!(cache,solver::RBSolver,op::TransientRBOperator,s)
  son = select_snapshots(s,online_params(solver))
  ron = get_realization(son)
  solve!(cache,solver,op,ron)
end

function Algebra.solve(
  solver::RBSolver,
  op::TransientRBOperator{NonlinearParamODE},
  r::AbstractParamRealization)

  @notimplemented "Split affine from nonlinear operator when running the RB solve"
end

function Algebra.solve!(
  cache,
  solver::RBSolver,
  op::TransientRBOperator{NonlinearParamODE},
  r::AbstractParamRealization)

  @notimplemented "Split affine from nonlinear operator when running the RB solve"
end

function Algebra.solve(
  solver::RBSolver,
  op::TransientRBOperator,
  r::TransientParamRealization)

  fesolver = get_fe_solver(solver)
  trial = get_trial(op)(r)
  fe_trial = get_fe_trial(op)(r)
  x̂ = zero_free_values(trial)
  y = zero_free_values(fe_trial)
  odecache = allocate_odecache(fesolver,op,r,(y,))
  cache = x̂,y,odecache
  solve!(cache,solver,op,r)
end

function Algebra.solve!(
  cache,
  solver::RBSolver,
  op::TransientRBOperator,
  r::TransientParamRealization)

  x̂,y,odecache = cache
  fesolver = get_fe_solver(solver)
  timer = get_timer(solver)
  reset_timer!(timer,"TEST")

  @timeit timer "TEST" solve!((x̂,),fesolver,op,r,(y,),odecache)
  set_nruns!(timer["TEST"],num_params(r))

  trial = get_trial(op)(r)
  x = recast(x̂,trial)
  i = get_vector_index_map(op)
  s = Snapshots(x,i,r)

  return s,cache
end
