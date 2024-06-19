function RBSteady.reduced_operator(
  solver::RBSolver,
  op::TransientPODOperator,
  s)

  red_lhs,red_rhs = reduced_jacobian_residual(solver,op,s)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = map(get_domains,red_lhs)
  new_op = change_triangulation(op,trians_rhs,trians_lhs)
  TransientPODMDEIMOperator(new_op,red_lhs,red_rhs)
end

function RBSteady.reduced_operator(
  solver::RBSolver,
  op::TransientPODOperator{LinearNonlinearParamODE},
  s)

  red_op_lin = reduced_operator(solver,get_linear_operator(op),s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(op),s)
  LinearNonlinearTransientPODMDEIMOperator(red_op_lin,red_op_nlin)
end

struct TransientPODMDEIMOperator{T} <: TransientRBOperator{T}
  op::TransientPODOperator{T}
  lhs
  rhs
end

FESpaces.get_trial(op::TransientPODMDEIMOperator) = get_trial(op.op)
FESpaces.get_test(op::TransientPODMDEIMOperator) = get_test(op.op)
ParamDataStructures.realization(op::TransientPODMDEIMOperator;kwargs...) = realization(op.op;kwargs...)
ParamSteady.get_fe_operator(op::TransientPODMDEIMOperator) = ParamSteady.get_fe_operator(op.op)
ParamSteady.get_vector_index_map(op::TransientPODMDEIMOperator) = get_vector_index_map(op.op)
ParamSteady.get_matrix_index_map(op::TransientPODMDEIMOperator) = get_matrix_index_map(op.op)
RBSteady.get_fe_trial(op::TransientPODMDEIMOperator) = get_fe_trial(op.op)
RBSteady.get_fe_test(op::TransientPODMDEIMOperator) = get_fe_test(op.op)

function ODEs.allocate_odeopcache(
  op::TransientPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_odeopcache(op.op,r,us)
end

function ODEs.update_odeopcache!(
  ode_cache,
  op::TransientPODMDEIMOperator,
  r::TransientParamRealization)

  update_odeopcache!(ode_cache,op.op,r)
end

function Algebra.allocate_residual(
  op::TransientPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_residual(op.op,r,us,odeopcache)
end

function Algebra.allocate_jacobian(
  op::TransientPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_jacobian(op.op,r,us,odeopcache)
end

function Algebra.residual!(
  b::Contribution,
  op::TransientPODMDEIMOperator,
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
  op::TransientPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  fe_sA = fe_jacobian!(A,op,r,us,ws,odeopcache)
  Â = mdeim_result(op.lhs,fe_sA)
  return Â
end

function RBSteady.jacobian_and_residual(solver::RBSolver,op::TransientPODMDEIMOperator,s)
  x = get_values(s)
  r = get_realization(s)
  fesolver = get_fe_solver(solver)
  odecache = allocate_odecache(fesolver,op,r,(x,))
  jacobian_and_residual(fesolver,op,r,(x,),odecache)
end

function _select_fe_space_at_time_locations(fs::FESpace,indices)
  @notimplemented
end

function _select_fe_space_at_time_locations(fs::TrivialParamFESpace,indices)
  TrivialParamFESpace(fs.space,Val(length(indices)))
end

function _select_fe_space_at_time_locations(fs::SingleFieldParamFESpace,indices)
  dvi = ParamArray(fs.dirichlet_values.data[indices])
  TrialParamFESpace(dvi,fs.space)
end

function _select_odecache_at_time_locations(us::Tuple{Vararg{AbstractParamVector}},odeopcache,indices)
  @unpack Us,Uts,tfeopcache,const_forms = odeopcache
  new_xhF = ()
  new_Us = ()
  for i = eachindex(us)
    new_Us = (new_Us...,_select_fe_space_at_time_locations(Us[i],indices))
    new_xhF = (new_xhF...,ParamArray(us[i].data[indices]))
  end
  new_odeopcache = ODEOpFromTFEOpCache(new_Us,Uts,tfeopcache,const_forms)
  return new_xhF,new_odeopcache
end

function _select_odecache_at_time_locations(us::Tuple{Vararg{BlockVectorOfVectors}},odeopcache,indices)
  @unpack Us,Uts,tfeopcache,const_forms = odeopcache
  new_xhF = ()
  new_Us = ()
  for i = eachindex(Us)
    spacei = Us[i]
    VT = spacei.vector_type
    style = spacei.multi_field_style
    spacesi = [_select_fe_space_at_time_locations(spaceij,indices) for spaceij in spacei]
    new_Us = (new_Us...,MultiFieldFESpace(VT,spacesi,style))
    new_xhF = (new_xhF...,mortar([ParamArray(us_i.data[indices]) for us_i in blocks(us[i])]))
  end
  new_odeopcache = ODEOpFromTFEOpCache(new_Us,Uts,tfeopcache,const_forms)
  return new_xhF,new_odeopcache
end

function _select_cache_at_time_locations(b::ArrayOfArrays,indices)
  ArrayOfArrays(b.data[indices])
end

function _select_cache_at_time_locations(A::MatrixOfSparseMatricesCSC,indices)
  MatrixOfSparseMatricesCSC(A.m,A.n,A.colptr,A.rowval,A.data[:,indices])
end

function _select_cache_at_time_locations(A::BlockArrayOfArrays,indices)
  map(a -> _select_cache_at_time_locations(a,indices),blocks(A)) |> mortar
end

function _select_cache_at_time_locations(cache::ArrayContribution,indices)
  contribution(cache.trians) do trian
    _select_cache_at_time_locations(cache[trian],indices)
  end
end

function _select_cache_at_time_locations(cache::TupOfArrayContribution,indices)
  red_cache = ()
  for c in cache
    red_cache = (red_cache...,_select_cache_at_time_locations(c,indices))
  end
  return red_cache
end

function _select_indices_at_time_locations(red_times;nparams=1)
  vec(transpose((red_times.-1)*nparams .+ collect(1:nparams)'))
end

function _select_fe_quantities_at_time_locations(cache,a,r,us,odeopcache)
  red_times = union_reduced_times(a)
  red_r = r[:,red_times]
  indices = _select_indices_at_time_locations(red_times;nparams=num_params(r))
  red_cache = _select_cache_at_time_locations(cache,indices)
  red_xhF,red_odeopcache = _select_odecache_at_time_locations(us,odeopcache,indices)
  return red_cache,red_r,red_times,red_xhF,red_odeopcache
end

_time_locations(a,red_times)::Vector{Int} = filter(!isnothing,indexin(get_indices_time(a),red_times))

function _select_snapshots_at_space_time_locations(s::AbstractSnapshots,a,red_times)
  ids_space = RBSteady.get_indices_space(a)
  ids_time::Vector{Int} = _time_locations(a,red_times)
  select_snapshots_entries(s,ids_space,ids_time)
end

function _select_snapshots_at_space_time_locations(s::BlockSnapshots,a,red_times)
  ids_space = RBSteady.get_indices_space(a)
  active_block_ids = get_touched_blocks(a)
  block_map = BlockMap(size(a),active_block_ids)
  blocks = [_time_locations(a[i],red_times) for i in active_block_ids]
  ids_time = return_cache(block_map,blocks...)
  select_snapshots_entries(s,ids_space,ids_time)
end

function _select_snapshots_at_space_time_locations(
  s::ArrayContribution,a::AffineContribution,red_times)
  contribution(s.trians) do trian
    _select_snapshots_at_space_time_locations(s[trian],a[trian],red_times)
  end
end

function RBSteady.fe_jacobian!(
  cache,
  op::TransientPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  red_cache,red_r,red_times,red_us,red_odeopcache = _select_fe_quantities_at_time_locations(
    cache,op.lhs,r,us,odeopcache)
  A = jacobian!(red_cache,op.op,red_r,red_us,ws,red_odeopcache)
  Ai = map(A,op.lhs) do A,lhs
    _select_snapshots_at_space_time_locations(A,lhs,red_times)
  end
  return Ai
end

function RBSteady.fe_residual!(
  cache,
  op::TransientPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  red_cache,red_r,red_times,red_us,red_odeopcache = _select_fe_quantities_at_time_locations(
    cache,op.rhs,r,us,odeopcache)
  b = residual!(red_cache,op.op,red_r,red_us,red_odeopcache)
  bi = _select_snapshots_at_space_time_locations(b,op.rhs,red_times)
  return bi
end

struct LinearNonlinearTransientPODMDEIMOperator <: TransientRBOperator{LinearNonlinearParamODE}
  op_linear::TransientPODMDEIMOperator{AbstractLinearParamODE}
  op_nonlinear::TransientPODMDEIMOperator{NonlinearParamODE}
end

ParamSteady.get_linear_operator(op::LinearNonlinearTransientPODMDEIMOperator) = op.op_linear
ParamSteady.get_nonlinear_operator(op::LinearNonlinearTransientPODMDEIMOperator) = op.op_nonlinear

function FESpaces.get_test(op::LinearNonlinearTransientPODMDEIMOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_nonlinear)
end

function FESpaces.get_trial(op::LinearNonlinearTransientPODMDEIMOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_nonlinear)
end

function ParamDataStructures.realization(op::LinearNonlinearTransientPODMDEIMOperator;kwargs...)
  realization(op.op_nonlinear;kwargs...)
end

function ParamSteady.get_fe_operator(op::LinearNonlinearTransientPODMDEIMOperator)
  join_operators(get_fe_operator(op.op_linear),get_fe_operator(op.op_nonlinear))
end

function RBSteady.get_fe_trial(op::LinearNonlinearTransientPODMDEIMOperator)
  @check RBSteady.get_fe_trial(op.op_linear) === get_fe_trial(op.op_nonlinear)
  RBSteady.get_fe_trial(op.op_nonlinear)
end

function RBSteady.get_fe_test(op::LinearNonlinearTransientPODMDEIMOperator)
  @check RBSteady.get_fe_test(op.op_linear) === RBSteady.get_fe_test(op.op_nonlinear)
  RBSteady.get_fe_test(op.op_nonlinear)
end

function ODEs.allocate_odeopcache(
  op::LinearNonlinearTransientPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_odeopcache(op.op_nonlinear,r,us)
end

function ODEs.update_odeopcache!(
  ode_cache,
  op::LinearNonlinearTransientPODMDEIMOperator,
  r::TransientParamRealization)

  update_odeopcache!(ode_cache,op.op_nonlinear,r)
end

function Algebra.allocate_residual(
  op::LinearNonlinearTransientPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  b_lin = allocate_residual(op.op_linear,r,us,odeopcache)
  b_nlin = copy(b_lin)
  return b_lin,b_nlin
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearTransientPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  A_lin = allocate_jacobian(op.op_linear,r,us,odeopcache)
  A_nlin = copy(A_lin)
  return A_lin,A_nlin
end

function Algebra.residual!(
  b::Tuple,
  op::LinearNonlinearTransientPODMDEIMOperator,
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
  op::LinearNonlinearTransientPODMDEIMOperator,
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

function Algebra.solve(
  solver::RBSolver,
  op::TransientRBOperator{NonlinearParamODE},
  r::AbstractParamRealization)

  @notimplemented "Split affine from nonlinear operator when running the RB solve"
end

function Algebra.solve(
  solver::RBSolver,
  op::TransientRBOperator,
  r::TransientParamRealization)

  stats = @timed begin
    fesolver = get_fe_solver(solver)
    trial = get_trial(op)(r)
    fe_trial = get_fe_trial(op)(r)
    x̂ = zero_free_values(trial)
    y = zero_free_values(fe_trial)
    odecache = allocate_odecache(fesolver,op,r,(y,))
    solve!((x̂,),fesolver,op,r,(y,),odecache)
  end

  x = recast(x̂,trial)
  i = get_vector_index_map(op)
  s = Snapshots(x,i,r)
  cs = ComputationalStats(stats,num_params(r))
  return s,cs
end
