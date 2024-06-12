function ODEs.allocate_odeopcache(
  op::PODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_odeopcache(op.op,r,us)
end

function ODEs.update_odeopcache!(
  ode_cache,
  op::PODMDEIMOperator,
  r::TransientParamRealization)

  update_odeopcache!(ode_cache,op.op,r)
end

function Algebra.allocate_residual(
  op::PODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_residual(op.op,r,us,odeopcache)
end

function Algebra.allocate_jacobian(
  op::PODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_jacobian(op.op,r,us,odeopcache)
end

function Algebra.residual!(
  b::Contribution,
  op::PODMDEIMOperator,
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
  op::PODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  fe_sA = fe_jacobian!(A,op,r,us,ws,odeopcache)
  Â = mdeim_result(op.lhs,fe_sA)
  return Â
end

function jacobian_and_residual(solver::RBSolver,op::PODMDEIMOperator,s::AbstractTransientSnapshots)
  x = get_values(s)
  r = get_realization(s)
  fesolver = get_fe_solver(solver)
  odecache = allocate_odecache(fesolver,op,r,(x,))
  jacobian_and_residual(fesolver,op,r,(x,),odecache)
end

function _select_fe_space_at_time_locations(fs::FESpace,indices)
  @notimplemented
end

function _select_fe_space_at_time_locations(fs::FESpaceToParamFESpace,indices)
  FESpaceToParamFESpace(fs.space,Val(length(indices)))
end

function _select_fe_space_at_time_locations(fs::SingleFieldParamFESpace,indices)
  dvi = ParamArray(fs.dirichlet_values[indices])
  TrialParamFESpace(dvi,fs.space)
end

function _select_cache_at_time_locations(us::Tuple{Vararg{AbstractParamVector}},odeopcache,indices)
  @unpack Us,Uts,tfeopcache,const_forms = odeopcache
  new_xhF = ()
  new_Us = ()
  for i = eachindex(us)
    new_Us = (new_Us...,_select_fe_space_at_time_locations(Us[i],indices))
    new_xhF = (new_xhF...,us[i][indices])
  end
  new_odeopcache = ODEOpFromTFEOpCache(new_Us,Uts,tfeopcache,const_forms)
  return new_xhF,new_odeopcache
end

function _select_cache_at_time_locations(us::Tuple{Vararg{BlockVectorOfVectors}},odeopcache,indices)
  @unpack Us,Uts,tfeopcache,const_forms = odeopcache
  new_xhF = ()
  new_Us = ()
  for i = eachindex(Us)
    spacei = Us[i]
    VT = spacei.vector_type
    style = spacei.multi_field_style
    spacesi = [_select_fe_space_at_time_locations(spaceij,indices) for spaceij in spacei]
    new_Us = (new_Us...,MultiFieldParamFESpace(VT,spacesi,style))
    new_xhF = (new_xhF...,ParamArray(us[i][indices]))
  end
  new_odeopcache = ODEOpFromTFEOpCache(new_Us,Uts,tfeopcache,const_forms)
  return new_xhF,new_odeopcache
end

function _select_indices_at_time_locations(red_times;nparams=1)
  vec(transpose((red_times.-1)*nparams .+ collect(1:nparams)'))
end

function _select_fe_quantities_at_time_locations(a,r,us,odeopcache)
  red_times = union_reduced_times(a)
  red_r = r[:,red_times]
  indices = _select_indices_at_time_locations(red_times;nparams=num_params(r))
  red_xhF,red_odeopcache = _select_cache_at_time_locations(us,odeopcache,indices)
  return red_r,red_times,red_xhF,red_odeopcache
end

function _select_snapshots_at_space_time_locations(s,a,red_times)
  ids_space = get_indices_space(a)
  ids_time::Vector{Int} = filter(!isnothing,indexin(get_indices_time(a),red_times))
  srev = reverse_snapshots(s)
  select_snapshots_entries(srev,ids_space,ids_time)
end

function _select_snapshots_at_space_time_locations(
  s::ArrayContribution,a::AffineContribution,red_times)
  contribution(s.trians) do trian
    _select_snapshots_at_space_time_locations(s[trian],a[trian],red_times)
  end
end

function fe_jacobian!(
  cache,
  op::PODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  red_r,red_times,red_us,red_odeopcache = _select_fe_quantities_at_time_locations(op.lhs,r,us,odeopcache)
  A = jacobian!(cache,op.op,red_r,red_us,ws,red_odeopcache)
  Ai = map(A,op.lhs) do A,lhs
    _select_snapshots_at_space_time_locations(A,lhs,red_times)
  end
  return Ai
end

function fe_residual!(
  cache,
  op::PODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  red_r,red_times,red_us,red_odeopcache = _select_fe_quantities_at_time_locations(op.rhs,r,us,odeopcache)
  b = residual!(cache,op.op,red_r,red_us,red_odeopcache)
  bi = _select_snapshots_at_space_time_locations(b,op.rhs,red_times)
  return bi
end

function ODEs.allocate_odeopcache(
  op::LinearNonlinearPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_odeopcache(op.op_nonlinear,r,us)
end

function ODEs.update_odeopcache!(
  ode_cache,
  op::LinearNonlinearPODMDEIMOperator,
  r::TransientParamRealization)

  update_odeopcache!(ode_cache,op.op_nonlinear,r)
end

function Algebra.allocate_residual(
  op::LinearNonlinearPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  b_lin = allocate_residual(op.op_linear,r,us,odeopcache)
  b_nlin = copy(b_lin)
  return b_lin,b_nlin
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  A_lin = allocate_jacobian(op.op_linear,r,us,odeopcache)
  A_nlin = copy(A_lin)
  return A_lin,A_nlin
end

function Algebra.residual!(
  b::Tuple,
  op::LinearNonlinearPODMDEIMOperator,
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
  op::LinearNonlinearPODMDEIMOperator,
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

function Algebra.solve(
  solver::RBSolver,
  op::RBOperator,
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
  s = Snapshots(x,r)
  cs = ComputationalStats(stats,num_params(r))
  return s,cs
end
