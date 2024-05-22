function reduced_operator(
  solver::RBSolver,
  op::PODOperator,
  s::S) where S

  red_lhs,red_rhs = reduced_jacobian_residual(solver,op,s)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = map(get_domains,red_lhs)
  new_op = change_triangulation(op,trians_rhs,trians_lhs)
  PODMDEIMOperator(new_op,red_lhs,red_rhs)
end

function reduced_operator(
  solver::RBSolver,
  op::PODOperator{LinearNonlinearParamODE},
  s::S) where S

  red_op_lin = reduced_operator(solver,get_linear_operator(op),s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(op),s)
  LinearNonlinearPODMDEIMOperator(red_op_lin,red_op_nlin)
end

struct PODMDEIMOperator{T} <: RBOperator{T}
  op::PODOperator{T}
  lhs
  rhs
end

FESpaces.get_trial(op::PODMDEIMOperator) = get_trial(op.op)
FESpaces.get_test(op::PODMDEIMOperator) = get_test(op.op)
FEM.realization(op::PODMDEIMOperator;kwargs...) = realization(op.op;kwargs...)
FEM.get_fe_operator(op::PODMDEIMOperator) = FEM.get_fe_operator(op.op)
get_fe_trial(op::PODMDEIMOperator) = get_fe_trial(op.op)
get_fe_test(op::PODMDEIMOperator) = get_fe_test(op.op)

function ODEs.allocate_odeopcache(
  op::PODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}})

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
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  allocate_residual(op.op,r,us,odeopcache)
end

function Algebra.allocate_jacobian(
  op::PODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  allocate_jacobian(op.op,r,us,odeopcache)
end

function Algebra.residual!(
  b::Contribution,
  op::PODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
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
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  fe_sA = fe_jacobian!(A,op,r,us,ws,odeopcache)
  Â = mdeim_result(op.lhs,fe_sA)
  return Â
end

function jacobian_and_residual(solver::RBSolver,op::PODMDEIMOperator,s::S) where S
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

function _select_cache_at_time_locations(us::Tuple{Vararg{ParamVector}},odeopcache,indices)
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

function _select_cache_at_time_locations(us::Tuple{Vararg{ParamBlockVector}},odeopcache,indices)
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
  us::Tuple{Vararg{AbstractVector}},
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
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  red_r,red_times,red_us,red_odeopcache = _select_fe_quantities_at_time_locations(op.rhs,r,us,odeopcache)
  b = residual!(cache,op.op,red_r,red_us,red_odeopcache)
  bi = _select_snapshots_at_space_time_locations(b,op.rhs,red_times)
  return bi
end

struct LinearNonlinearPODMDEIMOperator <: RBOperator{LinearNonlinearParamODE}
  op_linear::PODMDEIMOperator
  op_nonlinear::PODMDEIMOperator
  function LinearNonlinearPODMDEIMOperator(op_linear,op_nonlinear)
    @check isa(op_linear,PODMDEIMOperator{LinearParamODE})
    new(op_linear,op_nonlinear)
  end
end

FEM.get_linear_operator(op::LinearNonlinearPODMDEIMOperator) = op.op_linear
FEM.get_nonlinear_operator(op::LinearNonlinearPODMDEIMOperator) = op.op_nonlinear

function FESpaces.get_test(op::LinearNonlinearPODMDEIMOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_nonlinear)
end

function FESpaces.get_trial(op::LinearNonlinearPODMDEIMOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_nonlinear)
end

function FEM.realization(op::LinearNonlinearPODMDEIMOperator;kwargs...)
  realization(op.op_nonlinear;kwargs...)
end

function FEM.get_fe_operator(op::LinearNonlinearPODMDEIMOperator)
  join_operators(FEM.get_fe_operator(op.op_linear),FEM.get_fe_operator(op.op_nonlinear))
end

function get_fe_trial(op::LinearNonlinearPODMDEIMOperator)
  @check get_fe_trial(op.op_linear) === get_fe_trial(op.op_nonlinear)
  get_fe_trial(op.op_nonlinear)
end

function get_fe_test(op::LinearNonlinearPODMDEIMOperator)
  @check get_fe_test(op.op_linear) === get_fe_test(op.op_nonlinear)
  get_fe_test(op.op_nonlinear)
end

function ODEs.allocate_odeopcache(
  op::LinearNonlinearPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}})

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
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  b_lin = allocate_residual(op.op_linear,r,us,odeopcache)
  b_nlin = copy(b_lin)
  return b_lin,b_nlin
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  A_lin = allocate_jacobian(op.op_linear,r,us,odeopcache)
  A_nlin = copy(A_lin)
  return A_lin,A_nlin
end

function Algebra.residual!(
  b::Tuple,
  op::LinearNonlinearPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
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
  us::Tuple{Vararg{AbstractVector}},
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
  s::S) where S

  son = select_snapshots(s,online_params(solver))
  ron = get_realization(son)
  solve(solver,op,ron)
end

function Algebra.solve(
  solver::RBSolver,
  op::RBOperator{NonlinearParamODE},
  r::TransientParamRealization)

  @notimplemented "Split affine from nonlinear operator when running the RB solve"
end

function Algebra.solve(
  solver::ThetaMethodRBSolver,
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

# for testing/visualization purposes

function pod_mdeim_error(solver,feop,op::RBOperator,s::AbstractArray)
  pod_err = pod_error(get_trial(op),s,assemble_norm_matrix(feop))
  mdeim_err = mdeim_error(solver,feop,op,s)
  return pod_err,mdeim_err
end
