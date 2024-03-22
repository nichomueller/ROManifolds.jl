function reduced_operator(
  solver::RBSolver,
  op::PODOperator,
  s::S) where S

  red_lhs,red_rhs = reduced_matrix_vector_form(solver,op,s)
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
  odeopcache)

  fe_sb = fe_residual!(b,op,r,us,odeopcache)
  b̂ = mdeim_residual(op.rhs,fe_sb)
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
  Â = mdeim_jacobian(op.lhs,fe_sA)
  return Â
end

# function FEM.jacobian_and_residual(solver::RBSolver,op::PODMDEIMOperator,s::S) where S
#   x = get_values(s)
#   r = get_realization(s)
#   odeopcache = allocate_odeopcache(op,r,(x,))
#   b = allocate_residual(op,r,(x,),odeopcache)
#   A = allocate_jacobian(op,r,(x,),odeopcache)
#   odeopcache = update_odeopcache!(odeopcache,op,r)
#   return residual!()
# end

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
  ids_time = filter(!isnothing,indexin(get_indices_time(a),red_times))
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
    @check isa(op_linear,PODMDEIMOperator{LinearODE})
    new(op_linear,op_nonlinear)
  end
end

FEM.get_linear_operator(op::LinearNonlinearPODMDEIMOperator) = op.op_linear
FEM.get_nonlinear_operator(op::LinearNonlinearPODMDEIMOperator) = op.op_nonlinear

function FESpaces.get_test(op::LinearNonlinearPODMDEIMOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_linear)
end

function FESpaces.get_trial(op::LinearNonlinearPODMDEIMOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_linear)
end

function FEM.realization(op::LinearNonlinearPODMDEIMOperator;kwargs...)
  realization(op.op_linear;kwargs...)
end

function FEM.get_fe_operator(op::LinearNonlinearPODMDEIMOperator)
  FEM.get_fe_operator(op.op_linear),FEM.get_fe_operator(op.op_nonlinear)
end

function get_fe_trial(op::LinearNonlinearPODMDEIMOperator)
  @check get_fe_trial(op.op_linear) === get_fe_trial(op.op_nonlinear)
  get_fe_trial(op.op_linear)
end

function get_fe_test(op::LinearNonlinearPODMDEIMOperator)
  @check get_fe_test(op.op_linear) === get_fe_test(op.op_nonlinear)
  get_fe_test(op.op_linear)
end

function ODEs.allocate_odeopcache(
  op::LinearNonlinearPODMDEIMOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}})

  allocate_odeopcache(op.op_linear,r,us)
end

function ODEs.update_odeopcache!(
  ode_cache,
  op::LinearNonlinearPODMDEIMOperator,
  r::TransientParamRealization)

  update_odeopcache!(ode_cache,op.op_linear,r)
end

# function Algebra.allocate_residual(
#   op::LinearNonlinearPODMDEIMOperator,
#   r::TransientParamRealization,
#   x::AbstractVector,
#   ode_cache)

#   cache_lin = allocate_residual(op.op_linear,r,x,ode_cache)
#   cache_nlin = allocate_residual(op.op_nonlinear,r,x,ode_cache)
#   return cache_lin,cache_nlin
# end

# function Algebra.allocate_jacobian(
#   op::LinearNonlinearPODMDEIMOperator,
#   r::TransientParamRealization,
#   x::AbstractVector,
#   ode_cache)

#   cache_lin = allocate_jacobian(op.op_linear,r,x,ode_cache)
#   cache_nlin = allocate_jacobian(op.op_nonlinear,r,x,ode_cache)
#   return cache_lin,cache_nlin
# end

# # we assume that the linear components have already been computed during the
# # first call to residual() and jacobian() (first iteration of the Newton solver)
# function Algebra.residual!(
#   cache,
#   op::LinearNonlinearPODMDEIMOperator,
#   r::TransientParamRealization,
#   us::Tuple{Vararg{AbstractVector}},
#   ode_cache)

#   b_lin,cache_nl = cache
#   b_nlin = residual!(cache_nl,op.op_nonlinear,r,us,ode_cache)
#   # @. b_nlin = b_nlin - b_lin
#   # return b_nlin
#   return b_nlin - b_lin
# end

# function Algebra.jacobian!(
#   cache,
#   op::LinearNonlinearPODMDEIMOperator,
#   r::TransientParamRealization,
#   us::Tuple{Vararg{AbstractVector}},
#   i::Integer,
#   γᵢ::Real,
#   ode_cache)

#   A_lin,cache_nl = cache
#   A_nlin = jacobian!(cache_nl,op.op_nonlinear,r,us,i,γᵢ,ode_cache)
#   # @. A_nlin = A_nlin + A_lin
#   # return A_nlin
#   return A_nlin + A_lin
# end

# function ODEs.jacobians!(
#   cache,
#   op::LinearNonlinearPODMDEIMOperator,
#   r::TransientParamRealization,
#   us::Tuple{Vararg{AbstractVector}},
#   γ::Tuple{Vararg{Real}},
#   ode_cache)

#   A_lin,cache_nl = cache
#   A_nlin = jacobians!(cache_nl,op.op_nonlinear,r,us,γ,ode_cache)
#   # @. A_nlin = A_nlin + A_lin
#   # return A_nlin
#   return A_nlin + A_lin
# end

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
  s::S) where S

  @notimplemented "Split affine from nonlinear operator when running the RB solve"
end

# θ-Method specialization

# Solve a POD-MDEIM problem, linear case

function Algebra.solve(
  solver::ThetaMethodRBSolver,
  op::RBOperator{LinearParamODE},
  r::TransientParamRealization)

  fesolver = get_fe_solver(solver)
  dt = fesolver.dt
  θ = fesolver.θ

  FEM.shift_time!(r,dt*(θ-1))

  trial = get_trial(op)(r)
  fe_trial = get_fe_trial(op)(r)
  x̂ = zero_free_values(trial)
  y = zero_free_values(fe_trial)
  z = copy(y)
  us = (y,z)
  ws = (1,1)

  sysslvr = fesolver.sysslvr
  odecache = allocate_odecache(fesolver,op,r,(y,))
  odeslvrcache,odeopcache = odecache
  reuse,A,b,sysslvrcache = odeslvrcache

  stats = @timed begin
    update_odeopcache!(odeopcache,op,r)
    stageop = LinearParamStageOperator(op,odeopcache,r,us,ws,A,b,reuse,sysslvrcache)
    sysslvrcache = solve!(x̂,sysslvr,stageop,sysslvrcache)
  end

  FEM.shift_time!(r,dt*(1-θ))
  x = recast(x̂,trial)
  s = Snapshots(x,r)
  cs = ComputationalStats(stats,num_params(r))
  return s,cs
end

# # Nonlinear case

# function Algebra.solve(
#   solver::ThetaMethodRBSolver,
#   op::RBOperator{LinearNonlinearParamODE},
#   _r::TransientParamRealization)

#   fesolver = get_fe_solver(solver)
#   dt = fesolver.dt
#   θ = fesolver.θ
#   θ == 0.0 ? dtθ = dt : dtθ = dt*θ

#   r = copy(_r)
#   FEM.shift_time!(r,dt*(θ-1))

#   trial = get_trial(op)(r)
#   fe_trial = get_fe_trial(op)(r)
#   red_x = zero_free_values(trial)
#   y = zero_free_values(fe_trial)
#   z = similar(y)
#   z .= 0.0

#   ode_cache = allocate_cache(op,r)
#   cache_lin = ODETools._allocate_matrix_and_vector(op.op_linear,r,y,ode_cache)
#   cache_nlin = ODETools._allocate_matrix_and_vector(op.op_nonlinear,r,y,ode_cache)
#   cache = cache_lin,cache_nlin

#   stats = @timed begin
#     ode_cache = update_cache!(ode_cache,op,r)
#     nlop = RBThetaMethodParamOperator(op,r,dtθ,y,ode_cache,z)
#     solve!(red_x,fesolver.nls,nlop,cache)
#   end

#   x = recast(red_x,trial)
#   s = Snapshots(x,r)
#   cs = ComputationalStats(stats,num_params(r))
#   return s,cs
# end

# for testing/visualization purposes

function pod_mdeim_error(solver,feop,op::RBOperator,s::AbstractArray)
  pod_err = pod_error(get_trial(op),s,assemble_norm_matrix(feop))
  mdeim_err = mdeim_error(solver,feop,op,s)
  return pod_err,mdeim_err
end
