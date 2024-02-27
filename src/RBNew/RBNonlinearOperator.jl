function reduced_operator(
  solver::RBSolver,
  op::RBOperator,
  s::S) where S

  red_lhs,red_rhs = reduced_matrix_vector_form(solver,op,s)
  trians_lhs = map(get_domains,red_lhs)
  trians_rhs = get_domains(red_rhs)
  new_op = change_triangulation(op,trians_lhs,trians_rhs)
  RBNonlinearOperator(solver,new_op,red_lhs,red_rhs)
end

abstract type RBNonlinearOperator{T} <: NonlinearOperator end

ReferenceFEs.get_order(op::RBNonlinearOperator) = get_order(op.op)
FESpaces.get_trial(op::RBNonlinearOperator) = get_trial(op.op)
FESpaces.get_test(op::RBNonlinearOperator) = get_test(op.op)
FEM.realization(op::RBNonlinearOperator;kwargs...) = realization(op.op;kwargs...)
FEM.get_fe_operator(op::RBNonlinearOperator) = FEM.get_fe_operator(op.op)
get_fe_trial(op::RBNonlinearOperator) = get_fe_trial(op.op)
get_fe_test(op::RBNonlinearOperator) = get_fe_test(op.op)

function TransientFETools.allocate_cache(
  op::RBNonlinearOperator,
  r::TransientParamRealization)

  allocate_cache(op.op,r)
end

function TransientFETools.update_cache!(
  ode_cache,
  op::RBNonlinearOperator,
  r::TransientParamRealization)

  update_cache!(ode_cache,op.op,r)
end

# cache for residual/jacobians includes:
# 1) cache to assemble residuals/jacobians on reduced integration domain
# 2) cache to compute the mdeim coefficient
# 3) cache to perform the kronecker product between basis and coefficient

function Algebra.allocate_residual(
  op::RBNonlinearOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  test = get_test(op)
  fe_b = allocate_fe_vector(op.op,r,x,ode_cache)
  coeff_cache = allocate_mdeim_coeff(op.rhs,r)
  lincomb_cache = allocate_mdeim_lincomb(test,r)
  return fe_b,coeff_cache,lincomb_cache
end

function Algebra.allocate_jacobian(
  op::RBNonlinearOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  trial = get_trial(op)
  test = get_test(op)
  fe_A = allocate_fe_matrix(op.op,r,x,ode_cache)
  coeff_cache = ()
  for i = 1:get_order(op)+1
    coeff_cache = (coeff_cache...,allocate_mdeim_coeff(op.lhs[i],r))
  end
  lincomb_cache = allocate_mdeim_lincomb(trial,test,r)
  return fe_A,coeff_cache,lincomb_cache
end

function Algebra.residual!(
  cache,
  op::RBNonlinearOperator,
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
  op::RBNonlinearOperator,
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

function _select_cache_at_time_locations(xhF::Tuple{Vararg{ParamVector}},ode_cache,indices)
  Us,Uts,fecache = ode_cache
  new_xhF = ()
  new_Us = ()
  for i = eachindex(xhF)
    new_Us = (new_Us...,_select_fe_space_at_time_locations(Us[i],indices))
    new_xhF = (new_xhF...,xhF[i][indices])
  end
  new_ode_cache = new_Us,Uts,fecache
  return new_xhF,new_ode_cache
end

function _select_cache_at_time_locations(xhF::Tuple{Vararg{ParamBlockVector}},ode_cache,indices)
  Us,Uts,fecache = ode_cache
  new_xhF = ()
  new_Us = ()
  for i = eachindex(xhF)
    spacei = Us[i]
    VT = spacei.vector_type
    style = spacei.multi_field_style
    spacesi = [_select_fe_space_at_time_locations(spaceij,indices) for spaceij in spacei]
    new_Us = (new_Us...,MultiFieldParamFESpace(VT,spacesi,style))
    new_xhF = (new_xhF...,ParamArray(xhF[i][indices]))
  end
  new_ode_cache = new_Us,Uts,fecache
  return new_xhF,new_ode_cache
end

function _select_indices_at_time_locations(red_times;nparams=1)
  vec(transpose((red_times.-1)*nparams .+ collect(1:nparams)'))
end

function _select_fe_quantities_at_time_locations(a,r,xhF,ode_cache)
  red_times = union_reduced_times(a)
  red_r = r[:,red_times]
  indices = _select_indices_at_time_locations(red_times;nparams=num_params(r))
  red_xhF,red_ode_cache = _select_cache_at_time_locations(xhF,ode_cache,indices)
  return red_r,red_times,red_xhF,red_ode_cache
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
  contribution(s.trians) do trian
    _select_snapshots_at_space_time_locations(s[trian],a[trian],red_times)
  end
end

function fe_matrix!(
  cache,
  op::RBNonlinearOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  γ::Tuple{Vararg{Real}},
  ode_cache)

  red_r,red_times,red_xhF,red_ode_cache = _select_fe_quantities_at_time_locations(op.lhs,r,xhF,ode_cache)
  A = fe_matrix!(cache,op.op,red_r,red_xhF,γ,red_ode_cache)
  map(A,op.lhs) do A,lhs
    _select_snapshots_at_space_time_locations(A,lhs,red_times)
  end
end

function fe_vector!(
  cache,
  op::RBNonlinearOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  red_r,red_times,red_xhF,red_ode_cache = _select_fe_quantities_at_time_locations(op.rhs,r,xhF,ode_cache)
  b = fe_vector!(cache,op.op,red_r,red_xhF,red_ode_cache)
  bi = _select_snapshots_at_space_time_locations(b,op.rhs,red_times)
  return bi
end

struct ThetaMethodNonlinearOperator{T,L,R} <: RBNonlinearOperator{T}
  op::RBOperator{T}
  lhs::L
  rhs::R
end

const AffineThetaMethodNonlinearOperator = ThetaMethodNonlinearOperator{T,L,R} where {T<:Affine,L,R}

function RBNonlinearOperator(::ThetaMethodRBSolver,op::RBOperator,lhs,rhs)
  ThetaMethodNonlinearOperator(op,lhs,rhs)
end

function Algebra.solve(
  solver::RBSolver,
  op::ThetaMethodNonlinearOperator,
  s::S) where S

  son = select_snapshots(s,online_params(solver))
  ron = get_realization(son)
  solve(solver,op,ron)
end

function Algebra.solve(
  solver::ThetaMethodRBSolver,
  op::ThetaMethodNonlinearOperator,
  _r::TransientParamRealization)

  fesolver = get_fe_solver(solver)
  dt = fesolver.dt
  θ = fesolver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ

  r = copy(_r)
  FEM.shift_time!(r,dt*(θ-1))

  trial = get_trial(op)(r)
  fe_trial = get_fe_trial(op)(r)
  red_x = zero_free_values(trial)
  y = zero_free_values(fe_trial)
  z = similar(y)
  z .= 0.0

  ode_cache = allocate_cache(op,r)
  nl_cache = nothing

  stats = @timed begin
    ode_cache = update_cache!(ode_cache,op,r)
    nlop = ThetaMethodParamOperator(op,r,dtθ,y,ode_cache,z)
    solve!(red_x,fesolver.nls,nlop,nl_cache)
  end

  x = recast(red_x,trial)
  s = reverse_snapshots(x,r)
  cs = ComputationalStats(stats,num_params(r))
  return s,cs
end

function Algebra.solve(
  solver::ThetaMethodRBSolver,
  op::AffineThetaMethodNonlinearOperator,
  _r::TransientParamRealization)

  fesolver = get_fe_solver(solver)
  dt = fesolver.dt
  θ = fesolver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ

  r = copy(_r)
  FEM.shift_time!(r,dt*(θ-1))

  trial = get_trial(op)(r)
  fe_trial = get_fe_trial(op)(r)
  red_x = zero_free_values(trial)
  y = zero_free_values(fe_trial)
  z = similar(y)
  z .= 0.0

  ode_cache = allocate_cache(op,r)
  mat_cache,vec_cache = ODETools._allocate_matrix_and_vector(op,r,y,ode_cache)

  stats = @timed begin
    ode_cache = update_cache!(ode_cache,op,r)
    A,b = ODETools._matrix_and_vector!(mat_cache,vec_cache,op,r,dtθ,y,ode_cache,z)
    afop = AffineOperator(A,b)
    solve!(red_x,fesolver.nls,afop)
  end

  x = recast(red_x,trial)
  s = reverse_snapshots(x,r)
  cs = ComputationalStats(stats,num_params(r))
  return s,cs
end

function ODETools._matrix_and_vector!(cache_mat,cache_vec,op::ThetaMethodNonlinearOperator,r,dtθ,u0,ode_cache,vθ)
  A = ODETools._matrix!(cache_mat,op,r,dtθ,u0,ode_cache,vθ)
  b = ODETools._vector!(cache_vec,op,r,dtθ,u0,ode_cache,vθ)
  return A,b
end

function ODETools._matrix!(cache,op::ThetaMethodNonlinearOperator,r,dtθ,u0,ode_cache,vθ)
  fe_A,coeff_cache,lincomb_cache = cache
  for i = eachindex(fe_A)
    LinearAlgebra.fillstored!(fe_A[i],zero(eltype(fe_A[i])))
  end
  A = ODETools.jacobians!(cache,op,r,(u0,vθ),(1.0,1/dtθ),ode_cache)
  return A
end

function ODETools._vector!(cache,op::ThetaMethodNonlinearOperator,r,dtθ,u0,ode_cache,vθ)
  b = residual!(cache,op,r,(u0,vθ),ode_cache)
  b .*= -1.0
  return b
end
