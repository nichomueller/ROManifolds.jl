abstract type Affinity end
struct ZeroAffinity <: Affinity end
struct ParamAffinity <: Affinity end
struct TimeAffinity <: Affinity end
struct ParamTimeAffinity <: Affinity end
struct NonAffinity <: Affinity end

function affinity_residual(
  op::ParamFEOperator,
  ::FESolver,
  params::Table;
  filter=(1,1),
  ntests=10,
  trian=get_triangulation(feop.test))

  row, = filter
  test_row = get_test(op)[row]
  dv_row = _get_fe_basis(op.test,row)
  u = allocate_evaluation_function(op)

  μ = first(params)
  d = collect_cell_contribution(test_row,op.res(μ,u,dv_row),trian)
  if all(isempty,d)
    return ZeroAffinity()
  end

  cell = find_nonzero_cell_contribution(d)
  d0 = max.(abs.(d[cell]),eps())

  for μ in rand(params,ntests)
    d = collect_cell_contribution(test_row,op.res(μ,u,dv_row),trian)
    ratio = d[cell] ./ d0[cell]
    if !all(ratio .== ratio[1])
      return NonAffinity()
    end
  end

  return ParamAffinity()
end

function affinity_residual(
  op::ParamTransientFEOperator,
  solver::ODESolver,
  params::Table;
  filter=(1,1),
  ntests=10,
  trian=get_triangulation(feop.test))

  row, = filter
  times = get_times(solver)
  test_row = get_test(op)[row]
  dv_row = _get_fe_basis(op.test,row)
  u = allocate_evaluation_function(op)

  μ,t = first(params),first(times)
  d = collect_cell_contribution(test_row,op.res(μ,t,u,dv_row),trian)
  if all(isempty,d)
    return ZeroAffinity()
  end

  cell = find_nonzero_cell_contribution(d)
  d0 = max.(abs.(d[cell]),eps())

  for μ in rand(params,ntests)
    d = collect_cell_contribution(test_row,op.res(μ,t,u,dv_row),trian)
    ratio = d[cell] ./ d0[cell]
    if !all(ratio .== ratio[1])
      return NonAffinity()
    end
  end

  for t in rand(params,ntests)
    d = collect_cell_contribution(test_row,op.res(μ,t,u,dv_row),trian)
    ratio = d[cell] ./ d0[cell]
    if !all(ratio .== ratio[1])
      return ParamAffinity()
    end
  end

  return ParamTimeAffinity()
end

function affinity_jacobian(
  op::ParamFEOperator,
  ::FESolver,
  params::Table;
  filter=(1,1),
  ntests=10,
  trian=get_triangulation(feop.test))

  row, = filter
  test_row = get_test(op)[row]
  trial_col = get_trial(op)[col]
  dv_row = _get_fe_basis(op.test,row)
  du_col = _get_trial_fe_basis(get_trial(op)(nothing),col)
  u = allocate_evaluation_function(op)

  μ = first(params)
  d = collect_cell_contribution(trial_col,test_row,op.jac(μ,u,du_col,dv_row),trian)
  if all(isempty,d)
    return ZeroAffinity()
  end

  cell = find_nonzero_cell_contribution(d)
  d0 = max.(abs.(d[cell]),eps())

  for μ in rand(params,ntests)
    d = collect_cell_contribution(trial_col,test_row,op.jac(μ,u,du_col,dv_row),trian)
    ratio = d[cell] ./ d0[cell]
    if !all(ratio .== ratio[1])
      return NonAffinity()
    end
  end

  return ParamAffinity()
end

function affinity_residual(
  op::ParamTransientFEOperator,
  solver::ODESolver,
  params::Table;
  filter=(1,1),
  ntests=10,
  trian=get_triangulation(feop.test))

  row, = filter
  times = get_times(solver)
  test_row = get_test(op)[row]
  trial_col = get_trial(op)[col]
  dv_row = _get_fe_basis(op.test,row)
  du_col = _get_trial_fe_basis(get_trial(op)(nothing,nothing),col)
  u = allocate_evaluation_function(op)

  μ,t = first(params),first(times)
  d = collect_cell_contribution(trial_col,test_row,op.jac(μ,t,u,du_col,dv_row),trian)
  if all(isempty,d)
    return ZeroAffinity()
  end

  cell = find_nonzero_cell_contribution(d)
  d0 = max.(abs.(d[cell]),eps())

  for μ in rand(params,ntests)
    d = collect_cell_contribution(trial_col,test_row,op.jac(μ,t,u,du_col,dv_row),trian)
    ratio = d[cell] ./ d0[cell]
    if !all(ratio .== ratio[1])
      return NonAffinity()
    end
  end

  for t in rand(params,ntests)
    d = collect_cell_contribution(trial_col,test_row,op.jac(μ,t,u,du_col,dv_row),trian)
    ratio = d[cell] ./ d0[cell]
    if !all(ratio .== ratio[1])
      return ParamAffinity()
    end
  end

  return ParamTimeAffinity()
end

function find_nonzero_cell_contribution(data)
  global idx
  for (i,celli) in data
    if !iszero(sum(abs.(celli)))
      return i
    end
  end
  @unreachable
end
