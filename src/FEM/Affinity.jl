abstract type Affinity end
struct ZeroAffinity <: Affinity end
struct ParamAffinity <: Affinity end
struct TimeAffinity <: Affinity end
struct ParamTimeAffinity <: Affinity end
struct NonAffinity <: Affinity end

function affinity_residual(
  op::ParamTransientFEOperator,
  solver::ODESolver,
  params::Table,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}};
  ntests::Int=10)

  row, = filter
  times = get_times(solver)
  test_row = get_test(op)[row]
  dv_row = _get_fe_basis(op.test,row)
  u = allocate_evaluation_function(op)

  μ,t = rand(params),rand(times)
  d = collect_cell_contribution(test_row,op.res(μ,t,u,dv_row),trian)
  if all(isempty,d)
    return ZeroAffinity()
  end

  dir_cells = op.test.dirichlet_cells
  cell = find_nonzero_cell_contribution(d,dir_cells)
  d0 = max.(abs.(d[cell]),eps())

  for μ in rand(params,ntests)
    d = collect_cell_contribution(test_row,op.res(μ,t,u,dv_row),trian)
    ratio = d[cell] ./ d0
    if !all(ratio .== ratio[1])
      return NonAffinity()
    end
  end

  for t in rand(times,ntests)
    d = collect_cell_contribution(test_row,op.res(μ,t,u,dv_row),trian)
    ratio = d[cell] ./ d0
    if !all(ratio .== ratio[1])
      return ParamAffinity()
    end
  end

  return ParamTimeAffinity()
end

function affinity_jacobian(
  op::ParamTransientFEOperator,
  solver::ODESolver,
  params::Table,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}};
  ntests::Int=10,i::Int=1)

  row,col = filter
  times = get_times(solver)
  test_row = get_test(op)[row]
  trial_col = get_trial(op)[col]
  dv_row = _get_fe_basis(op.test,row)
  du_col = _get_trial_fe_basis(get_trial(op)(nothing,nothing),col)
  u = allocate_evaluation_function(op)
  ucol = filter_evaluation_function(u,col)

  μ,t = first(params),first(times)
  d = collect_cell_contribution(trial_col(μ,t),test_row,op.jacs[i](μ,t,ucol,du_col,dv_row),trian)
  if all(isempty,d)
    return ZeroAffinity()
  end

  dir_cells = op.test.dirichlet_cells
  cell = find_nonzero_cell_contribution(d,dir_cells)
  d0 = max.(abs.(d[cell]),eps())

  for μ in rand(params,ntests)
    d = collect_cell_contribution(trial_col(μ,t),test_row,op.jacs[i](μ,t,ucol,du_col,dv_row),trian)
    ratio = d[cell] ./ d0
    if !all(ratio .== ratio[1])
      return NonAffinity()
    end
  end

  for t in rand(times,ntests)
    d = collect_cell_contribution(trial_col(μ,t),test_row,op.jacs[i](μ,t,ucol,du_col,dv_row),trian)
    ratio = d[cell] ./ d0
    if !all(ratio .== ratio[1])
      return ParamAffinity()
    end
  end

  return ParamTimeAffinity()
end

function find_nonzero_cell_contribution(data,dir_cells)
  global idx
  for celli in dir_cells
    if !iszero(sum(abs.(data[celli])))
      return celli
    end
  end
  @unreachable
end
