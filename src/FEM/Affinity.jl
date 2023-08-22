abstract type Affinity end
struct ZeroAffinity <: Affinity end
struct ParamAffinity <: Affinity end
struct TimeAffinity <: Affinity end
struct ParamTimeAffinity <: Affinity end
struct NonAffinity <: Affinity end

function affinity_residual(
  op::ParamTransientFEOperator,
  solver::ODESolver,
  trian::Triangulation;
  ntests::Int=10)

  dv = get_fe_basis(op.test)
  u = allocate_evaluation_function(op)

  test_params = realization(op,ntests)
  test_times = rand(get_times(solver),ntests)

  μ,t = rand(test_params),rand(test_times)
  d = collect_cell_contribution(op.test,op.res(μ,t,u,dv),trian)
  if all(isempty,d)
    return ZeroAffinity()
  end

  dir_cells = op.test.dirichlet_cells
  cell = find_nonzero_cell_contribution(d,dir_cells)
  d0 = max.(abs.(d[cell]),eps())

  for μ in test_params
    d = collect_cell_contribution(op.test,op.res(μ,t,u,dv),trian)
    ratio = d[cell] ./ d0
    if !all(ratio .== ratio[1])
      return NonAffinity()
    end
  end

  for t in test_times
    d = collect_cell_contribution(op.test,op.res(μ,t,u,dv),trian)
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
  trian::Triangulation;
  ntests::Int=10,i::Int=1)

  trial_hom = allocate_trial_space(op.trial)
  du = get_trial_fe_basis(trial_hom)
  dv = get_fe_basis(op.test)
  u = allocate_evaluation_function(op)

  test_params = realization(op,ntests)
  test_times = rand(get_times(solver),ntests)

  μ,t = rand(test_params),rand(test_times)
  d = collect_cell_contribution(op.trial(μ,t),op.test,op.jacs[i](μ,t,u,du,dv),trian)
  if all(isempty,d)
    return ZeroAffinity()
  end

  dir_cells = op.test.dirichlet_cells
  cell = find_nonzero_cell_contribution(d,dir_cells)
  d0 = max.(abs.(d[cell]),eps())

  for μ in test_params
    d = collect_cell_contribution(op.trial(μ,t),op.test,op.jacs[i](μ,t,u,du,dv),trian)
    ratio = d[cell] ./ d0
    if !all(ratio .== ratio[1])
      return NonAffinity()
    end
  end

  for t in test_times
    d = collect_cell_contribution(op.trial(μ,t),op.test,op.jacs[i](μ,t,u,du,dv),trian)
    ratio = d[cell] ./ d0
    if !all(ratio .== ratio[1])
      return ParamAffinity()
    end
  end

  return ParamTimeAffinity()
end

function allocate_evaluation_function(op::ParamTransientFEOperator)
  μ,t = realization(op),1.
  trial = get_trial(op)(μ,t)
  x = fill(0.,num_free_dofs(op.test))
  xh = EvaluationFunction(trial,x)
  dxh = ()
  for _ in 1:get_order(op)
    dxh = (dxh...,xh)
  end
  TransientCellField(xh,dxh)
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

get_times(::Affinity,solver) = solver.t0
get_times(::Union{TimeAffinity,ParamTimeAffinity},solver) = get_times(solver)
