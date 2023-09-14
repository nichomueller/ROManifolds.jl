function get_iterative_quantities(
  aff::Affinity,
  op::PTFEOperator,
  solver::ODESolver,
  sols::AbstractArray,
  params::Table;
  intermediate_step=false)

  pop = get_algebraic_operator(op)
  odecache = allocate_cache(pop)
  _params = get_params(aff,params)
  _times = get_times(aff,solver)
  xh = Vector{TransientCellField}(length(_params)*length(_times))
  trials = Vector{TrialFESpace}(length(_params)*length(_times))
  x = zeros(op.test)
  x0,x1,vθ = copy(x),copy(x),copy(x)

  count = 0
  countμ = 0
  @inbounds for μ = _params
    countμ += 1
    countt = 0
    @inbounds for t = _times
      count += 1
      countt += 1
      if t == first(_times)
        x0 = setup_initial_condition(solver,μ)
      else
        @. x0 = sols[countμ][countt-1]
      end
      if intermediate_step
        @. vθ = (x1-x0)/(solver.θ*solver.dt)
      end
      @. x1 = sols[countμ][countt]
      @. x = x0*solver.θ + x1*(1-solver.θ)
      xh[count] = evaluation_function(op,(x,vθ),odecache)
      update_cache!(odecache,pop,μ,t)
      trials[count] = first(odecache)
    end
  end
  inputs = _reorder_xtp(x,_params,_times)
  return trials,inputs
end

function _reorder_iterative_quantities(trials,x,p,t)
  _create_tuple(a,b) = map((ai,bi)->(ai,bi),a,b)
  pt = reshape(Iterators.product(p,t) |> collect,:)
  _create_tuple(_create_tuple(trials,x)...,pt...)
end

function collect_residuals(
  solver::ThetaMethod,
  op::PTFEOperator,
  sols::AbstractArray,
  params::Table,
  trian::Triangulation,
  args...)

  dv = get_fe_basis(op.test)
  aff = affinity_residual(feop,fesolver,trian)
  _,inputs = get_iterative_quantities(aff,feop,solver,sols,params)
  nin = length(inputs)
  dcs = evaluate(op.res(dv,args...),inputs)
  vecdata = collect_cell_vector(op.test,dcs,trian)
  b = allocate_residual(op)
  bs = fill(b,nin)
  assemble_vector_add!(bs,op.assem,vecdata)

  return aff,bs
end

function collect_jacobians(
  solver::ThetaMethod,
  op::PTFEOperator,
  sols::AbstractArray,
  params::Table,
  trian::Triangulation,
  args...;
  i=1)

  trial = get_trial(op)
  trial_hom = allocate_trial_space(trial)
  dv = get_fe_basis(op.test)
  du = get_trial_fe_basis(trial_hom)
  aff = affinity_jacobian(op,solver,trian)
  trials,inputs = get_iterative_quantities(aff,feop,solver,sols,params;intermediate_step=true)
  nin = length(inputs)
  γᵢ = (1.0,1/(solver.dt*solver.θ))[i]

  dcs = evaluate(op.jacs[i](du,dv,args...),inputs)
  matdata = collect_cell_matrix(trials,op.test,dcs,trian)
  b = allocate_nnz_vector(op)
  bs = fill(b,nin)
  assemble_matrix_add!(bs,op.assem,matdata)

  return aff,bs*γᵢ
end

function setup_initial_condition(
  solver::ODESolver,
  μ::AbstractArray)

  uh0 = solver.uh0(μ)
  get_free_dof_values(uh0)
end

function evaluation_function(
  op::PTFEOperator,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  Xh, = ode_cache
  dxh = ()
  for i in 2:get_order(op)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
  end
  TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
end

function compress_function(
  f::Function,
  solver::ThetaMethod,
  trian::Triangulation,
  params::Table;
  kwargs...)

  times = get_times(solver)
  p_t = Iterators.product(times,params) |> collect

  phys_map = get_cell_map(trian)
  cell_points = get_cell_points(trian) |> get_data
  quad_points = lazy_map(evaluate,phys_map,cell_points)
  cell_fields = lazy_map(pt -> CellField(x->f(x,pt...),trian,PhysicalDomain()),p_t)

  fevals = lazy_map(cell_fields) do field
    feval_x = map(evaluate,get_data(field),quad_points)
    reduce(vcat,feval_x)
  end

  fcat = collect(fevals)
  fpod = tpod(fcat;kwargs...)
  fblocks = eachcol(reshape(fpod,:,length(cell_points)))
  GenericCellField(fblocks,trian,PhysicalDomain())
end
