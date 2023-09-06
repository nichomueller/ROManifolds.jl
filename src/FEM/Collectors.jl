abstract type CollectorMap{A} <: Map end

struct CollectSolutionsMap <: CollectorMap{NonAffinity}
  f::Function

  function CollectSolutionsMap(
    solver::ODESolver,
    op::ParamTransientFEOperator)

    t0,tF = solver.t0,solver.tF
    uh0 = solver.uh0
    f = μ::AbstractArray -> solve(solver,op,μ,uh0(μ),t0,tF)
    new(f)
  end
end

function Arrays.evaluate!(cache,k::CollectSolutionsMap,μ::AbstractArray)
  sol_μ = k.f(μ)
  l = length(sol_μ)
  T = return_type(sol_μ)
  v = Vector{T}(undef,l)
  @fastmath @inbounds for (n,sol_μn) in enumerate(sol_μ)
    uhn,_ = sol_μn
    v[n] = uhn
  end
  sol_μ_nnz = compress(v)
  sol_μ_nnz
end

function collect_inputs(
  ::ParamTimeAffinity,
  feop::ParamTransientFEOperator,
  solver::ODESolver,
  sols::AbstractArray,
  params::Table)

  pop = get_algebraic_operator(op)
  odecache = allocate_cache(pop)
  times = get_times(solver)
  trials,x = _allocate_xtrial(sols,params,times)

  count = 0
  countμ = 0
  @inbounds for μ = params
    countμ += 1
    countt = 0
    @inbounds for t = times
      count += 1
      countt += 1
      if t == first(times)
        x0 = setup_initial_condition(solver,μ)
        x1 = sols[countμ][1]
      else
        x0 = sols[countμ][countt-1]
        x1 = sols[countμ][countt]
      end
      x[count] = x0*solver.θ + x1*(1-solver.θ)
      update_cache!(odecache,μ,t)
      trials[count] = odecache
    end
  end
  trials,_reorder_xtp(x,params,times)
end

function collect_inputs(
  ::ParamAffinity,
  solver::θMethod,
  sols::AbstractArray,
  params::Table)

  times = get_times(solver)
  μ = testitem(params)
  x0 = setup_initial_condition(solver,μ)
  x = testitem(sols)
  x1 = pushfirst!(x[2:end],x0)
  x1θ = x1*solver.θ + x1*(1-solver.θ)
  _reorder_xtp(x1θ,μ,times)
end

function collect_inputs(
  ::ParamTimeAffinity,
  solver::ODESolver,
  sols::AbstractArray,
  params::Table)

  return map(testitem,(sols,params,get_times(solver)))
end

function _allocate_xtrial(x,p,t)
  npt = length(p)*length(t)
  T = eltype(eltype(x))
  x = Vector{T}(undef,npt)
  Tode = typeof(odecache)
  trials = Vector{Tode}(undef,npt)
  trials,x
end

function _reorder_xtp(x,p,t)
  pt = Iterators.product(p,t) |> collect
  xpt = map((a,b)->(a,b...),x,pt)
  return xpt
end

struct CollectResidualsMap{A} <: CollectorMap{A}
  f::Function

  function CollectResidualsMap(
    solver::θMethod,
    op::ParamTransientFEOperator,
    trian::Triangulation=get_triangulation(get_test(op)),
    args...)

    dv = get_fe_basis(op.test)
    v0 = zeros(op.test)
    pop = get_algebraic_operator(op)
    A = affinity_residual(op,solver,trian)
    times = get_times(A,solver)

    function f(μ::AbstractArray)
      xh = allocate_evaluation_function(op)
      v,rcache,odecache, = allocate_residual_cache(pop,times)
      @fastmath @inbounds for (n,t) in enumerate(times)
        r = copy(rcache)
        update_cache!(odecache,pop,μ,t)
        vecdata = collect_cell_vector(op.test,op.res(μ,t,xh,dv,args...),trian)
        assemble_vector_add!(r,op.assem,vecdata)
        @. r *= -1.0
        v[n] = r
      end
      v
    end

    function f(sol::AbstractArray,μ::AbstractArray)
      x0 = setup_initial_condition(solver,μ)
      v,rcache,odecache,x = allocate_residual_cache(pop,times)
      @fastmath @inbounds for (n,t) in enumerate(times)
        r = copy(rcache)
        update_cache!(odecache,pop,μ,t)
        @. x = sol[:,n]*solver.θ + x0*(1-solver.θ)
        xh = evaluation_function(op,(x,v0),odecache)
        @. x0 = sol[:,n]
        vecdata = collect_cell_vector(op.test,op.res(μ,t,xh,dv,args...),trian)
        assemble_vector_add!(r,op.assem,vecdata)
        @. r *= -1.0
        v[n] = r
      end
      v
    end

    new{A}(f)
  end
end

function Arrays.evaluate!(cache,k::CollectResidualsMap,args...)
  res_μ = k.f(args...)
  res_μ_nnz = compress(res_μ)
  res_μ_nnz
end

struct CollectJacobiansMap{A} <: CollectorMap{A}
  f::Function

  function CollectJacobiansMap(
    solver::θMethod,
    op::ParamTransientFEOperator,
    trian::Triangulation=get_triangulation(get_test(op)),
    args...;
    i::Int=1)

    trial = get_trial(op)
    trial_hom = allocate_trial_space(trial)
    dv = get_fe_basis(op.test)
    du = get_trial_fe_basis(trial_hom)
    pop = get_algebraic_operator(op)
    A = affinity_jacobian(op,solver,trian)
    times = get_times(A,solver)
    γ = (1.0,1/(solver.dt*solver.θ))

    function f(μ::AbstractArray)
      xh = allocate_evaluation_function(op)
      v,Jcache,odecache, = allocate_jacobian_cache(pop,times)
      @fastmath @inbounds for (n,t) in enumerate(times)
        J = copy(Jcache)
        update_cache!(odecache,pop,μ,t)
        trial, = odecache[1]
        matdata = collect_cell_matrix(trial,op.test,γ[i]*op.jacs[i](μ,t,xh,du,dv,args...),trian)
        assemble_matrix_add!(J,op.assem,matdata)
        v[n] = compress(J)
      end
      v
    end

    function f(sol::AbstractArray,μ::AbstractArray)
      x0 = setup_initial_condition(solver,μ)
      v,Jcache,odecache,x,vθ = allocate_jacobian_cache(pop,times)
      @fastmath @inbounds for (n,t) in enumerate(times)
        J = copy(Jcache)
        update_cache!(odecache,pop,μ,t)
        trial, = odecache[1]
        @. x = sol[:,n]*solver.θ + x0*(1-solver.θ)
        @. vθ = (x-x0)/(solver.θ*solver.dt)
        xh = evaluation_function(op,(x,vθ),odecache)
        @. x0 = sol[:,n]
        matdata = collect_cell_matrix(trial,op.test,γ[i]*op.jacs[i](μ,t,xh,du,dv,args...),trian)
        assemble_matrix_add!(J,op.assem,matdata)
        v[n] = compress(J)
      end
      v
    end

    new{A}(f)
  end
end

function Arrays.evaluate!(cache,k::CollectJacobiansMap,args...)
  jac_μ_nnz = k.f(args...)
  hcat(jac_μ_nnz...)
end

function setup_initial_condition(
  solver::ODESolver,
  μ::AbstractArray)

  uh0 = solver.uh0(μ)
  get_free_dof_values(uh0)
end

function evaluation_function(
  op::ParamTransientFEOperator,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  Xh, = ode_cache
  dxh = ()
  for i in 2:get_order(op)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
  end
  TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
end

function allocate_residual_cache(
  pop::ParamODEOperator,
  times::Vector{<:Real})

  x = zeros(pop.feop.test)
  odecache = allocate_cache(pop)
  rcache = allocate_residual(pop,x,odecache)
  T = typeof(rcache)
  l = length(times)
  v = Vector{T}(undef,l)

  return v,rcache,odecache,x
end

function allocate_jacobian_cache(
  pop::ParamODEOperator,
  times::Vector{<:Real})

  x = zeros(pop.feop.test)
  vθ = copy(x)
  odecache = allocate_cache(pop)
  Jcache = allocate_jacobian(pop,x,odecache)
  J_nnz = compress(Jcache)
  T = typeof(J_nnz)
  l = length(times)
  v = Vector{T}(undef,l)

  return v,Jcache,odecache,x,vθ
end

function compress_function(
  f::Function,
  solver::θMethod,
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
