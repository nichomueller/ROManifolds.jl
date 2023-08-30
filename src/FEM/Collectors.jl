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
  for (n,sol_μn) in enumerate(sol_μ)
    uhn,_ = sol_μn
    v[n] = uhn
  end
  sol_μ_nnz = compress(v)
  sol_μ_nnz
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
    cache = allocate_cache(pop)
    A = affinity_residual(op,solver,trian)
    times = get_times(A,solver)

    function f(μ::AbstractArray)
      x0 = setup_initial_condition(solver,μ)
      xh = allocate_evaluation_function(op)

      map(times) do t
        update_cache!(cache,pop,μ,t)

        vecdata = collect_cell_vector(op.test,op.res(μ,t,xh,dv,args...),trian)
        r = allocate_residual(pop,x0,cache)
        assemble_vector_add!(r,op.assem,vecdata)
        r .*= -1.0
      end
    end

    function f(sol::AbstractArray,μ::AbstractArray)
      x0 = setup_initial_condition(solver,μ)

      map(enumerate(times)) do (it,t)
        update_cache!(cache,pop,μ,t)

        x = sol[:,it]*solver.θ + x0*(1-solver.θ)
        xh = evaluation_function(op,(x,v0),cache)
        x0 .= sol[:,it]

        vecdata = collect_cell_vector(op.test,op.res(μ,t,xh,dv,args...),trian)
        r = allocate_residual(pop,x0,cache)
        assemble_vector_add!(r,op.assem,vecdata)
        r .*= -1.0
      end
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
    cache = allocate_cache(pop)
    A = affinity_jacobian(op,solver,trian)
    times = get_times(A,solver)
    γ = (1.0,1/(solver.dt*solver.θ))

    function f(μ::AbstractArray)
      x0 = setup_initial_condition(solver,μ)
      xh = allocate_evaluation_function(op)

      map(times) do t
        update_cache!(cache,pop,μ,t)
        trial, = cache[1]

        matdata = collect_cell_matrix(trial,op.test,γ[i]*op.jacs[i](μ,t,xh,du,dv,args...),trian)
        J = allocate_jacobian(pop,x0,cache)
        assemble_matrix_add!(J,op.assem,matdata)
        compress(J)
      end
    end

    function f(sol::AbstractArray,μ::AbstractArray)
      x0 = setup_initial_condition(solver,μ)

      map(enumerate(times)) do (it,t)
        update_cache!(cache,pop,μ,t)
        trial, = cache[1]

        x = sol[it]*solver.θ + x0*(1-solver.θ)
        vθ = (x-x0)/(solver.θ*solver.dt)
        xh = evaluation_function(op,(x,vθ),cache)
        x0 .= sol[it]

        matdata = collect_cell_matrix(trial,op.test,γ[i]*op.jacs[i](μ,t,xh,du,dv,args...),trian)
        J = allocate_jacobian(pop,x0,cache)
        assemble_matrix_add!(J,op.assem,matdata)
        compress(J)
      end
    end

    new{A}(f)
  end
end

function Arrays.evaluate!(cache,k::CollectJacobiansMap,args...)
  jac_μ_nnz = k.f(args...)
  reduce(hcat,jac_μ_nnz)
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

  fevals = map(cell_fields) do field
    feval_x = map(evaluate,get_data(field),quad_points)
    reduce(vcat,feval_x)
  end

  fcat = reduce(hcat,fevals)
  fpod = tpod(fcat;kwargs...)
  fblocks = eachcol(reshape(fpod,:,length(cell_points)))
  GenericCellField(fblocks,trian,PhysicalDomain())
end
