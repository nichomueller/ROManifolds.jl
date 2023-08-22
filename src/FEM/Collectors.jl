struct CollectSolutionMap <: Map
  f::Function

  function CollectSolutionMap(
    solver::ODESolver,
    op::ParamTransientFEOperator)

    t0,tF = solver.t0,solver.tF
    uh0 = solver.uh0
    f = μ::AbstractArray -> ParamTransientFESolution(solver,op,μ,uh0(μ),t0,tF)
    new(f)
  end
end

function Arrays.evaluate!(cache,k::CollectSolutionMap,μ::AbstractArray)
  sol_μ = k.f(μ)
  l = length(sol_μ)
  T = return_type(sol_μ)
  vT = Vector{T}(undef,l)
  for (n,sol_μn) in enumerate(sol_μ)
    uhn,_ = sol_μn
    vT[n] = uhn
  end
  vT
end

struct CollectResidualsMap{A} <: Map
  f::Function

  function CollectResidualsMap(
    solver::θMethod,
    op::ParamTransientFEOperator,
    aff::A,
    trian::Triangulation) where A

    dv = get_fe_basis(op.test)
    v0 = zero(op.test)
    pop = get_algebraic_operator(op)
    cache = allocate_cache(pop)
    times = get_times(aff,solver)

    function f(sol::AbstractArray,μ::AbstractArray)
      x0 = setup_initial_condition(pop,solver,μ)

      map(enumerate(times)) do (it,t)
        update_cache!(cache,pop,μ,t)

        x = sol[it]*solver.θ + x0*(1-solver.θ)
        xh = evaluation_function(op,(x,v0),cache)
        x0 .= sol[it]

        vecdata = collect_cell_vector(op.test,op.res(μ,t,xh,dv),trian)
        r = allocate_residual(pop,x0,cache)
        assemble_vector_add!(r,op.assem,vecdata)
        r .*= -1.0
      end
    end

    new{A}(res_μ)
  end
end

function Arrays.evaluate!(cache,k::CollectResidualsMap,pp::ParamPair)
  sol,μ = get_snap(pp),get_param(pp)
  res_μ = k.f(sol,μ)
  res_μ_nnz = compress(res_μ)
  res_μ_nnz
end

struct CollectJacobiansMap <: Map
  f::Function

  function CollectJacobiansMap(
    solver::θMethod,
    op::ParamTransientFEOperator,
    trian::Triangulation;
    i::Int=1)

    trial = get_trial(op)
    trial_hom = allocate_trial_space(trial)
    dv = get_fe_basis(op.test)
    du = get_trial_fe_basis(trial_hom)
    pop = get_algebraic_operator(op)
    cache = allocate_cache(pop)
    times = get_times(aff,solver)
    γ = (1.0,1/(solver.dt*solver.θ))

    function f(sol::AbstractArray,μ::AbstractArray)
      x0 = setup_initial_condition(pop,solver,μ)

      map(enumerate(times)) do (it,t)
        update_cache!(cache,pop,μ,t)
        trial, = cache[1]

        x = sol[it]*solver.θ + x0*(1-solver.θ)
        vθ = (x-x0)/(solver.θ*solver.dt)
        xh = evaluation_function(op,(x,vθ),cache)
        x0 .= sol[it]

        matdata = collect_cell_matrix(trial,op.test,γ[i]*op.jacs[i](μ,t,xh,du,dv),trian)

        J = allocate_jacobian(pop,x0,cache)
        assemble_matrix_add!(J,op.assem,matdata)
        compress(J)
      end
    end

    new(f)
  end
end

function Arrays.evaluate!(cache,k::CollectJacobiansMap,pp::ParamPair)
  sol,μ = get_snap(pp),get_param(pp)
  jac_μ_nnz = k.f(sol,μ)
  jac_μ_nnz
end

function setup_initial_condition(
  op::ParamTransientFEOperator,
  solver::ODESolver,
  μ::AbstractArray=realization(op))

  solver.uh0(μ)
end

function setup_initial_condition(
  pop::ParamODEOpFromFEOp,
  solver::ODESolver,
  μ::AbstractArray=realization(pop.feop))

  uh0 = setup_initial_condition(pop.feop,solver,μ)
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

# WITH FILTERED STUFF
# struct CollectResidualsMap <: Map
#   f::Function

#   function CollectResidualsMap(
#     ::ODESolver,
#     op::ParamTransientFEOperator{Affine},
#     trian::Triangulation,
#     filter::Tuple{Vararg{Int}})

#     row,_ = filter
#     test_row = get_test(op)[row]
#     dv_row = _get_fe_basis(op.test,row)
#     assem_row = SparseMatrixAssembler(test_row,test_row)
#     pop = get_algebraic_operator(op)

#     function f(μ,t,cache)
#       update_cache!(cache,pop,μ,t)
#       x0 = setup_initial_condition(pop,solver,μ)
#       xh = evaluation_function(op,(x0,x0),cache)
#       vecdata = collect_cell_vector(op.test,op.res(μ,t,xh,dv_row),trian)

#       r = allocate_residual(pop,x0,cache)
#       assemble_vector_add!(r,assem_row,vecdata)
#       r .*= -1.0
#     end

#     new(res_μ)
#   end

#   function CollectResidualsMap(
#     ::ODESolver,
#     op::ParamTransientFEOperator,
#     trian::Triangulation,
#     filter::Tuple{Vararg{Int}})

#     row,_ = filter
#     test_row = get_test(op)[row]
#     dv_row = _get_fe_basis(op.test,row)
#     assem_row = SparseMatrixAssembler(test_row,test_row)
#     pop = get_algebraic_operator(op)

#     function f(μ,t,xh,cache)
#       update_cache!(cache,pop,μ,t)
#       x0 = setup_initial_condition(pop,solver,μ)
#       vecdata = collect_cell_vector(op.test,op.res(μ,t,xh,dv_row),trian)

#       r = allocate_residual(pop,x0,cache)
#       assemble_vector_add!(r,assem_row,vecdata)
#       r .*= -1.0
#     end

#     new(res_μ)
#   end
# end
