_get_nsnaps(args...) = 1

_get_nsnaps(::Union{TimeAffinity,NonAffinity},params::Table) = length(params)

for (Tsnp,Top,Tslv) in zip(
  (:Snapshots,:TransientSnapshots),
  (:ParamFEOperator,:ParamTransientFEOperator),
  (:FESolver,:ODESolver))

  @eval begin
    function collect_solutions(
      op::$Top,
      solver::$Tslv,
      params::Table)

      aff = NonAffinity()
      nsol = _get_nsnaps(aff,params)
      printstyled("Generating $nsol solution snapshots\n";color=:blue)

      s = allocate_solution(op.test,solver)
      sol = solutions(s,op,solver,params)
      $Tsnp(aff,sol,nsol)
    end

    function collect_residuals(
      op::$Top,
      solver::$Tslv,
      sols::$Tsnp,
      params::Table,
      args...)

      aff = affinity_residual(op,solver,params,args...)
      nres = _get_nsnaps(aff,params)

      printstyled("Generating $nres residual snapshots, affinity: $aff\n";color=:blue)

      rcache = allocate_residual(op)
      res_iter = init_res_iterator(op,solver,args...)
      res = residuals(aff,rcache,op,solver,res_iter,sols,params)
      $Tsnp(aff,res,nres)
    end

    function collect_jacobians(
      op::$Top,
      solver::$Tslv,
      sols::$Tsnp,
      params::Table,
      args...)

      aff = affinity_jacobian(op,solver,params,args...)
      njac = _get_nsnaps(aff,params)

      printstyled("Generating $njac jacobian snapshots, affinity: $aff\n";color=:blue)

      j = allocate_jacobian(op)
      jnnz = compress(j)
      jcache = (j,jnnz)
      jac_iter = init_jac_iterator(op,solver,args...)
      jac = jacobians(aff,jcache,op,solver,jac_iter,sols,params)
      $Tsnp(aff,jac,njac)
    end
  end
end

function allocate_solution(test::FESpace,::FESolver)
  space_ndofs = test.nfree
  cache = fill(0.,space_ndofs,1)
  cache
end

function allocate_solution(test::FESpace,solver::ODESolver)
  space_ndofs = test.nfree
  time_ndofs = get_time_ndofs(solver)
  cache = fill(0.,space_ndofs,time_ndofs)
  cache
end

function allocate_solution(test::MultiFieldFESpace,args...)
  map(t->solution_cache(t,args...),test.spaces)
end

for (Top,Tslv) in zip(
  (:ParamFEOperator,:ParamTransientFEOperator),
  (:FESolver,:ODESolver))

  @eval begin
    function solutions(
      cache,
      op::$Top,
      solver::$Tslv,
      params::Table)

      pmap(params) do μ
        solution!(cache,op,solver,μ)
      end
    end
  end
end

function solution!(
  cache,
  op::ParamFEOperator,
  solver::FESolver,
  μ::AbstractVector)

  sol = solve(op,solver,μ)
  if isa(cache,AbstractMatrix)
    copyto!(cache,sol.uh)
  elseif isa(cache,Vector{<:AbstractMatrix})
    map((c,sol) -> copyto!(c,sol),cache,sol.uh)
  else
    @unreachable
  end
  cache
end

function solution!(
  cache,
  op::ParamTransientFEOperator,
  solver::ODESolver,
  μ::AbstractVector)

  sol = solve(op,solver,μ,solver.uh0(μ))
  n = 1
  if isa(cache,AbstractMatrix)
    for soln in sol
      setindex!(cache,soln,:,n)
      n += 1
    end
  elseif isa(cache,Vector{<:AbstractMatrix})
    for soln in sol
      map((cache,sol) -> setindex!(cache,sol,:,n),cache,soln)
      n += 1
    end
  else
    @unreachable
  end
  cache
end

abstract type IterativeCollector end
abstract type TransientIterativeCollector end

mutable struct IterativeVecCollector <: IterativeCollector
  f::Function
  xh::AbstractVector
  μ::AbstractVector
  cache
end

mutable struct TransientIterativeVecCollector <: TransientIterativeCollector
  f::Function
  xh::Tuple{Vararg{AbstractVector}}
  μ::AbstractVector
  t::Float
  cache
end

mutable struct IterativeMatCollector <: IterativeCollector
  f::Function
  xh::AbstractVector
  μ::AbstractVector
  cache
end

mutable struct TransientIterativeMatCollector <: TransientIterativeCollector
  f::Function
  xh::Tuple{Vararg{AbstractVector}}
  μ::AbstractVector
  t::Float
  cache
end

function init_res_iterator(
  op::ParamFEOperator,
  ::FESolver,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  args...)

  row,_ = filter
  test_row = get_test(op)[row]
  dv_row = _get_fe_basis(op.test,row)
  assem_row = SparseMatrixAssembler(test_row,test_row)

  r = allocate_residual(op,cache)
  function f(
    r::AbstractVector,
    xh::AbstractVector,
    μ::AbstractVector,
    cache)

    u = evaluation_function(op,xh,cache)
    vecdata = collect_cell_vector(test_row,op.res(μ,u,dv_row,args...),trian)
    assemble_vector_add!(r,assem_row,vecdata)
    r
  end

  xh = get_free_dof_values(zero(op.test))
  μ = realization(op)
  cache = allocate_cache(op)

  IterativeVecCollector(f,xh,μ,cache)
end

function init_res_iterator(
  op::ParamTransientFEOperator,
  ::θMethod,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  args...)

  row,_ = filter
  test_row = get_test(op)[row]
  dv_row = _get_fe_basis(op.test,row)
  assem_row = SparseMatrixAssembler(test_row,test_row)

  function f(
    r::AbstractVector,
    xh::Tuple{Vararg{AbstractVector}},
    μ::AbstractVector,
    t::Real,
    cache)

    u = evaluation_function(op,xh,cache)
    vecdata = collect_cell_vector(test_row,op.res(μ,t,u,dv_row,args...),trian)
    assemble_vector_add!(r,assem_row,vecdata)
    r
  end

  xh0 = get_free_dof_values(zero(op.test))
  xh = (xh0,xh0)
  μ = realization(op)
  t = 0.
  cache = allocate_cache(op)

  TransientIterativeVecCollector(f,xh,μ,t,cache)
end

function init_jac_iterator(
  op::ParamFEOperator,
  ::FESolver,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  args...)

  row,col = filter
  test_row = op.test[row]
  _trial = get_trial(op)(nothing)
  _trial_col = _trial[col]
  dv_row = _get_fe_basis(op.test,row)
  du_col = _get_trial_fe_basis(_trial,col)
  assem_row_col = SparseMatrixAssembler(_trial_col,test_row)

  function f(
    j::SparseMatrixCSC,
    xh::AbstractVector,
    μ::AbstractVector,
    cache)

    u = evaluation_function(op,xh,cache)
    vecdata = collect_cell_vector(test_row,op.jac(μ,u,du_col,dv_row,args...),trian)
    assemble_matrix_add!(j,assem_row_col,vecdata)
    j
  end

  xh = get_free_dof_values(zero(op.test))
  μ = realization(op)
  cache = allocate_cache(op)

  IterativeVecCollector(f,xh,μ,cache)
end

function init_jac_iterator(
  op::ParamTransientFEOperator,
  solver::θMethod,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  args...)

  row,col = filter
  test_row = op.test[row]
  _trial = get_trial(op)(nothing,nothing)
  _trial_col = _trial[col]
  dv_row = _get_fe_basis(op.test,row)
  du_col = _get_trial_fe_basis(_trial,col)
  assem_row_col = SparseMatrixAssembler(_trial_col,test_row)
  γ = (1.0,1/(solver.dt*solver.θ))

  function f(
    j::SparseMatrixCSC,
    xh::Tuple{Vararg{AbstractVector}},
    μ::AbstractVector,
    t::Float,
    cache)

    trial, = cache[1]
    trial_col = trial[col]
    u = evaluation_function(op,xh,cache)
    u_col = filter_evaluation_function(u,col)
    _matdata = ()
    for (i,γᵢ) in enumerate(γ)
      if γᵢ > 0.0
        _matdata = (_matdata...,collect_cell_matrix(trial_col,test_row,
          γᵢ*op.jacs[i](μ,t,u_col,du_col,dv_row,args...),trian))
      end
    end
    matdata = _vcat_matdata(_matdata)
    assemble_matrix_add!(j,assem_row_col,matdata)
    j
  end

  xh0 = get_free_dof_values(zero(op.test))
  xh = (xh0,xh0)
  μ = realization(op)
  t = 0.
  cache = allocate_cache(op)
  TransientIterativeMatCollector(f,xh,μ,t,cache)
end

function update!(
  it::IterativeCollector,
  op::ParamFEOperator,
  xh::AbstractVector,
  μ::AbstractVector)

  it.xh = xh
  it.μ = μ
  cache = it.cache
  update_cache!(cache,op,μ)
  it.cache = cache
  return
end

function update!(
  it::TransientIterativeCollector,
  op::ParamTransientFEOperator,
  xh::Tuple{Vararg{AbstractVector}},
  μ::AbstractVector,
  t::Real)

  it.xh = xh
  it.μ = μ
  cache = it.cache
  update_cache!(cache,op,μ,t)
  it.cache = cache
  return
end

function evaluate!(
  rcache::AbstractVector,
  it::IterativeVecCollector,
  op::ParamFEOperator,
  xh::AbstractVector,
  μ::AbstractVector)

  update!(it,op,xh,μ)
  it.f(rcache,it.xh,it.μ,it.cache)
  rcache
end

function evaluate!(
  jcache::Tuple{SparseMatrixCSC,NnzArray},
  it::IterativeMatCollector,
  op::ParamFEOperator,
  xh::AbstractVector,
  μ::AbstractVector)

  jmat,jnnz = jcache
  update!(it,op,xh,μ)
  it.f(jmat,it.xh,it.μ,it.cache)
  nnz_i,nnz_j = compress_array(jmat)
  jnnz.nonzero_val = nnz_j
  jnnz.nonzero_idx = nnz_i
  jnnz
end

function evaluate!(
  rcache::AbstractVector,
  it::TransientIterativeVecCollector,
  op::ParamTransientFEOperator,
  xh::Tuple{Vararg{AbstractVector}},
  μ::AbstractVector,
  t::Real)

  update!(it,op,xh,μ,t)
  it.f(rcache,it.xh,it.μ,it.t,it.cache)
  rcache
end

function evaluate!(
  jcache::Tuple{SparseMatrixCSC,NnzArray},
  it::TransientIterativeMatCollector,
  op::ParamTransientFEOperator,
  xh::Tuple{Vararg{AbstractVector}},
  μ::AbstractVector,
  t::Real)

  jmat,jnnz = jcache
  update!(it,op,xh,μ,t)
  it.f(jmat,it.xh,it.μ,it.t,it.cache)
  nnz_i,nnz_j = compress_array(jmat)
  jnnz.nonzero_val = nnz_j
  jnnz.nonzero_idx = nnz_i
  jnnz
end

for (fun) in (:residuals,:jacobians)
  @eval begin
    function $fun(
      ::Union{ZeroAffinity,ParamAffinity},
      cache::Any,
      op::ParamFEOperator,
      ::FESolver,
      it::IterativeCollector,
      sols::Snapshots,
      params::Table)

      μ = first(params)
      xh = get_datum(sols[1])
      evaluate!(cache,it,op,xh,μ)
    end

    function $fun(
      ::NonAffinity,
      cache::Any,
      op::ParamFEOperator,
      ::FESolver,
      it::IterativeCollector,
      sols::Snapshots,
      params::Table)

      sols_μ = get_datum(sols)
      pmap(enumerate(params)) do (nμ,μ)
        xh = sols_μ[:,nμ]
        evaluate!(cache,it,op,xh,μ)
      end
    end

    function $fun(
      ::Union{ZeroAffinity,ParamTimeAffinity},
      cache::Any,
      op::ParamTransientFEOperator,
      solver::ODESolver,
      it::TransientIterativeCollector,
      sols::TransientSnapshots,
      params::Table)

      μ = first(params)
      t = first(get_times(solver))
      xh = get_datum(sols[1])[:,1]
      evaluate!(cache,it,op,xh,μ,t)
    end

    function $fun(
      ::ParamAffinity,
      cache::Any,
      op::ParamTransientFEOperator,
      solver::ODESolver,
      it::TransientIterativeCollector,
      sols::TransientSnapshots,
      params::Table)

      times = get_times(solver)
      μ = first(params)
      sols_t = get_datum(sols[1])
      pmap(enumerate(times)) do (nt,t)
        xh = sols_t[:,nt]
        evaluate!(cache,it,op,xh,μ,t)
      end
    end

    function $fun(
      ::TimeAffinity,
      cache::Any,
      op::ParamTransientFEOperator,
      solver::ODESolver,
      it::TransientIterativeCollector,
      sols::TransientSnapshots,
      params::Table)

      t = first(get_times(solver))
      sols_t = get_datum(sols[:,1])
      pmap(enumerate(params)) do (nμ,μ)
        xh = sols_t[:,nμ]
        evaluate!(cache,it,op,xh,μ,t)
      end
    end

    function $fun(
      ::NonAffinity,
      cache::Any,
      op::ParamTransientFEOperator,
      solver::ODESolver,
      it::TransientIterativeCollector,
      sols::TransientSnapshots,
      params::Table,)

      times = get_times(solver)
      tdofs = length(times)
      sols_μt = get_datum(sols)
      xh = get_free_dof_values(zero(op.test))
      xhθ = copy(xh)

      q = pmap(enumerate(params)) do (nμ,μ)
        sols_μ = hcat(xh,sols_μt[:,(nμ-1)*tdofs+1:nμ*tdofs])
        map(enumerate(times)) do (nt,t)
          _update_x!(solver,xhθ,sols_μ,nt)
          evaluate!(cache,it,op,(xh,xhθ),μ,t)
        end
      end
      pmap(x->map(hcat,x),q)
    end
  end
end

function _update_x!(
  solver::θMethod,
  xhθ::AbstractVector,
  sols_μ::AbstractMatrix,
  nt::Int)

  x = sols_μ[:,nt+1]
  xprev = sols_μ[:,nt]
  θ = solver.θ
  dtθ = solver.dt*θ
  copyto!(xhθ,(θ*x + (1-θ)*xprev)/dtθ)
  xhθ
end
