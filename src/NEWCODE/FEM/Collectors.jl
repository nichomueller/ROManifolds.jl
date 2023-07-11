_get_nsnaps(args...) = 1

_get_nsnaps(::Union{TimeAffinity,NonAffinity},params::Table) = length(params)

for (Top,Tslv) in zip(
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
      solutions!(s,op,solver,params)
      s
    end

    function collect_residuals(
      op::$Top,
      solver::$Tslv,
      sols::Tsnp,
      params::Table,
      args...) where Tsnp

      cache = allocate_cache(op)
      aff = affinity_residual(op,solver,params,args...)
      nres = _get_nsnaps(aff,params)

      printstyled("Generating $nres residual snapshots, affinity: $aff\n";color=:blue)

      r = allocate_residual(op,cache)
      rcache = r,cache
      residuals!(aff,op,solver,sols,params,rcache,args...)
      Snapshots(aff,r,nres)
    end

    function collect_jacobians(
      op::$Top,
      solver::$Tslv,
      sols::Tsnp,
      params::Table,
      args...) where Tsnp

      cache = allocate_cache(op)
      aff = affinity_jacobian(op,solver,params,args...)
      njac = _get_nsnaps(aff,params)

      printstyled("Generating $njac jacobian snapshots, affinity: $aff\n";color=:blue)

      j = allocate_jacobian(op,cache)
      jnnz = compress(j)
      jcache = (j,jnnz),cache
      jacobians!(aff,op,solver,sols,params,jcache,args...)
      Snapshots(aff,j,njac)
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
    function solutions!(
      cache,
      op::$Top,
      solver::$Tslv,
      params::Table)

      pmap(params) do μ
        solution!(cache,op,solver,μ)
      end
      cache
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

for (fun,cfun) in zip((:residuals!,:jacobians!),(:residual!,:nz_jacobian!))
  @eval begin
    function $fun(
      ::Union{ZeroAffinity,ParamAffinity},
      op::ParamFEOperator,
      ::FESolver,
      sols::Tsnp,
      params::Table,
      cache,
      args...) where Tsnp

      μ = first(params)
      xh = get_datum(sols[1])
      a,c = cache
      $cfun(a,op,solver,xh,μ,c,args...)
      a
    end

    function $fun(
      ::NonAffinity,
      op::ParamFEOperator,
      ::FESolver,
      sols::Tsnp,
      params::Table,
      cache,
      args...) where Tsnp

      a,c = cache
      pmap(enumerate(params)) do (nμ,μ)
        xh = get_datum(sols[nμ])
        update_cache!(c,op,μ)
        $cfun(a,op,solver,xh,μ,c,args...)
      end
      a
    end

    function $fun(
      ::Union{ZeroAffinity,ParamTimeAffinity},
      op::ParamTransientFEOperator,
      solver::ODESolver,
      sols::Tsnp,
      params::Table,
      cache,
      args...) where Tsnp

      μ = first(params)
      t = first(get_times(solver))
      xh = get_datum(sols[1,1])
      a,c = cache
      $cfun(a,op,solver,xh,μ,t,c,args...)
      a
    end

    function $fun(
      ::ParamAffinity,
      op::ParamTransientFEOperator,
      solver::ODESolver,
      sols::Tsnp,
      params::Table,
      cache,
      args...) where Tsnp

      times = get_times(solver)
      μ = first(params)
      a,c = cache
      pmap(enumerate(times)) do (nt,t)
        xh = get_datum(sols[1,nt])
        update_cache!(cache,op,μ,t)
        $cfun(a,op,solver,xh,μ,t,cache,args...)
      end
      a
    end

    function $fun(
      ::TimeAffinity,
      op::ParamTransientFEOperator,
      solver::ODESolver,
      sols::Tsnp,
      params::Table,
      cache,
      args...) where Tsnp

      t = first(get_times(solver))
      a,c = cache
      pmap(enumerate(params)) do (nμ,μ)
        xh = get_datum(sols[nμ,1])
        update_cache!(cache,op,μ,t)
        $cfun(a,op,solver,xh,μ,t,cache,args...)
      end
      a
    end

    function $fun(
      ::NonAffinity,
      op::ParamTransientFEOperator,
      solver::ODESolver,
      sols::Tsnp,
      params::Table,
      cache,
      args...) where Tsnp

      times = get_times(solver)
      a,c = cache
      pmap(enumerate(params)) do (nμ,μ)
        pmap(enumerate(times)) do (nt,t)
          xh = get_datum(sols[nμ,nt])
          update_cache!(cache,op,μ,t)
          $cfun(a,op,solver,xh,μ,t,cache,args...)
        end
      end
      a
    end
  end
end

function residual!(
  r::AbstractVector,
  op::ParamFEOperator,
  ::FESolver,
  xh::AbstractArray,
  μ::AbstractArray,
  cache,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  args...)

  row,_ = filter
  test_row = op.test[row]
  dv_row = _get_fe_basis(op.test,row)
  u = evaluation_function(op,xh,cache)
  assem_row = SparseMatrixAssembler(test_row,test_row)
  vecdata = collect_cell_vector(test_row,op.res(μ,u,dv_row,args...),trian)
  assemble_vector_add!(r,assem_row,vecdata)
end

function residual!(
  r::AbstractVector,
  op::ParamTransientFEOperator,
  ::θMethod,
  xh::AbstractArray,
  μ::AbstractArray,
  t::Float,
  cache,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  args...)

  row,_ = filter
  test_row = op.test[row]
  dv_row = _get_fe_basis(op.test,row)
  u = evaluation_function(op,xh,cache)
  assem_row = SparseMatrixAssembler(test_row,test_row)
  vecdata = collect_cell_vector(test_row,op.res(μ,t,u,dv_row,args...),trian)
  assemble_vector_add!(r,assem_row,vecdata)
end

function nz_jacobian!(j::Tuple{SparseMatrixCSC,NnzArray},args...)
  js,jnz = j
  jacobian!(js,args...)
  jnz.nonzero_val = compress(js)
  jnz
end

function jacobian!(
  j::AbstractMatrix,
  op::ParamFEOperator,
  ::FESolver,
  xh::AbstractArray,
  μ::AbstractArray,
  cache,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  args...)

  row,col = filter
  test_row = op.test[row]
  trial,_ = cache
  trial_col = trial[col]
  dv_row = _get_fe_basis(op.test,row)
  du_col = _get_trial_fe_basis(trial,col)
  u = evaluation_function(op,xh,cache)
  u_col = filter_evaluation_function(u,col)
  assem_row_col = SparseMatrixAssembler(trial_col,test_row)
  matdata = collect_cell_matrix(trial_col,test_row,
    op.jacs(μ,u_col,du_col,dv_row,args...),trian)

  assemble_matrix_add!(j,assem_row_col,matdata)
end

function jacobian!(
  j::AbstractMatrix,
  op::ParamTransientFEOperator,
  solver::θMethod,
  xh::AbstractArray,
  μ::AbstractArray,
  t::Float,
  cache,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  args...)

  row,col = filter
  test_row = op.test[row]
  trial,_ = cache
  trial_col = trial[col]
  dv_row = _get_fe_basis(op.test,row)
  du_col = _get_trial_fe_basis(trial,col)
  u = evaluation_function(op,xh,cache)
  u_col = filter_evaluation_function(u,col)
  assem_row_col = SparseMatrixAssembler(trial_col,test_row)

  γ = (1.0,1/(solver.dt*solver.θ))
  _matdata = ()
  for (i,γᵢ) in enumerate(γ)
    if γᵢ > 0.0
      _matdata = (_matdata...,collect_cell_matrix(trial_col,test_row,
        γᵢ*op.jacs[i](μ,t,u_col,du_col,dv_row,args...),trian))
    end
  end
  matdata = _vcat_matdata(_matdata)

  assemble_matrix_add!(j,assem_row_col,matdata)
end
