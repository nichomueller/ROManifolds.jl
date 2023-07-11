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

      s = allocate_solution(op.test,fesolver)
      solutions!(s,op,solver,μ)
      s
    end

    function collect_residuals(
      op::$Top,
      solver::$Tslv,
      sols::AbstractArray,
      params::Table;
      kwargs...)

      cache = allocate_cache(op)
      aff = affinity_residual(op,solver,params;kwargs...)
      nres = _get_nsnaps(aff,params)

      printstyled("Generating $nres residual snapshots, affinity: $aff\n";color=:blue)

      r = allocate_residual(op,cache)
      rcache = r,cache
      residuals!(aff,op,solver,sols,params,rcache)
      Snapshots(aff,r,nres)
    end

    function collect_jacobians(
      op::$Top,
      solver::$Tslv,
      sols::AbstractArray,
      params::Table;
      kwargs...)

      cache = allocate_cache(op)
      aff = affinity_jacobian(op,solver,params;kwargs...)
      njac = _get_nsnaps(aff,params)

      printstyled("Generating $njac jacobian snapshots, affinity: $aff\n";color=:blue)

      j = allocate_jacobian(op,cache)
      jnnz = compress(j)
      jcache = (j,jnnz),cache
      jacobians!(aff,op,solver,sols,params,jcache)
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

function solutions!(
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

function solutions!(
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

for (fun,cfun) in zip((:residuals!,:jacobians!),(:residual!,:nz_jacobian!))
  @eval begin
    function $fun(
      ::Union{ZeroAffinity,ParamAffinity},
      op::ParamFEOperator,
      ::FESolver,
      sols::AbstractArray,
      params::Table,
      cache)

      μ = first(params)
      xh = get_sol(sols,nμ)
      a,c = cache
      $cfun(a,op,μ,t,xh,c)
      a
    end

    function $fun(
      ::NonAffinity,
      op::ParamFEOperator,
      ::FESolver,
      sols::AbstractArray,
      params::Table,
      cache)

      a,c = cache
      pmap(enumerate(params)) do (nμ,μ)
        xh = get_sol(sols,nμ)
        update_cache!(c,op,μ)
        $cfun(a,op,μ,t,xh,c)
      end
      a
    end

    function $fun(
      ::Union{ZeroAffinity,ParamTimeAffinity},
      op::ParamTransientFEOperator,
      solver::ODESolver,
      sols::AbstractArray,
      params::Table,
      cache)

      μ = first(params)
      t = first(get_times(solver))
      xh = get_sol(sols,nμ)
      a,c = cache
      $cfun(a,op,μ,t,xh,c)
      a
    end

    function $fun(
      ::ParamAffinity,
      op::ParamTransientFEOperator,
      solver::ODESolver,
      sols::AbstractArray,
      params::Table,
      cache)

      times = get_times(solver)
      μ = first(params)
      a,c = cache
      pmap(enumerate(times)) do (nt,t)
        xh = get_sol(sols,nμ,nt)
        update_cache!(cache,op,μ,t)
        $cfun(a,op,μ,t,xh,cache)
      end
      a
    end

    function $fun(
      ::TimeAffinity,
      op::ParamTransientFEOperator,
      solver::ODESolver,
      sols::AbstractArray,
      params::Table,
      cache)

      t = first(get_times(solver))
      a,c = cache
      pmap(enumerate(params)) do (nμ,μ)
        xh = get_sol(sols,nμ,nt)
        update_cache!(cache,op,μ,t)
        $cfun(a,op,μ,t,xh,cache)
      end
      a
    end

    function $fun(
      ::NonAffinity,
      op::ParamTransientFEOperator,
      solver::ODESolver,
      sols::AbstractArray,
      params::Table,
      cache)

      times = get_times(solver)
      a,c = cache
      pmap(enumerate(params)) do (nμ,μ)
        pmap(enumerate(times)) do (nt,t)
          xh = get_sol(sols,nμ,nt)
          update_cache!(cache,op,μ,t)
          $cfun(a,op,μ,t,xh,cache)
        end
      end
      a
    end
  end
end

function nz_jacobian!(j::Tuple{SparseMatrixCSC,NnzArray},args...)
  js,jnz = j
  jacobian!(js,args...)
  jnz.nonzero_val = compress(js)
  jnz
end
