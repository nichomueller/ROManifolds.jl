function solution_cache(test::FESpace,::FESolver)
  space_ndofs = test.nfree
  cache = fill(0.,space_ndofs,1)
  cache
end

function solution_cache(test::MultiFieldFESpace,args...)
  map(t->solution_cache(t,args...),test.spaces)
end

function collect_solution!(
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

function collect_solution(
  op::ParamFEOperator,
  solver::FESolver,
  μ::AbstractVector)

  cache = solution_cache(op.test,fesolver)
  collect_solution!(cache,op,solver,μ)
  cache
end

function solution_cache(test::FESpace,solver::ODESolver)
  space_ndofs = test.nfree
  time_ndofs = get_time_ndofs(solver)
  cache = fill(0.,space_ndofs,time_ndofs)
  cache
end

function solution_cache(test::MultiFieldFESpace,solver::ODESolver)
  map(t->solution_cache(t,solver),test.spaces)
end

function collect_solution!(
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

# function residuals_cache(assem::SparseMatrixAssembler,vecdata)
#   d = first(vecdata)
#   vec = allocate_vector(assem,d)
#   vec
# end

# function jacobians_cache(assem::SparseMatrixAssembler,matdata)
#   d = first(matdata)
#   mat = allocate_matrix(assem,d)
#   mat_nnz = compress(mat)
#   mat,mat_nnz
# end

# function collect_residuals(assem::SparseMatrixAssembler,vecdata)
#   cache = residuals_cache(assem,vecdata)
#   numeric_loop_vector!(cache,assem,vecdata)
#   cache
# end

# function collect_jacobians(assem::SparseMatrixAssembler,matdata)
#   mat,mat_nnz = jacobians_cache(feop.assem,data)
#   numeric_loop_matrix!(mat,assem,matdata)
#   nnz_i,nnz_v = compress_array(mat)
#   mat_nnz.nonzero_val = nnz_v
#   mat_nnz.nonzero_idx = nnz_i
#   mat_nnz
# end

_get_nsnaps(args...) = 1

_get_nsnaps(::Union{TimeAffinity,NonAffinity},params::Table) = length(params)

for (Top,Tslv) in zip(
  (:ParamFEOperator,:ParamTransientFEOperator),
  (:FESolver,:ODESolver))

  @eval begin
    function collect_solution(
      op::$Top,
      solver::$Tslv,
      params::Table)

      aff = NonAffinity()
      nsol = _get_nsnaps(aff,params)
      printstyled("Generating $nsol solution snapshots\n";color=:blue)

      s = solution_cache(op.test,fesolver)
      collect_solution!(s,op,solver,μ)
      s
    end

    function collect_residuals(
      op::$Top,
      solver::$Tslv,
      sols::AbstractArray,
      params::Table)

      cache = allocate_cache(op)
      aff =  affinity_residual(op,solver,sols,params)
      nres = _get_nsnaps(aff,params)

      printstyled("Generating $nress residual snapshots, affinity: $aff\n";color=:blue)

      r = allocate_residual(op,u0,cache)
      residuals!(aff,r,op,solver,sols,params,cache)
      Snapshots(aff,r,nres)
    end

    function collect_jacobians(
      op::$Top,
      solver::$Tslv,
      sols::AbstractArray,
      params::Table)

      cache = allocate_cache(op)
      aff =  affinity_jacobian(op,solver,sols,params)
      njacs = _get_nsnaps(aff,params)

      j = allocate_jacobian(op,u0,cache)
      jacobians!(aff,j,op,solver,sols,params,cache)
      Snapshots(aff,j,njacs)
    end
  end
end


for (fun,cfun) in zip((:residuals!,:jacobians!),(:residual!,:nz_jacobian!))
  @eval begin
    function $fun(
      ::Union{ZeroAffinity,ParamAffinity},
      a::AbstractArray,
      op::ParamFEOperator,
      ::FESolver,
      sols::AbstractArray,
      params::Table,
      cache)

      μ = first(params)
      xh = get_sol(sols,nμ)
      $cfun(a,op,μ,t,xh,cache)
      a
    end

    function $fun(
      ::NonAffinity,
      a::AbstractArray,
      op::ParamFEOperator,
      ::FESolver,
      sols::AbstractArray,
      params::Table,
      cache)

      pmap(enumerate(params)) do (nμ,μ)
        xh = get_sol(sols,nμ)
        update_cache!(cache,op,μ)
        $cfun(a,op,μ,t,xh,cache)
      end
      a
    end

    function $fun(
      ::Union{ZeroAffinity,ParamTimeAffinity},
      a::AbstractArray,
      op::ParamTransientFEOperator,
      solver::ODESolver,
      sols::AbstractArray,
      params::Table,
      cache)

      μ = first(params)
      t = first(get_times(solver))
      xh = get_sol(sols,nμ)
      $cfun(a,op,μ,t,xh,cache)
      a
    end

    function $fun(
      ::ParamAffinity,
      a::AbstractArray,
      op::ParamTransientFEOperator,
      solver::ODESolver,
      sols::AbstractArray,
      params::Table,
      cache)

      times = get_times(solver)
      μ = first(params)
      pmap(enumerate(times)) do (nt,t)
        xh = get_sol(sols,nμ,nt)
        update_cache!(cache,op,μ,t)
        $cfun(a,op,μ,t,xh,cache)
      end
      a
    end

    function $fun(
      ::TimeAffinity,
      a::AbstractArray,
      op::ParamTransientFEOperator,
      solver::ODESolver,
      sols::AbstractArray,
      params::Table,
      cache)

      t = first(get_times(solver))
      pmap(enumerate(params)) do (nμ,μ)
        xh = get_sol(sols,nμ,nt)
        update_cache!(cache,op,μ,t)
        $cfun(a,op,μ,t,xh,cache)
      end
      a
    end

    function $fun(
      ::NonAffinity,
      a::AbstractArray,
      op::ParamTransientFEOperator,
      solver::ODESolver,
      sols::AbstractArray,
      params::Table,
      cache)

      times = get_times(solver)
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
