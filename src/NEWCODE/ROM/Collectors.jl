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
      params::Table;
      kwargs...)

      aff = NonAffinity()
      nsol = _get_nsnaps(aff,params)
      printstyled("Generating $nsol solution snapshots\n";color=:blue)

      s = allocate_solution(op.test,solver)
      sol = solutions(s,op,solver,params)
      $Tsnp(aff,sol,nsol;kwargs...)
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

      res_iter = init_vec_iterator(op,solver,args...)
      res = residuals(aff,op,solver,res_iter,sols,params)
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

      jac_iter = init_mat_iterator(op,solver,args...)
      jac = jacobians(aff,op,solver,jac_iter,sols,params)
      $Tsnp(aff,jac,njac)
    end

    function collect_djacobians(
      op::$Top,
      solver::$Tslv,
      sols::$Tsnp,
      params::Table,
      args...)

      aff = affinity_jacobian(op,solver,params,args...;i=2)
      ndjac = _get_nsnaps(aff,params)

      printstyled("Generating $ndjac djacobian/dt snapshots, affinity: $aff\n";color=:blue)

      djac_iter = init_mat_iterator(op,solver,args...;i=2)
      djac = djacobians(aff,op,solver,djac_iter,sols,params)
      $Tsnp(aff,djac,ndjac)
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

function init_vec_iterator(
  op::ParamFEOperator,
  ::FESolver,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  args...)

  row,_ = filter
  test_row = get_test(op)[row]
  dv_row = _get_fe_basis(op.test,row)
  assem_row = SparseMatrixAssembler(test_row,test_row)
  r = allocate_residual(op;assem=assem_row)

  function f(
    xh::AbstractVector,
    μ::AbstractVector,
    cache)

    u = evaluation_function(op,xh,cache)
    vecdata = collect_cell_vector(test_row,op.res(μ,u,dv_row,args...),trian)
    rnew = copy(r)
    assemble_vector_add!(rnew,assem_row,vecdata)
  end

  xh = zeros(op.test.nfree)
  μ = realization(op)
  cache = allocate_cache(op)

  IterativeVecCollector(f,xh,μ,cache)
end

function init_vec_iterator(
  op::ParamTransientFEOperator,
  ::θMethod,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  args...)

  row,_ = filter
  test_row = get_test(op)[row]
  dv_row = _get_fe_basis(op.test,row)
  assem_row = SparseMatrixAssembler(test_row,test_row)
  r = allocate_residual(op;assem=assem_row)

  function f(
    xh::Tuple{Vararg{AbstractVector}},
    μ::AbstractVector,
    t::Real,
    cache)

    u = evaluation_function(op,xh,cache)
    vecdata = collect_cell_vector(test_row,op.res(μ,t,u,dv_row,args...),trian)
    rnew = copy(r)
    assemble_vector_add!(rnew,assem_row,vecdata)
  end

  μ = realization(op)
  t = 0.
  xhθ = zeros(op.test.nfree)
  xh0 = copy(xhθ)
  cache = allocate_cache(op)

  TransientIterativeVecCollector(f,(xhθ,xh0),μ,t,cache)
end

function init_mat_iterator(
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
  j = allocate_jacobian(op;assem=assem_row_col)
  jnnz = compress(j)

  function f(
    xh::AbstractVector,
    μ::AbstractVector,
    cache)

    trial, = cache[1]
    trial_col = trial[col]
    u = evaluation_function(op,xh,cache)
    matdata = collect_cell_matrix(trial_col,test_row,
      op.jac(μ,u,du_col,dv_row,args...),trian)
    jnew = copy(j)
    jnew = assemble_matrix_add!(jnew,assem_row_col,matdata)
    nnz_i,nnz_j = compress_array(jnew)
    jnnznew = copy(jnnz)
    jnnznew.nonzero_val = nnz_j
    jnnznew.nonzero_idx = nnz_i
    jnnznew
  end

  xh = zeros(op.test.nfree)
  μ = realization(op)
  cache = allocate_cache(op)

  IterativeVecCollector(f,xh,μ,cache)
end

function init_mat_iterator(
  op::ParamTransientFEOperator,
  solver::θMethod,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  args...;
  i::Int=1)

  row,col = filter
  test_row = op.test[row]
  _trial = get_trial(op)(nothing,nothing)
  _trial_col = _trial[col]
  dv_row = _get_fe_basis(op.test,row)
  du_col = _get_trial_fe_basis(_trial,col)
  assem_row_col = SparseMatrixAssembler(_trial_col,test_row)
  j = allocate_jacobian(op;assem=assem_row_col)
  jnnz = compress(j)
  γ = (1.0,1/(solver.dt*solver.θ))

  function f(
    xh::Tuple{Vararg{AbstractVector}},
    μ::AbstractVector,
    t::Float,
    cache)

    trial, = cache[1]
    trial_col = trial[col]
    u = evaluation_function(op,xh,cache)
    u_col = filter_evaluation_function(u,col)
    matdata = collect_cell_matrix(trial_col,test_row,
      γ[i]*op.jacs[i](μ,t,u_col,du_col,dv_row,args...),trian)
    jnew = copy(j)
    jnew = assemble_matrix_add!(jnew,assem_row_col,matdata)
    nnz_i,nnz_j = compress_array(jnew)
    jnnznew = copy(jnnz)
    jnnznew.nonzero_val = nnz_j
    jnnznew.nonzero_idx = nnz_i
    jnnznew
  end

  μ = realization(op)
  t = 0.
  xhθ = zeros(op.test.nfree)
  xh0 = copy(xhθ)
  cache = allocate_cache(op)
  TransientIterativeMatCollector(f,(xhθ,xh0),μ,t,cache)
end

function update!(
  itc::IterativeCollector,
  op::ParamFEOperator,
  ::FESolver,
  μ::AbstractVector,
  xh=itc.xh)

  itc.μ = μ
  itc.xh = xh
  cache = itc.cache
  update_cache!(cache,op,μ)
  itc.cache = cache
  return
end

function update!(
  itc::TransientIterativeCollector,
  op::ParamTransientFEOperator,
  solver::θMethod,
  μ::AbstractVector,
  t::Real,
  xhθ=get_free_dof_values(solver.uh0(μ)))

  itc.μ = μ
  itc.t = t
  xh0 = copy(xhθ)
  itc.xh = (xhθ,xh0)
  cache = itc.cache
  update_cache!(cache,op,μ,t)
  itc.cache = cache
  return
end

function evaluate!(itc::IterativeCollector)
  itc.f(itc.xh,itc.μ,itc.cache)
end

function evaluate!(itc::TransientIterativeCollector)
  itc.f(itc.xh,itc.μ,itc.t,itc.cache)
end

for fun in (:residuals,:jacobians)
  @eval begin
    function $fun(
      ::Union{ZeroAffinity,ParamAffinity},
      op::ParamFEOperator,
      ::FESolver,
      itc::IterativeCollector,
      ::Snapshots,
      params::Table)

      μ = first(params)
      update!(itc,op,solver,μ)
      evaluate!(itc)
    end

    function $fun(
      ::NonAffinity,
      op::ParamFEOperator{Affine},
      solver::FESolver,
      itc::IterativeCollector,
      ::Snapshots,
      params::Table)

      pmap(enumerate(params)) do (nμ,μ)
        update!(itc,op,solver,μ)
        evaluate!(itc)
      end
    end

    function $fun(
      ::NonAffinity,
      op::ParamFEOperator,
      solver::FESolver,
      itc::IterativeCollector,
      s::Snapshots,
      params::Table)

      sols = get_datum(s)

      pmap(enumerate(params)) do (nμ,μ)
        xh = sols[:,nμ]
        update!(itc,op,solver,μ,xh)
        evaluate!(itc)
      end
    end
  end
end

for fun in (:residuals,:jacobians,:djacobians)
  @eval begin
    function $fun(
      ::Union{ZeroAffinity,ParamTimeAffinity},
      op::ParamTransientFEOperator,
      solver::θMethod,
      itc::TransientIterativeCollector,
      ::TransientSnapshots,
      params::Table)

      μ = first(params)
      t = first(get_times(solver))
      update!(itc,op,solver,μ,t)
      evaluate!(itc)
    end

    function $fun(
      ::ParamAffinity,
      op::ParamTransientFEOperator,
      solver::θMethod,
      itc::TransientIterativeCollector,
      ::TransientSnapshots,
      params::Table)

      times = get_times(solver)
      μ = first(params)
      map(times) do t
        update!(itc,op,solver,μ,t)
        evaluate!(itc)
      end
    end

    function $fun(
      ::TimeAffinity,
      op::ParamTransientFEOperator,
      solver::θMethod,
      itc::TransientIterativeCollector,
      ::TransientSnapshots,
      params::Table)

      t = first(get_times(solver))
      pmap(enumerate(params)) do (nμ,μ)
        update!(itc,op,solver,μ,t)
        evaluate!(itc)
      end
    end

    function $fun(
      ::NonAffinity,
      op::ParamTransientFEOperator{Affine},
      solver::θMethod,
      itc::TransientIterativeCollector,
      ::TransientSnapshots,
      params::Table)

      times = get_times(solver)
      tdofs = length(times)
      pmap(enumerate(params)) do (nμ,μ)
        qt = map(times) do t
          update!(itc,op,solver,μ,t)
          evaluate!(itc)
        end
        hcat(qt...)
      end
    end

    function $fun(
      ::NonAffinity,
      op::ParamTransientFEOperator,
      solver::θMethod,
      itc::TransientIterativeCollector,
      s::TransientSnapshots,
      params::Table)

      times = get_times(solver)
      tdofs = length(times)
      sols = get_datum(s)

      pmap(enumerate(params)) do (nμ,μ)
        function _snaps_μ_tθ(sols,μ,nμ)
          θ = solver.θ
          ic = solver.uh0(μ)
          ich = get_free_dof_values(ic)
          idx_nμ = (nμ-1)*tdofs+1:nμ*tdofs
          prev_sols = hcat(ich,sols[:,idx_nμ[1:end-1]])
          θ*sols[:,idx_nμ] + (1-θ)*prev_sols
        end

        ic = solver.uh0(μ)
        ich = get_free_dof_values(ic)
        idx_nμ = (nμ-1)*tdofs+1:nμ*tdofs
        sols_μθ = _snaps_μ_tθ(sols,μ,nμ)
        qt = map(enumerate(times)) do (nt,t)
          xhθ = sols_μθ[:,nt]
          update!(itc,op,solver,μ,t,xhθ)
          evaluate!(itc)
        end
        hcat(qt...)
      end
    end
  end
end
